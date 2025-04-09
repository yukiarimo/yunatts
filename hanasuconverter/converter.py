import torch
import numpy as np
import librosa
import soundfile as sf
from hanasuconverter.models import SynthesizerTrn

class VoiceConverter:
    """
    Hanasu Voice Converter - Core voice conversion functionality
    """
    def __init__(self, 
                config_path, 
                device=None):
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        print(f"Using device: {device}")

        # Load configuration
        from hanasuconverter.utils import get_hparams_from_file
        self.hps = get_hparams_from_file(config_path)

        # Initialize model
        model = SynthesizerTrn(
            self.hps.data.n_mel_channels,
            self.hps.model.inter_channels,
            self.hps.model.hidden_channels,
            self.hps.model.filter_channels,
            self.hps.model.resblock,
            self.hps.model.resblock_kernel_sizes,
            self.hps.model.resblock_dilation_sizes,
            self.hps.model.upsample_rates,
            self.hps.model.upsample_initial_channel,
            self.hps.model.upsample_kernel_sizes,
            n_speakers=0,                         # <-- Set n_speakers=0 for inference
            gin_channels=self.hps.model.gin_channels,
            use_external_speaker_embedding=True   # <-- Set True for inference
        ).to(device)

        model.eval()
        self.model = model
        self.version = getattr(self.hps, 'version', "v1.0")

    def load_checkpoint(self, ckpt_path):
        """
        Load model checkpoint
        """
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print(f"Loaded checkpoint '{ckpt_path}'")
        print(f"Missing/unexpected keys: {a}, {b}")

    def extract_speaker_embedding(self, audio_path_list, se_save_path=None):
        """
        Extract speaker embedding from audio files

        Args:
            audio_path_list: List of audio file paths or single audio file path
            se_save_path: Path to save speaker embedding

        Returns:
            Speaker embedding tensor
        """
        if isinstance(audio_path_list, str):
            audio_path_list = [audio_path_list]

        device = self.device
        hps = self.hps
        speaker_embeddings = []

        for fname in audio_path_list:
            # Load audio with correct sample rate
            audio, sr = librosa.load(fname, sr=hps.data.sampling_rate, mono=False)

            # Convert to mono if stereo for embedding extraction
            audio_mono = audio
            if audio.ndim > 1 and audio.shape[0] > 1:
                audio_mono = np.mean(audio, axis=0)
            elif audio.ndim != 1:
                # Handle unexpected shapes if necessary
                 print(f"Warning: Unexpected audio shape {audio.shape} for {fname}, attempting mean.")
                 try:
                      audio_mono = np.mean(audio, axis=0) # Try averaging first dim
                      if audio_mono.ndim != 1: raise ValueError("Still not mono")
                 except Exception as e:
                      print(f"Error: Could not convert {fname} to mono: {e}. Skipping file.")
                      continue

            # Convert mono audio to tensor
            y = torch.FloatTensor(audio_mono).to(device)
            y = y.unsqueeze(0) # Add batch dimension: [1, T_wav]

            # --- Extract MEL spectrogram ---
            from hanasuconverter.audio import mel_spectrogram # <<< CORRECT FUNCTION
            y_mel = mel_spectrogram(
                y,
                hps.data.filter_length,
                hps.data.n_mel_channels, # Use correct number of mel bins
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
                center=False # Keep center=False if consistent with training
            ).to(device)
            # y_mel shape: [1, n_mels, T_frames] = [1, 128, T_frames]
            # -------------------------------

            # Extract speaker embedding
            with torch.no_grad():
                # Input to ref_enc needs [B, T_frames, n_features=n_mels]
                # Transpose y_mel: [1, T_frames, 128]
                g = self.model.ref_enc(y_mel.transpose(1, 2)) # Output: [1, gin_channels]
                # Keep embedding as [B, C] or unsqueeze based on downstream use
                # If WN layers expect [B, C, 1], unsqueeze here. Otherwise, might not need it.
                # Let's assume downstream (like convert_voice) needs [B, C, 1]
                g = g.unsqueeze(-1) # Output: [1, gin_channels, 1]
                speaker_embeddings.append(g.detach()) # Store [1, gin_channels, 1] tensors

        # Average speaker embeddings if multiple files
        speaker_embedding = torch.stack(speaker_embeddings).mean(0)

        # Save speaker embedding if path is provided
        if se_save_path is not None:
            import os
            # --- Create output directory only if necessary ---
            output_dir = os.path.dirname(se_save_path)
            if output_dir and not os.path.exists(output_dir): # Check if dirname is not empty and doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created directory for embedding: {output_dir}")
            # -----------------------------------------------
            try:
                 torch.save(speaker_embedding.cpu(), se_save_path)
                 # Don't print success here, let the calling function handle it
            except Exception as e:
                 print(f"Error saving speaker embedding to {se_save_path}: {e}")
                 # Optionally re-raise the exception or return None/False

        return speaker_embedding

    def convert_voice(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3):
        from hanasuconverter.audio import mel_spectrogram
        """
        Convert voice from source to target using MEL spectrograms.

        Args:
            audio_src_path: Source audio file path
            src_se: Source speaker embedding tensor [1, gin_channels, 1]
            tgt_se: Target speaker embedding tensor [1, gin_channels, 1]
            output_path: Output audio file path (optional)
            tau: Temperature parameter for posterior encoder sampling

        Returns:
            Converted audio as numpy array [C, T_wav] if output_path is None,
            otherwise saves stereo audio to file.
        """
        hps = self.hps
        # Ensure embeddings are on the correct device
        src_se = src_se.to(self.device)
        tgt_se = tgt_se.to(self.device)

        # Load source audio with target sample rate, keep channels
        audio, sr = librosa.load(audio_src_path, sr=hps.data.sampling_rate, mono=False)
        audio = torch.FloatTensor(audio).to(self.device)

        # Determine if input was stereo
        is_stereo_input = audio.dim() > 1 and audio.shape[0] == 2

        if is_stereo_input:
            print("Processing stereo input...")
            # Process each channel separately using its own mel spectrogram
            audio_left = audio[0].unsqueeze(0)  # Shape [1, T_wav]
            audio_right = audio[1].unsqueeze(0) # Shape [1, T_wav]
            audio_out_channels = []

            # Process left channel
            with torch.no_grad():
                # Calculate MEL spectrogram
                spec_left = mel_spectrogram(
                    audio_left,
                    hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax, center=False
                ).to(self.device) # Shape [1, 128, T_frames]
                spec_lengths_left = torch.LongTensor([spec_left.size(-1)]).to(self.device)

                # Perform conversion
                # Model returns y_hat of shape [1, 2, T_wav_out]
                y_hat_left = self.model.voice_conversion(
                    spec_left, spec_lengths_left, g_src=src_se, g_tgt=tgt_se, tau=tau
                )
                # Extract the full sequence for the first batch item, first channel
                audio_out_left = y_hat_left[0, 0].data.cpu().float().numpy() # Shape [T_wav_out]
                audio_out_channels.append(audio_out_left)

            # Process right channel
            with torch.no_grad():
                 # Calculate MEL spectrogram
                spec_right = mel_spectrogram(
                    audio_right,
                    hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax, center=False
                ).to(self.device) # Shape [1, 128, T_frames]
                spec_lengths_right = torch.LongTensor([spec_right.size(-1)]).to(self.device)

                # Perform conversion
                y_hat_right = self.model.voice_conversion(
                    spec_right, spec_lengths_right, g_src=src_se, g_tgt=tgt_se, tau=tau
                )
                # Extract the full sequence for the first batch item, second channel
                audio_out_right = y_hat_right[0, 1].data.cpu().float().numpy() # Shape [T_wav_out]
                audio_out_channels.append(audio_out_right)

            # Combine processed channels back into stereo [2, T_wav_out]
            audio_out = np.stack(audio_out_channels)

        else:
            # Process mono input audio
            print("Processing mono input...")
            if audio.dim() == 2 and audio.shape[0] == 1: # If shape [1, T_wav]
                 audio_mono = audio
            elif audio.dim() == 1: # If shape [T_wav]
                 audio_mono = audio.unsqueeze(0) # Add batch dim -> [1, T_wav]
            else:
                 # Handle unexpected mono shapes if necessary
                 print(f"Warning: Unexpected mono audio shape {audio.shape}. Attempting to use first channel or mean.")
                 audio_mono = audio[0].unsqueeze(0) if audio.dim() > 1 else audio.unsqueeze(0)

            with torch.no_grad():
                 # Calculate MEL spectrogram for the mono input
                spec = mel_spectrogram(
                    audio_mono,
                    hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax, center=False
                ).to(self.device) # Shape [1, 128, T_frames]
                spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)

                # Perform conversion (model generates stereo output: [1, 2, T_wav_out])
                y_hat = self.model.voice_conversion(
                    spec, spec_lengths, g_src=src_se, g_tgt=tgt_se, tau=tau
                )
                # Extract the full stereo sequence for the first batch item
                audio_out = y_hat[0].data.cpu().float().numpy() # Shape [2, T_wav_out]

        # Save or return audio
        if output_path is None:
            return audio_out # Return numpy array [2, T_wav_out]
        else:
            # sf.write expects [T_wav, C] for stereo. Transpose the [2, T_wav] array.
            save_data = audio_out.T
            print(f"Saving audio with shape: {save_data.shape} to {output_path}")
            try:
                sf.write(output_path, save_data, hps.data.sampling_rate)
                # Success message printed by the calling function in inference.py
            except Exception as e:
                print(f"Error saving converted audio to {output_path}: {e}")

    def segment_audio(self, audio_path, output_dir, segment_seconds=10.0):
        """
        Segment audio file into smaller chunks for processing

        Args:
            audio_path: Audio file path
            output_dir: Output directory for segments
            segment_seconds: Length of each segment in seconds

        Returns:
            List of segment file paths
        """
        import os
        from pydub import AudioSegment
        from pydub.silence import split_on_silence

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load audio
        audio = AudioSegment.from_file(audio_path)

        # Detect silence for VAD
        segments = split_on_silence(
            audio,
            min_silence_len=500,  # minimum silence length in ms
            silence_thresh=-40,   # silence threshold in dB
            keep_silence=200      # keep 200ms of silence at the beginning and end
        )

        # If no segments found (continuous speech), use fixed-length segments
        if len(segments) == 0:
            segments = [audio[i:i+int(segment_seconds*1000)] 
                       for i in range(0, len(audio), int(segment_seconds*1000))]

        # Save segments
        segment_paths = []
        for i, segment in enumerate(segments):
            segment_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
            segment.export(segment_path, format="wav")
            segment_paths.append(segment_path)

        return segment_paths
