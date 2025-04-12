import torch
import numpy as np
import librosa
import soundfile as sf
from hanasuconverter.models import SynthesizerTrn
from hanasuconverter.audio import mel_spectrogram

class VoiceConverter:
    """
    Hanasu Voice Converter - Core voice conversion functionality
    Implements retrieval-based voice conversion (RVC-style)
    """
    def __init__(self, config_path, device=None):
        # Auto-detect device with MPS priority for Mac
        if device is None:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                print("Using MPS (Apple Silicon)")
            elif torch.cuda.is_available():
                device = 'cuda:0'
                print("Using CUDA")
            else:
                device = 'cpu'
                print("Using CPU")

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
            n_speakers=2,  # Set n_speakers=2 for inference
            gin_channels=self.hps.model.gin_channels,
            use_external_speaker_embedding=False  # Set True for inference
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
        if a or b:
            print(f"Missing keys: {a}")
            print(f"Unexpected keys: {b}")

    def extract_speaker_embedding(self, audio_path_list, se_save_path=None):
        """
        Extract speaker embedding from audio files
        Optimized for high-pitched anime voices

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

            # Handle stereo audio
            if audio.ndim > 1 and audio.shape[0] > 1:
                # Keep stereo for processing
                audio_mono = np.mean(audio, axis=0)
            else:
                # If already mono, ensure it has the right shape
                if audio.ndim == 1:
                    audio_mono = audio
                else:
                    audio_mono = audio[0]

            # Convert mono audio to tensor
            y = torch.FloatTensor(audio_mono).to(device)
            y = y.unsqueeze(0)  # Add batch dimension: [1, T_wav]

            # Extract MEL spectrogram
            y_mel = mel_spectrogram(
                y,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
                center=False
            ).to(device)
            # y_mel shape: [1, n_mels, T_frames]

            # Extract speaker embedding using the improved speaker encoder
            with torch.no_grad():
                # Input to speaker_encoder needs [B, T_frames, n_mels]
                g = self.model.speaker_encoder(y_mel.transpose(1, 2))  # Output: [1, gin_channels]
                g = g.unsqueeze(-1)  # Output: [1, gin_channels, 1]
                speaker_embeddings.append(g.detach())

        # Average speaker embeddings if multiple files
        speaker_embedding = torch.stack(speaker_embeddings).mean(0)

        # Save speaker embedding if path is provided
        if se_save_path is not None:
            import os
            output_dir = os.path.dirname(se_save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created directory for embedding: {output_dir}")

            try:
                torch.save(speaker_embedding.cpu(), se_save_path)
                print(f"Saved speaker embedding to {se_save_path}")
            except Exception as e:
                print(f"Error saving speaker embedding to {se_save_path}: {e}")

        return speaker_embedding

    def convert_voice(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.7):
        """
        Convert voice from source to target using retrieval-based approach.
        Preserves content while changing voice characteristics.

        Args:
            audio_src_path: Source audio file path
            src_se: Source speaker embedding tensor [1, gin_channels, 1]
            tgt_se: Target speaker embedding tensor [1, gin_channels, 1]
            output_path: Output audio file path (optional)
            tau: Temperature parameter for conversion

        Returns:
            Converted audio as numpy array if output_path is None,
            otherwise saves audio to file.
        """
        hps = self.hps

        # Ensure embeddings are on the correct device
        src_se = src_se.to(self.device)
        tgt_se = tgt_se.to(self.device)

        # Load source audio with target sample rate, keep stereo if present
        audio, sr = librosa.load(audio_src_path, sr=hps.data.sampling_rate, mono=False)

        # Process audio based on whether it's stereo or mono
        is_stereo_input = (audio.ndim > 1 and audio.shape[0] > 1)

        # Convert to tensor
        audio = torch.FloatTensor(audio).to(self.device)

        # Ensure audio has the right shape
        if is_stereo_input:
            # Already has shape [2, T_wav]
            audio_mono = torch.mean(audio, dim=0, keepdim=True)  # [1, T_wav]
        else:
            # If already mono, ensure it has shape [1, T_wav]
            if audio.dim() == 1:
                audio_mono = audio.unsqueeze(0)  # [1, T_wav]
            else:
                audio_mono = audio  # Already [1, T_wav]

        # Calculate MEL spectrogram from mono audio
        spec = mel_spectrogram(
            audio_mono,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
            center=False
        ).to(self.device)  # Shape [1, n_mels, T_frames]

        spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)

        # Process in chunks if the audio is too long
        max_chunk_length = hps.inference.segment_size // hps.data.hop_length

        if spec.size(-1) > max_chunk_length:
            # Process in chunks
            print(f"Processing long audio in chunks (length: {spec.size(-1)} frames)")
            output_chunks = []

            for start_idx in range(0, spec.size(-1), max_chunk_length):
                end_idx = min(start_idx + max_chunk_length, spec.size(-1))
                spec_chunk = spec[:, :, start_idx:end_idx]
                spec_chunk_lengths = torch.LongTensor([spec_chunk.size(-1)]).to(self.device)

                # Convert chunk
                with torch.no_grad():
                    y_hat_chunk = self.model.voice_conversion(
                        spec_chunk, spec_chunk_lengths, g_src=src_se, g_tgt=tgt_se, tau=tau
                    )

                # Add to output chunks
                output_chunks.append(y_hat_chunk.cpu().numpy())

            # Concatenate chunks
            audio_out = np.concatenate(output_chunks, axis=-1)
        else:
            # Process entire audio at once
            with torch.no_grad():
                y_hat = self.model.voice_conversion(
                    spec, spec_lengths, g_src=src_se, g_tgt=tgt_se, tau=tau
                )
                audio_out = y_hat.cpu().numpy()

        # Save or return audio
        if output_path is None:
            return audio_out  # Return numpy array [B, 2, T_wav_out]
        else:
            # sf.write expects [T_wav, C] for stereo. Transpose the [B, 2, T_wav] array.
            save_data = audio_out[0].T  # [T_wav, 2]
            print(f"Saving audio with shape: {save_data.shape} to {output_path}")
            try:
                sf.write(output_path, save_data, hps.data.sampling_rate)
                print(f"Successfully saved converted audio to {output_path}")
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
            min_silence_len=self.hps.audio.min_silence_duration,
            silence_thresh=self.hps.audio.vad_threshold,
            keep_silence=self.hps.audio.keep_silence_duration
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