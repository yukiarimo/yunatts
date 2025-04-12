import os
import torch
import numpy as np
import librosa
import soundfile as sf

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """
    Convert audio waveform to mel spectrogram

    Args:
        y: Audio waveform [B, T]
        n_fft: FFT size
        sampling_rate: Audio sampling rate
        hop_size: Hop size
        win_size: Window size
        center: Whether to pad for centered FFT

    Returns:
        Mel spectrogram [B, n_fft//2+1, T']
    """
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    Convert audio waveform to mel spectrogram

    Args:
        y: Audio waveform [B, T]
        n_fft: FFT size
        num_mels: Number of mel bands
        sampling_rate: Audio sampling rate
        hop_size: Hop size
        win_size: Window size
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Whether to pad for centered FFT

    Returns:
        Mel spectrogram [B, num_mels, T']
    """
    # Get spectrogram
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center)

    # Initialize mel filterbank if needed
    global mel_basis
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)

    # Apply mel filterbank
    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)

    # Convert to log scale
    melspec = spectral_normalize_torch(melspec)

    return melspec

def spectral_normalize_torch(magnitudes):
    """
    Normalize spectrogram to log scale
    """
    output = torch.log(torch.clamp(magnitudes, min=1e-5))
    return output

def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    """
    Load audio file to torch tensor

    Args:
        full_path: Path to audio file
        target_sr: Target sample rate (if None, keep original)
        return_empty_on_exception: Whether to return empty tensor on exception

    Returns:
        Audio tensor [C, T], sample rate
    """
    try:
        # Load audio with librosa (handles resampling and various formats)
        audio, sr = librosa.load(full_path, sr=target_sr, mono=False)

        # Handle mono vs stereo
        if audio.ndim == 1:
            # Mono audio, add channel dimension
            audio = np.expand_dims(audio, 0)
        elif audio.ndim > 2:
            # Unexpected shape, flatten to stereo
            audio = audio[:2]

        # Convert to torch tensor
        audio = torch.FloatTensor(audio)

        return audio, sr

    except Exception as e:
        print(f"Error loading audio file {full_path}: {e}")
        if return_empty_on_exception:
            return torch.zeros(2, 0), target_sr if target_sr is not None else 44100
        else:
            raise e

def save_audio(path, audio, sample_rate):
    """
    Save audio tensor to file

    Args:
        path: Output file path
        audio: Audio tensor [C, T] or [T, C]
        sample_rate: Audio sample rate
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Convert to numpy
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    # Handle shape
    if audio.ndim == 1:
        # Mono audio, add channel dimension
        audio = np.expand_dims(audio, 0)

    # Ensure shape is [T, C] for soundfile
    if audio.shape[0] <= 2 and audio.shape[1] > 2:
        # Shape is [C, T], transpose to [T, C]
        audio = audio.T

    # Save audio
    sf.write(path, audio, sample_rate)

# Global variables for caching
hann_window = {}
mel_basis = {}