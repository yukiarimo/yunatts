import torch
import numpy as np
import librosa
import soundfile as sf
from librosa.filters import mel as librosa_mel_fn

def load_wav(full_path, sr=None):
    """
    Load a wav file with librosa

    Args:
        full_path (str): Path to wav file
        sr (int, optional): Target sampling rate

    Returns:
        numpy array: Audio data
    """
    return librosa.load(full_path, sr=sr)[0]

def load_wav_to_torch(full_path, sr=None):
    """
    Load a wav file with librosa and convert to torch tensor

    Args:
        full_path (str): Path to wav file
        sr (int, optional): Target sampling rate

    Returns:
        torch tensor: Audio data
        int: Sampling rate
    """
    audio, sampling_rate = librosa.load(full_path, sr=sr)
    return torch.FloatTensor(audio), sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    Dynamic range compression for audio

    Args:
        x (tensor): Input tensor
        C (float): Compression factor
        clip_val (float): Clipping value

    Returns:
        tensor: Compressed tensor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    """
    Dynamic range decompression for audio

    Args:
        x (tensor): Input tensor
        C (float): Compression factor

    Returns:
        tensor: Decompressed tensor
    """
    return torch.exp(x) / C

def spectral_normalize(magnitudes):
    """
    Normalize spectrogram

    Args:
        magnitudes (tensor): Input spectrogram

    Returns:
        tensor: Normalized spectrogram
    """
    return dynamic_range_compression(magnitudes)

def spectral_de_normalize(magnitudes):
    """
    Denormalize spectrogram

    Args:
        magnitudes (tensor): Input spectrogram

    Returns:
        tensor: Denormalized spectrogram
    """
    return dynamic_range_decompression(magnitudes)

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    Generate mel spectrogram from audio

    Args:
        y (tensor): Audio tensor
        n_fft (int): FFT size
        num_mels (int): Number of mel bands
        sampling_rate (int): Sampling rate
        hop_size (int): Hop size
        win_size (int): Window size
        fmin (float): Minimum frequency
        fmax (float): Maximum frequency
        center (bool): Center the STFT

    Returns:
        tensor: Mel spectrogram
    """
    # Map to device of input
    device = y.device

    # Create mel basis if it doesn't exist
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # Get the correct mel basis and window
    mel_basis_key = str(fmax)+'_'+str(y.device)
    if mel_basis_key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)
    if str(y.device) not in hann_window:
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # Compute spectrogram
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # Complex tensor for storing the STFT result
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    # Convert to magnitude
    spec = torch.abs(spec)

    # Convert to mel scale
    spec = torch.matmul(mel_basis[mel_basis_key], spec)

    # Apply dynamic range compression
    spec = spectral_normalize(spec)

    return spec

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """
    Generate spectrogram from audio

    Args:
        y (tensor): Audio tensor
        n_fft (int): FFT size
        sampling_rate (int): Sampling rate
        hop_size (int): Hop size
        win_size (int): Window size
        center (bool): Center the STFT

    Returns:
        tensor: Spectrogram
    """
    # Map to device of input
    device = y.device

    # Create window if it doesn't exist
    global hann_window
    if str(y.device) not in hann_window:
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # Compute spectrogram
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # Complex tensor for storing the STFT result
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

    # Convert to magnitude
    spec = torch.abs(spec)

    return spec

def convert_to_mono(audio):
    """
    Convert stereo audio to mono

    Args:
        audio (numpy array): Audio data

    Returns:
        numpy array: Mono audio data
    """
    if len(audio.shape) > 1:
        return np.mean(audio, axis=0)
    return audio

def normalize_audio(audio, target_level=-27):
    """
    Normalize audio to target level

    Args:
        audio (numpy array): Audio data
        target_level (float): Target level in dB

    Returns:
        numpy array: Normalized audio data
    """
    # Calculate current level
    rms = np.sqrt(np.mean(audio**2))
    current_level = 20 * np.log10(rms) if rms > 0 else -100

    # Calculate gain
    gain = 10**((target_level - current_level) / 20)

    # Apply gain
    return audio * gain

def resample_audio(audio, orig_sr, target_sr):
    """
    Resample audio to target sampling rate

    Args:
        audio (numpy array): Audio data
        orig_sr (int): Original sampling rate
        target_sr (int): Target sampling rate

    Returns:
        numpy array: Resampled audio data
    """
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def save_audio(audio, path, sr=48000):
    """
    Save audio to file

    Args:
        audio (numpy array): Audio data
        path (str): Output path
        sr (int): Sampling rate
    """
    sf.write(path, audio, sr)

def get_mel_from_file(file_path, hps):
    """
    Generate mel spectrogram from audio file

    Args:
        file_path (str): Path to audio file
        hps (HParams): Hyperparameters

    Returns:
        tensor: Mel spectrogram
    """
    audio, sr = load_wav_to_torch(file_path, sr=hps.data.sampling_rate)

    # Handle stereo by converting to mono for mel extraction
    if len(audio.shape) > 1:
        audio = torch.mean(audio, dim=0)

    audio = audio.unsqueeze(0)
    mel = mel_spectrogram(
        audio, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length, 
        hps.data.mel_fmin, 
        hps.data.mel_fmax
    )
    return mel[0]
