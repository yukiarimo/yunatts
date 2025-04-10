import torch
from librosa.filters import mel as librosa_mel_fn
MAX_WAV_VALUE = 32768.0

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor

    Note: Function modified to bypass compression
    """
    # Return input directly without logarithmic compression
    return torch.clamp(x, min=clip_val)

def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress

    Note: Function modified to bypass decompression
    """
    # Return input directly
    return x

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """
    Compute spectrogram for audio input (mono or stereo)

    Args:
        y: Audio input tensor of shape [batch_size, channels, time] or [batch_size, time]
        n_fft: FFT size
        sampling_rate: Audio sampling rate
        hop_size: Hop size
        win_size: Window size
        center: Whether to pad on both sides

    Returns:
        Spectrogram tensor of shape [batch_size, channels, n_fft//2+1, time] or [batch_size, n_fft//2+1, time]
    """
    # Check if input is stereo (has channel dimension)
    is_stereo = y.dim() == 3

    if torch.min(y) < -1.1:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.1:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    if is_stereo:
        # Process each channel separately
        batch_size, channels, time = y.shape
        specs = []

        for c in range(channels):
            channel_y = y[:, c, :]

            # Pad
            padded_y = torch.nn.functional.pad(
                channel_y.unsqueeze(1),
                (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                mode="reflect",
            )
            padded_y = padded_y.squeeze(1)

            # Compute STFT
            spec = torch.stft(
                padded_y,
                n_fft,
                hop_length=hop_size,
                win_length=win_size,
                window=hann_window[wnsize_dtype_device],
                center=center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=False,
            )

            # Convert to magnitude
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
            specs.append(spec)

        # Stack along channel dimension
        return torch.stack(specs, dim=1)
    else:
        # Original mono processing
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec

def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    """
    Convert spectrogram to mel spectrogram (supports both mono and stereo)

    Args:
        spec: Spectrogram tensor of shape [batch_size, channels, n_fft//2+1, time] or [batch_size, n_fft//2+1, time]
        n_fft: FFT size
        num_mels: Number of mel bands
        sampling_rate: Audio sampling rate
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel spectrogram tensor
    """
    # Check if input is stereo (has channel dimension)
    is_stereo = spec.dim() == 4

    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )

    if is_stereo:
        # Process each channel separately
        batch_size, channels, freq, time = spec.shape
        mel_specs = []

        for c in range(channels):
            channel_spec = spec[:, c]
            mel_spec = torch.matmul(mel_basis[fmax_dtype_device], channel_spec)
            mel_spec = spectral_normalize_torch(mel_spec)
            mel_specs.append(mel_spec)

        # Stack along channel dimension
        return torch.stack(mel_specs, dim=1)
    else:
        # Original mono processing
        spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
        spec = spectral_normalize_torch(spec)
        return spec

def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """
    Compute mel spectrogram for audio input (mono or stereo)

    Args:
        y: Audio input tensor of shape [batch_size, channels, time] or [batch_size, time]
        n_fft: FFT size
        num_mels: Number of mel bands
        sampling_rate: Audio sampling rate
        hop_size: Hop size
        win_size: Window size
        fmin: Minimum frequency
        fmax: Maximum frequency
        center: Whether to pad on both sides

    Returns:
        Mel spectrogram tensor
    """
    # Check if input is stereo (has channel dimension)
    is_stereo = y.dim() == 3

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=y.dtype, device=y.device
        )
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    if is_stereo:
        # Process each channel separately
        batch_size, channels, time = y.shape
        mel_specs = []

        for c in range(channels):
            channel_y = y[:, c, :]

            # Pad
            padded_y = torch.nn.functional.pad(
                channel_y.unsqueeze(1),
                (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                mode="reflect",
            )
            padded_y = padded_y.squeeze(1)

            # Compute STFT
            spec = torch.stft(
                padded_y,
                n_fft,
                hop_length=hop_size,
                win_length=win_size,
                window=hann_window[wnsize_dtype_device],
                center=center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=False,
            )

            # Convert to magnitude
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

            # Convert to mel
            mel_spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
            mel_spec = spectral_normalize_torch(mel_spec)
            mel_specs.append(mel_spec)

        # Stack along channel dimension
        return torch.stack(mel_specs, dim=1)
    else:
        # Original mono processing
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
        spec = spectral_normalize_torch(spec)

        return spec
