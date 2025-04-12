import os
import torch
import numpy as np
from tqdm import tqdm

def get_device():
    """
    Get the optimal device for the current system,
    prioritizing MPS on Mac, then CUDA, then CPU
    """
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'

def get_hparams_from_file(config_path):
    """
    Load hyperparameters from config file
    """
    import json
    from collections import namedtuple

    with open(config_path, 'r') as f:
        data = json.load(f)

    hparams = HParams(**data)
    return hparams

def HParams(**kwargs):
    """
    Simple wrapper for dictionary to access keys as attributes
    """
    class HParamsClass:
        def __init__(self, **entries):
            for key, value in entries.items():
                if isinstance(value, dict):
                    value = HParams(**value)
                self.__dict__[key] = value

        def __repr__(self):
            return str(self.__dict__)

    return HParamsClass(**kwargs)

def save_checkpoint(model, optimizer, hps, iteration, checkpoint_path, scaler=None):
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint_dict = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "config": {k: (v.__dict__ if hasattr(v, '__dict__') else v) 
                   for k, v in hps.__dict__.items()}  # Convert to regular dict
    }

    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()

    torch.save(checkpoint_dict, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None):
    """
    Load model checkpoint
    """
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    iteration = checkpoint_dict.get("iteration", 0)

    # Load model state
    model_dict = checkpoint_dict["model"]
    model.load_state_dict(model_dict, strict=False)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    # Load scaler state if provided
    if scaler is not None and "scaler" in checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler"])

    return model, optimizer, checkpoint_dict.get("config", None), iteration, scaler

def scan_checkpoint(checkpoint_dir, prefix):
    """
    Scan checkpoint directory for the latest checkpoint
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    pattern = prefix + "*"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix)]

    if not checkpoint_files:
        return None

    # Extract iteration numbers from filenames
    iterations = []
    for f in checkpoint_files:
        try:
            # Extract iteration number from filename (e.g., model_00000020.pt -> 20)
            iter_num = int(f.replace(prefix, "").replace(".pt", "").strip())
            iterations.append((iter_num, os.path.join(checkpoint_dir, f)))
        except ValueError:
            continue

    if not iterations:
        return None

    # Return the checkpoint with the highest iteration number
    return max(iterations, key=lambda x: x[0])[1]

def list_audio_files(directory, extensions=['.wav', '.mp3', '.flac', '.ogg']):
    """
    List all audio files in a directory
    """
    audio_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))

    return audio_files

def create_dataset_csv(data_dir, output_csv, speaker_subdirs=True):
    """
    Create a CSV file listing all audio files and their speakers

    Args:
        data_dir: Directory containing audio files
        output_csv: Output CSV file path
        speaker_subdirs: Whether speaker names are subdirectory names
    """
    import csv

    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if speaker_subdirs:
            # Each subdirectory is a speaker
            for speaker_dir in os.listdir(data_dir):
                speaker_path = os.path.join(data_dir, speaker_dir)

                if not os.path.isdir(speaker_path):
                    continue

                # List all audio files for this speaker
                for root, _, files in os.walk(speaker_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in audio_extensions):
                            audio_path = os.path.join(root, file)
                            writer.writerow([audio_path, speaker_dir])
        else:
            # Audio files have speaker information in filename or metadata
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_path = os.path.join(root, file)

                        # Extract speaker from parent directory name
                        speaker = os.path.basename(root)

                        writer.writerow([audio_path, speaker])

    print(f"Created dataset CSV at {output_csv}")

def preprocess_dataset(input_csv, output_dir, sample_rate=48000, normalize=True, target_level=-27):
    """
    Preprocess dataset by resampling, normalizing, and converting to WAV

    Args:
        input_csv: Input CSV file with audio paths and speakers
        output_dir: Output directory for processed files
        sample_rate: Target sample rate
        normalize: Whether to normalize audio
        target_level: Target RMS level in dB for normalization
    """
    import csv
    import soundfile as sf
    from pydub import AudioSegment

    os.makedirs(output_dir, exist_ok=True)

    # Create output CSV
    output_csv = os.path.join(output_dir, 'dataset.csv')

    with open(input_csv, 'r', encoding='utf-8') as f_in, open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        for row in tqdm(reader, desc="Preprocessing dataset"):
            if len(row) < 2:
                continue

            audio_path, speaker = row[0], row[1]

            try:
                # Create speaker directory
                speaker_dir = os.path.join(output_dir, speaker)
                os.makedirs(speaker_dir, exist_ok=True)

                # Output filename
                filename = os.path.basename(audio_path)
                output_filename = os.path.splitext(filename)[0] + '.wav'
                output_path = os.path.join(speaker_dir, output_filename)

                # Load audio
                audio = AudioSegment.from_file(audio_path)

                # Convert to stereo if mono
                if audio.channels == 1:
                    audio = audio.set_channels(2)

                # Resample
                audio = audio.set_frame_rate(sample_rate)

                # Normalize if requested
                if normalize:
                    # Calculate current RMS level
                    rms = audio.rms
                    if rms > 0:
                        current_level = 20 * np.log10(rms)
                        gain = target_level - current_level
                        audio = audio.apply_gain(gain)

                # Export as WAV
                audio.export(output_path, format='wav')

                # Add to output CSV
                writer.writerow([output_path, speaker])

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

    print(f"Preprocessing complete. Output CSV: {output_csv}")

def optimize_for_device(model, device):
    """
    Apply device-specific optimizations to the model
    """
    model = model.to(device)

    # MPS-specific optimizations
    if device == 'mps':
        # Set model to eval mode for inference
        model.eval()

        # Use float32 for MPS as it's more stable than float16
        model = model.to(torch.float32)

        # Optimize model parameters for MPS
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.contiguous()

    return model

def chunk_audio_processing(func, audio, max_chunk_size, *args, **kwargs):
    """
    Process audio in chunks to avoid memory issues

    Args:
        func: Function to apply to each chunk
        audio: Audio tensor or array
        max_chunk_size: Maximum chunk size in samples
        *args, **kwargs: Additional arguments to pass to func

    Returns:
        Concatenated results from all chunks
    """
    # Get audio length
    audio_length = audio.shape[-1]

    # If audio is smaller than max_chunk_size, process directly
    if audio_length <= max_chunk_size:
        return func(audio, *args, **kwargs)

    # Process in chunks
    results = []
    for start_idx in range(0, audio_length, max_chunk_size):
        end_idx = min(start_idx + max_chunk_size, audio_length)

        # Extract chunk
        if isinstance(audio, torch.Tensor):
            chunk = audio[..., start_idx:end_idx]
        else:
            chunk = audio[..., start_idx:end_idx]

        # Process chunk
        result = func(chunk, *args, **kwargs)

        # Store result
        results.append(result)

    # Concatenate results
    if isinstance(results[0], torch.Tensor):
        return torch.cat(results, dim=-1)
    else:
        return np.concatenate(results, axis=-1)