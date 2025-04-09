import json
import numpy as np
import torch
import os
import glob
import csv # Add csv import
import librosa # Add librosa import
import soundfile as sf # Add soundfile import
from hanasuconverter.audio import normalize_audio # Import normalize_audio
from tqdm import tqdm # Import tqdm for progress bar

def get_hparams_from_file(config_path):
    """
    Load hyperparameters from a JSON file

    Args:
        config_path (str): Path to config file

    Returns:
        HParams: Hyperparameters object
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = f.read()
        config = json.loads(data)
        hparams = HParams(**config)
        return hparams
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON config file at {config_path}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred loading config {config_path}: {e}")
        raise

class HParams:
    """
    Hyperparameters class for storing configuration as attributes.
    Allows nested access like `hps.train.batch_size`.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Recursively convert dicts to HParams during setting
        if isinstance(value, dict):
             value = HParams(**value)
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        # Pretty print the HParams object
        items = (f"{k}={v!r}" for k, v in self.__dict__.items())
        return "{}(\n  {}\n)".format(type(self).__name__, "\n  ".join(items))

    def to_dict(self):
        # Convert HParams object back to a dictionary
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, HParams):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, scaler=None): # Add scaler argument
    """
    Save model checkpoint

    Args:
        model: PyTorch model (potentially DDP wrapped)
        optimizer: PyTorch optimizer
        learning_rate: Current learning rate
        iteration: Current iteration (global step)
        checkpoint_path: Path to save checkpoint
        scaler: GradScaler object (optional)
    """
    print(f"Saving checkpoint to {checkpoint_path} at iteration {iteration}...")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Get state dict from model, handling DDP automatically
    model_state_dict = model.state_dict()

    save_dict = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'iteration': iteration,
        # 'hparams': hparams.to_dict() # Optionally save hparams used for this checkpoint
    }
    if scaler is not None:
        save_dict['scaler'] = scaler.state_dict() # Save scaler state if provided

    try:
        torch.save(save_dict, checkpoint_path)
        print(f"Saved checkpoint successfully.")
    except Exception as e:
        print(f"Error saving checkpoint to {checkpoint_path}: {e}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, strict=True):
    """
    Load model checkpoint, automatically handling DDP 'module.' prefix.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: PyTorch model (can be DDP or non-DDP).
        optimizer: PyTorch optimizer (optional).
        scaler: GradScaler object (optional).
        strict: Whether to strictly enforce that the keys in state_dict match.

    Returns:
        model: Loaded model.
        optimizer: Loaded optimizer (if provided).
        learning_rate: Learning rate from checkpoint (can be None).
        iteration: Iteration from checkpoint.
        scaler: Loaded scaler (if provided).
    """
    if not os.path.isfile(checkpoint_path):
         raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint '{checkpoint_path}'...")
    # Load to CPU first to prevent GPU mismatches
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    saved_state_dict = checkpoint_dict['model']

    # --- Automatic DDP prefix handling ---
    # Check if the current model is DDP-wrapped
    is_model_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    # Check if the saved state_dict has the 'module.' prefix
    saved_with_ddp_prefix = all(key.startswith('module.') for key in saved_state_dict.keys())

    if is_model_ddp and not saved_with_ddp_prefix:
        # Current model is DDP, saved state is not -> Add 'module.' prefix
        print("Adding 'module.' prefix to saved state dict keys for DDP model.")
        new_state_dict = {'module.' + k: v for k, v in saved_state_dict.items()}
    elif not is_model_ddp and saved_with_ddp_prefix:
        # Current model is not DDP, saved state is -> Remove 'module.' prefix
        print("Removing 'module.' prefix from saved state dict keys for non-DDP model.")
        new_state_dict = {k.replace('module.', '', 1): v for k, v in saved_state_dict.items()}
    else:
        # Prefixes match (both DDP or both not DDP) -> Use as is
        new_state_dict = saved_state_dict
    # -------------------------------------

    # Load the state dict into the model
    try:
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
        if not strict:
             if missing_keys: print(f"Warning: Missing keys in state_dict: {missing_keys}")
             if unexpected_keys: print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
    except Exception as e:
         print(f"Error loading model state_dict: {e}")
         # Optionally re-raise or return None/defaults depending on desired behavior
         raise e

    iteration = checkpoint_dict.get('iteration', checkpoint_dict.get('global_step', 0)) # Handle different key names
    learning_rate = checkpoint_dict.get('learning_rate', None) # Get LR if saved

    if optimizer is not None and 'optimizer' in checkpoint_dict:
        try:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            print("Loaded optimizer state.")
        except Exception as e:
            print(f"Warning: Could not load optimizer state dict: {e}. Optimizer might reset.")

    if scaler is not None and 'scaler' in checkpoint_dict:
        try:
            scaler.load_state_dict(checkpoint_dict['scaler'])
            print("Loaded scaler state.")
        except Exception as e:
            print(f"Warning: Could not load scaler state dict: {e}. Scaler might reset.")

    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model, optimizer, learning_rate, iteration, scaler # Return scaler

def scan_checkpoint(cp_dir, prefix='model_'):
    """
    Scan directory for checkpoints with a specific prefix and return the latest one.
    Assumes checkpoint filenames are like 'prefix_ITERATION.pt'.

    Args:
        cp_dir (str): Checkpoint directory.
        prefix (str): Checkpoint filename prefix (e.g., 'model_').

    Returns:
        str or None: Path to the latest checkpoint file, or None if no matching files found.
    """
    pattern = os.path.join(cp_dir, prefix + '*.pt') # Match prefix*.pt
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None

    # Extract iteration numbers and find the latest
    latest_file = None
    max_iteration = -1
    for f in checkpoint_files:
        try:
            # Extract iteration number assuming format prefix_ITERATION.pt
            basename = os.path.basename(f)
            iteration_str = basename.replace(prefix, '').replace('.pt', '')
            iteration = int(iteration_str)
            if iteration > max_iteration:
                max_iteration = iteration
                latest_file = f
        except ValueError:
            # Skip files that don't match the expected numeric iteration format
            print(f"Warning: Skipping file with unexpected format: {f}")
            continue

    return latest_file

def get_device():
    """
    Get the best available device for PyTorch ('cuda', 'mps', or 'cpu')

    Returns:
        torch.device: The selected PyTorch device object.
    """
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        return torch.device('mps')
    else:
        print("CUDA and MPS not available. Using CPU.")
        return torch.device('cpu')

def list_audio_files(directory, extensions=None):
    """
    Recursively list all audio files in a directory with given extensions.

    Args:
        directory (str): Directory path to search.
        extensions (list, optional): List of lowercase audio file extensions
                                     (e.g., ['.wav', '.mp3']). Defaults to common types.

    Returns:
        list: List of absolute paths to found audio files.
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a'] # Added .m4a

    audio_files = []
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found: {directory}")
        return audio_files

    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file extension is in the allowed list (case-insensitive)
            if any(file.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files

def create_dataset_csv(input_dir, output_csv, extensions=None):
    """
    Create a CSV file listing all audio files in a directory, assuming
    the immediate parent directory is the speaker ID.

    Args:
        input_dir (str): Directory containing speaker subdirectories with audio files.
        output_csv (str): Path to the output CSV file.
        extensions (list, optional): List of audio file extensions. Defaults to common types.
    """
    print(f"Creating dataset CSV '{output_csv}' from directory '{input_dir}'...")
    audio_files = list_audio_files(input_dir, extensions)

    if not audio_files:
        print("No audio files found.")
        return

    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'speaker_id']) # Standard header names

            count = 0
            for file_path in audio_files:
                try:
                    # Extract speaker ID from the parent directory name
                    speaker_id = os.path.basename(os.path.dirname(file_path))
                    if speaker_id:
                        writer.writerow([file_path, speaker_id])
                        count += 1
                    else:
                        print(f"Warning: Could not determine speaker ID for {file_path}")
                except Exception as e:
                    print(f"Warning: Error processing file {file_path}: {e}")

        print(f"Successfully created dataset CSV at '{output_csv}' with {count} files.")
    except Exception as e:
        print(f"Error creating dataset CSV: {e}")

def preprocess_dataset(input_csv_or_dir, output_dir, target_sr=48000, target_level=-27.0, output_format='wav'):
    """
    Preprocess audio files listed in a CSV or found in a directory.
    Converts to target sample rate, normalizes audio level, converts to stereo,
    and saves in the specified output format (default WAV).

    Args:
        input_csv_or_dir (str): Path to input CSV file or input directory.
        output_dir (str): Directory to save preprocessed files (maintains speaker structure).
        target_sr (int): Target sample rate.
        target_level (float): Target audio level in dBFS for normalization. Set to None to disable.
        output_format (str): Output audio format (e.g., 'wav', 'flac').
    """
    print(f"Starting dataset preprocessing...")
    print(f"Input: {input_csv_or_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Sample Rate: {target_sr} Hz")
    print(f"Target Normalization Level: {target_level} dBFS" if target_level is not None else "Normalization Disabled")
    print(f"Output Format: .{output_format}")

    # Get file list
    file_list = [] # List of (input_path, speaker_id)
    if os.path.isfile(input_csv_or_dir) and input_csv_or_dir.lower().endswith('.csv'):
        try:
            with open(input_csv_or_dir, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                header_lower = [h.lower().strip() for h in header]
                path_idx = header_lower.index('path')
                speaker_idx = header_lower.index('speaker_id')
                for row in reader:
                     if len(row) > max(path_idx, speaker_idx):
                          file_list.append((row[path_idx].strip(), row[speaker_idx].strip()))
                     else:
                          print(f"Warning: Skipping malformed row in CSV: {row}")
        except Exception as e:
            print(f"Error reading input CSV {input_csv_or_dir}: {e}")
            return
    elif os.path.isdir(input_csv_or_dir):
        all_files = list_audio_files(input_csv_or_dir)
        for fpath in all_files:
            try:
                 speaker = os.path.basename(os.path.dirname(fpath))
                 if speaker: file_list.append((fpath, speaker))
            except Exception: pass # Ignore files not in speaker folders
    else:
        print(f"Error: Input '{input_csv_or_dir}' is not a valid CSV or directory.")
        return

    if not file_list:
        print("No audio files found to preprocess.")
        return

    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0
    error_count = 0

    for file_path, speaker_id in tqdm(file_list, desc="Preprocessing"):
        try:
            # Create speaker directory in output
            speaker_dir = os.path.join(output_dir, speaker_id)
            os.makedirs(speaker_dir, exist_ok=True)

            # Define output path
            base_filename = os.path.basename(file_path)
            filename_no_ext = os.path.splitext(base_filename)[0]
            output_filename = f"{filename_no_ext}.{output_format.lower()}"
            output_file = os.path.join(speaker_dir, output_filename)

            # Load audio, resample to target SR, ensure it's float32
            # Use mono=False to load original channels, resample preserves channels
            audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
            # Ensure float32 numpy array
            audio = audio.astype(np.float32)

            # Convert mono to stereo if necessary
            if audio.ndim == 1:
                # If mono, duplicate to create stereo [2, T]
                print(f"Info: Converting mono file to stereo: {file_path}")
                audio = np.stack([audio, audio])
            elif audio.ndim == 2 and audio.shape[0] > 2:
                 # If more than 2 channels, take the first two
                 print(f"Warning: File has more than 2 channels ({audio.shape[0]}). Taking first two: {file_path}")
                 audio = audio[:2, :]
            elif audio.ndim != 2 or audio.shape[0] != 2:
                 print(f"Warning: Unexpected audio shape {audio.shape} for {file_path}. Skipping.")
                 error_count += 1
                 continue

            # Normalize audio channels independently if target_level is set
            if target_level is not None:
                 # Transpose to [T, C] for normalize_audio function if it expects that,
                 # or process channels directly. Assuming normalize_audio works on [T] arrays.
                 audio[0] = normalize_audio(audio[0], target_level=target_level)
                 audio[1] = normalize_audio(audio[1], target_level=target_level)

            # Save processed audio (soundfile expects [T, C] format)
            sf.write(output_file, audio.T, target_sr, format=output_format.upper()) # Use .T transpose
            processed_count += 1

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            error_count += 1

    print(f"\nPreprocessing complete.")
    print(f"Processed: {processed_count} files.")
    print(f"Errors: {error_count} files.")
    print(f"Output saved to: {output_dir}")