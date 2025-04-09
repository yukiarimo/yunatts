import os
import argparse
import torch
from tqdm import tqdm
from hanasuconverter.converter import VoiceConverter
from hanasuconverter.utils import get_hparams_from_file, get_device

def process_file(converter, source_path, target_embedding_path, output_path, tau=0.3):
    """
    Process a single audio file for voice conversion

    Args:
        converter: VoiceConverter instance
        source_path: Path to source audio file
        target_embedding_path: Path to target speaker embedding
        output_path: Path to save converted audio
        tau: Temperature parameter for conversion
    """
    # Load target speaker embedding
    target_embedding = torch.load(target_embedding_path, map_location=converter.device)

    # Extract source speaker embedding
    source_embedding = converter.extract_speaker_embedding(source_path)

    # Convert voice
    converter.convert_voice(
        source_path,
        source_embedding,
        target_embedding,
        output_path,
        tau=tau
    )

    print(f"Converted {source_path} to {output_path}")

def process_directory(converter, source_dir, target_embedding_path, output_dir, tau=0.3):
    """
    Process all audio files in a directory for voice conversion

    Args:
        converter: VoiceConverter instance
        source_dir: Directory containing source audio files
        target_embedding_path: Path to target speaker embedding
        output_dir: Directory to save converted audio files
        tau: Temperature parameter for conversion
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load target speaker embedding
    target_embedding = torch.load(target_embedding_path, map_location=converter.device)

    # Get all audio files in source directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))

    # Process each audio file
    for source_path in tqdm(audio_files, desc="Converting audio files"):
        # Create output path
        rel_path = os.path.relpath(source_path, source_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Ensure output file is .wav
        if not output_path.lower().endswith('.wav'):
            output_path = os.path.splitext(output_path)[0] + '.wav'

        # Extract source speaker embedding
        source_embedding = converter.extract_speaker_embedding(source_path)

        # Convert voice
        converter.convert_voice(
            source_path,
            source_embedding,
            target_embedding,
            output_path,
            tau=tau
        )

    print(f"Converted {len(audio_files)} files from {source_dir} to {output_dir}")

def extract_reference_embedding(converter, reference_path, output_path):
    """
    Extract speaker embedding from reference audio

    Args:
        converter: VoiceConverter instance
        reference_path: Path to reference audio file or directory
        output_path: Path to save speaker embedding
    """
    # --- Create output directory only if necessary ---
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir): # Check if dirname is not empty and doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    # -----------------------------------------------

    # Check if reference_path is a directory
    if os.path.isdir(reference_path):
        # Get all audio files in reference directory
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        print(f"Extracting embedding from directory: {reference_path}")
        for root, _, files in os.walk(reference_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))

        if not audio_files:
             print(f"Error: No supported audio files found in directory {reference_path}")
             return

        # Extract speaker embedding from all files (pass the list)
        # The converter method handles saving internally if path is given
        converter.extract_speaker_embedding(audio_files, output_path)

    elif os.path.isfile(reference_path):
         print(f"Extracting embedding from file: {reference_path}")
         # Extract speaker embedding from single file (pass the string)
         # The converter method handles saving internally if path is given
         converter.extract_speaker_embedding(reference_path, output_path)
    else:
         print(f"Error: Reference path '{reference_path}' is not a valid file or directory.")
         return # Exit if reference path is invalid

    if os.path.exists(output_path):
        print(f"Successfully extracted speaker embedding to {output_path}")
    else:
        # This might happen if extract_speaker_embedding failed silently
        print(f"Warning: Speaker embedding extraction might have failed, output file not found at {output_path}")

def batch_conversion(converter, source_dir, reference_dir, output_dir, tau=0.3):
    """
    Perform batch conversion between multiple speakers

    Args:
        converter: VoiceConverter instance
        source_dir: Directory containing source audio files organized by speaker
        reference_dir: Directory containing reference audio files organized by speaker
        output_dir: Directory to save converted audio files
        tau: Temperature parameter for conversion
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all speaker directories in source_dir
    source_speakers = [d for d in os.listdir(source_dir) 
                      if os.path.isdir(os.path.join(source_dir, d))]

    # Get all speaker directories in reference_dir
    reference_speakers = [d for d in os.listdir(reference_dir) 
                         if os.path.isdir(os.path.join(reference_dir, d))]

    print(f"Found {len(source_speakers)} source speakers: {source_speakers}")
    print(f"Found {len(reference_speakers)} reference speakers: {reference_speakers}")

    # Extract speaker embeddings for all reference speakers
    embeddings = {}
    for speaker in tqdm(reference_speakers, desc="Extracting reference embeddings"):
        speaker_dir = os.path.join(reference_dir, speaker)
        embedding_path = os.path.join(output_dir, f"{speaker}_embedding.pth")

        # Extract speaker embedding
        embedding = converter.extract_speaker_embedding(speaker_dir, embedding_path)
        embeddings[speaker] = embedding

    # Convert audio for all source speakers to all reference speakers
    for source_speaker in tqdm(source_speakers, desc="Processing source speakers"):
        source_speaker_dir = os.path.join(source_dir, source_speaker)

        # Get all audio files for this speaker
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        for root, _, files in os.walk(source_speaker_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))

        # Skip if no audio files found
        if not audio_files:
            print(f"No audio files found for speaker {source_speaker}, skipping")
            continue

        # Extract source speaker embedding (shared across all conversions)
        source_embedding_path = os.path.join(output_dir, f"{source_speaker}_embedding.pth")
        source_embedding = converter.extract_speaker_embedding(audio_files, source_embedding_path)

        # Convert to each reference speaker
        for target_speaker, target_embedding in embeddings.items():
            # Create output directory for this conversion
            conversion_dir = os.path.join(output_dir, f"{source_speaker}_to_{target_speaker}")
            os.makedirs(conversion_dir, exist_ok=True)

            # Process each audio file
            for source_path in tqdm(audio_files, desc=f"Converting {source_speaker} to {target_speaker}"):
                # Create output path
                output_filename = os.path.basename(source_path)
                if not output_filename.lower().endswith('.wav'):
                    output_filename = os.path.splitext(output_filename)[0] + '.wav'
                output_path = os.path.join(conversion_dir, output_filename)

                # Convert voice
                converter.convert_voice(
                    source_path,
                    source_embedding,
                    target_embedding,
                    output_path,
                    tau=tau
                )

    print(f"Batch conversion complete. Results saved to {output_dir}")

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Hanasu Converter - Voice Conversion Tool")

    # General arguments
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, help='Device to use (cuda, mps, cpu)')

    # Mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Single file conversion mode
    convert_parser = subparsers.add_parser('convert', help='Convert a single audio file')
    convert_parser.add_argument('--source', type=str, required=True, help='Path to source audio file')
    convert_parser.add_argument('--target_embedding', type=str, required=True, help='Path to target speaker embedding')
    convert_parser.add_argument('--output', type=str, required=True, help='Path to save converted audio')
    convert_parser.add_argument('--tau', type=float, default=0.3, help='Temperature parameter for conversion')

    # Directory conversion mode
    convert_dir_parser = subparsers.add_parser('convert_dir', help='Convert all audio files in a directory')
    convert_dir_parser.add_argument('--source_dir', type=str, required=True, help='Directory containing source audio files')
    convert_dir_parser.add_argument('--target_embedding', type=str, required=True, help='Path to target speaker embedding')
    convert_dir_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save converted audio files')
    convert_dir_parser.add_argument('--tau', type=float, default=0.3, help='Temperature parameter for conversion')

    # Extract embedding mode
    extract_parser = subparsers.add_parser('extract', help='Extract speaker embedding from reference audio')
    extract_parser.add_argument('--reference', type=str, required=True, help='Path to reference audio file or directory')
    extract_parser.add_argument('--output', type=str, required=True, help='Path to save speaker embedding')

    # Batch conversion mode
    batch_parser = subparsers.add_parser('batch', help='Perform batch conversion between multiple speakers')
    batch_parser.add_argument('--source_dir', type=str, required=True, help='Directory containing source audio files organized by speaker')
    batch_parser.add_argument('--reference_dir', type=str, required=True, help='Directory containing reference audio files organized by speaker')
    batch_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save converted audio files')
    batch_parser.add_argument('--tau', type=float, default=0.3, help='Temperature parameter for conversion')

    args = parser.parse_args()

    # Load config
    hps = get_hparams_from_file(args.config)

    # Determine device
    device = args.device if args.device else get_device()

    # Create converter
    converter = VoiceConverter(args.config, device=device)

    # Load checkpoint
    converter.load_checkpoint(args.checkpoint)

    # Execute selected mode
    if args.mode == 'convert':
        process_file(converter, args.source, args.target_embedding, args.output, args.tau)

    elif args.mode == 'convert_dir':
        process_directory(converter, args.source_dir, args.target_embedding, args.output_dir, args.tau)

    elif args.mode == 'extract':
        extract_reference_embedding(converter, args.reference, args.output)

    elif args.mode == 'batch':
        batch_conversion(converter, args.source_dir, args.reference_dir, args.output_dir, args.tau)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
