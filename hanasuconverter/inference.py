import os
import argparse
from tqdm import tqdm

from hanasuconverter.converter import VoiceConverter
from hanasuconverter.utils import get_hparams_from_file

def process_file(converter, source_path, model_path, output_path, tau=0.7):
    """
    Process a single audio file for voice conversion

    Args:
        converter: VoiceConverter instance
        source_path: Path to source audio file
        model_path: Path to trained model containing target voice
        output_path: Path to save converted audio
        tau: Temperature parameter for conversion
    """
    # Load model checkpoint
    converter.load_checkpoint(model_path)

    # Extract source speaker embedding
    source_embedding = converter.extract_speaker_embedding(source_path)

    # Get target speaker embedding from the model
    # In retrieval-based approach, we use the trained speaker embedding
    target_embedding = get_model_speaker_embedding(converter.model)

    # Convert voice
    converter.convert_voice(
        source_path,
        source_embedding,
        target_embedding,
        output_path,
        tau=tau
    )

    print(f"Converted {source_path} to {output_path}")

def process_directory(converter, source_dir, model_path, output_dir, tau=0.7):
    """
    Process all audio files in a directory for voice conversion

    Args:
        converter: VoiceConverter instance
        source_dir: Directory containing source audio files
        model_path: Path to trained model containing target voice
        output_dir: Directory to save converted audio files
        tau: Temperature parameter for conversion
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model checkpoint
    converter.load_checkpoint(model_path)

    # Get target speaker embedding from the model
    target_embedding = get_model_speaker_embedding(converter.model)

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

def get_model_speaker_embedding(model):
    """
    Get the speaker embedding from the trained model
    """
    # Check for the embedding in the model state dict
    if hasattr(model, 'state_dict') and 'emb_g.weight' in model.state_dict():
        embedding = model.state_dict()['emb_g.weight'][0].detach().clone()
        embedding = embedding.unsqueeze(0).unsqueeze(-1)  # [1, gin_channels, 1]
        return embedding
    # Check for unexpected keys that might contain the embedding
    elif hasattr(model, '_parameters') and 'emb_g.weight' in model._parameters:
        embedding = model._parameters['emb_g.weight'][0].detach().clone()
        embedding = embedding.unsqueeze(0).unsqueeze(-1)  # [1, gin_channels, 1]
        return embedding
    # Original check
    elif hasattr(model, 'emb_g') and model.emb_g is not None:
        embedding = model.emb_g.weight[0].detach().clone()
        embedding = embedding.unsqueeze(0).unsqueeze(-1)  # [1, gin_channels, 1]
        return embedding
    else:
        raise ValueError("Model does not have speaker embeddings. Make sure it was trained properly.")

def batch_conversion_multi_model(converter, source_dir, model_dir, output_dir, tau=0.7):
    """
    Perform batch conversion using multiple trained models

    Args:
        converter: VoiceConverter instance
        source_dir: Directory containing source audio files
        model_dir: Directory containing trained models
        output_dir: Directory to save converted audio files
        tau: Temperature parameter for conversion
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all model files in model_dir
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.pt'):
            model_files.append(os.path.join(model_dir, file))

    if not model_files:
        print(f"No model files found in {model_dir}")
        return

    print(f"Found {len(model_files)} model files")

    # Get all audio files in source directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print(f"No audio files found in {source_dir}")
        return

    print(f"Found {len(audio_files)} audio files")

    # Process each model
    for model_path in model_files:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        print(f"Processing model: {model_name}")

        # Load model checkpoint
        converter.load_checkpoint(model_path)

        # Get target speaker embedding from the model
        try:
            target_embedding = get_model_speaker_embedding(converter.model)

            # Process each audio file
            for source_path in tqdm(audio_files, desc=f"Converting to {model_name}"):
                # Create output path
                output_filename = os.path.basename(source_path)
                if not output_filename.lower().endswith('.wav'):
                    output_filename = os.path.splitext(output_filename)[0] + '.wav'
                output_path = os.path.join(model_output_dir, output_filename)

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
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

    print(f"Batch conversion complete. Results saved to {output_dir}")

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Hanasu Converter - Voice Conversion Tool")

    # General arguments
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, help='Device to use (cuda, mps, cpu)')

    # Mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Single file conversion mode
    convert_parser = subparsers.add_parser('convert', help='Convert a single audio file')
    convert_parser.add_argument('--source', type=str, required=True, help='Path to source audio file')
    convert_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    convert_parser.add_argument('--output', type=str, required=True, help='Path to save converted audio')
    convert_parser.add_argument('--tau', type=float, default=0.7, help='Temperature parameter for conversion')

    # Directory conversion mode
    convert_dir_parser = subparsers.add_parser('convert_dir', help='Convert all audio files in a directory')
    convert_dir_parser.add_argument('--source_dir', type=str, required=True, help='Directory containing source audio files')
    convert_dir_parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    convert_dir_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save converted audio files')
    convert_dir_parser.add_argument('--tau', type=float, default=0.7, help='Temperature parameter for conversion')

    # Batch conversion with multiple models mode
    batch_parser = subparsers.add_parser('batch', help='Perform batch conversion with multiple models')
    batch_parser.add_argument('--source_dir', type=str, required=True, help='Directory containing source audio files')
    batch_parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained models')
    batch_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save converted audio files')
    batch_parser.add_argument('--tau', type=float, default=0.7, help='Temperature parameter for conversion')

    args = parser.parse_args()

    # Load config
    hps = get_hparams_from_file(args.config)

    # Determine device
    device = args.device

    # Create converter
    converter = VoiceConverter(args.config, device=device)

    # Execute selected mode
    if args.mode == 'convert':
        process_file(converter, args.source, args.model, args.output, args.tau)

    elif args.mode == 'convert_dir':
        process_directory(converter, args.source_dir, args.model, args.output_dir, args.tau)

    elif args.mode == 'batch':
        batch_conversion_multi_model(converter, args.source_dir, args.model_dir, args.output_dir, args.tau)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()