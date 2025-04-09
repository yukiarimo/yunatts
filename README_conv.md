# Hanasu Converter

Hanasu Converter is a powerful voice-to-voice conversion system that allows you to transform audio from one voice to another. It supports training on multiple speakers and converting between their voices, as well as training on one speaker and converting to another using reference audio.

## Features

- **Direct Voice-to-Voice Conversion**: Convert between voices without intermediate text representation
- **Multiple Speaker Support**: Train on multiple speakers and convert between any of them
- **Reference-Based Conversion**: Use reference audio to define target voice characteristics
- **High-Quality Audio**: Works with 48kHz stereo audio for high-quality, high-pitched voice recordings
- **Cross-Platform Support**: Automatically detects and uses CPU, CUDA, or MPS (Apple Silicon)
- **Comprehensive Tools**: Includes training, inference, and utility scripts for a complete workflow

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA (optional, for GPU acceleration)
- MPS-enabled PyTorch (optional, for Apple Silicon)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hanasu-converter.git
cd hanasu-converter
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Converting Voice with Pre-trained Models

1. Download a pre-trained model (if available) or train your own (see [Training](#training)).
2. Extract the target speaker embedding from reference audio:

```bash
python inference.py --config config.json --checkpoint path/to/checkpoint.pth \
    extract --reference path/to/reference_audio.wav --output path/to/target_embedding.pth
```

3. Convert a source audio file to the target voice:

```bash
python inference.py --config config.json --checkpoint path/to/checkpoint.pth \
    convert --source path/to/source_audio.wav --target_embedding path/to/target_embedding.pth \
    --output path/to/output_audio.wav
```

## Data Preparation

Hanasu Converter requires audio data for training. The data should be organized as follows:

```
data/
├── speaker1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── speaker2/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```

### Audio Requirements

- Format: WAV, MP3, FLAC, or OGG (will be converted to WAV during preprocessing)
- Sample Rate: 48kHz (recommended, will be resampled if different)
- Channels: Mono or Stereo (stereo recommended for high-quality conversion)
- Duration: 3-10 seconds per file is ideal (longer files will be segmented)

### Preprocessing

You can preprocess your dataset using the utility functions:

```bash
# Create a CSV file listing all audio files
python -c "from hanasu.utils import create_dataset_csv; create_dataset_csv('dataset', 'dataset.csv')"

# Preprocess the dataset (resample, normalize, etc.)
python -c "from hanasu.utils import preprocess_dataset; preprocess_dataset('dataset.csv', 'path/to/processed_data')"
```

## Training

### Training from Scratch

To train a model from scratch:

```bash
python train.py --config config.json --data_path dataset.csv --output_dir outputs
```

### Continuous Training

To continue training from a checkpoint:

```bash
python train.py --config config.json --data_path dataset.csv --output_dir outputs \
    --checkpoint_path outputs/checkpoints/model_latest.pt
```

### Training Options

- `--config`: Path to the configuration file
- `--data_path`: Path to the data directory or CSV file
- `--output_dir`: Directory to save output files
- `--checkpoint_dir`: Directory to save checkpoints
- `--log_dir`: Directory to save logs
- `--checkpoint_path`: Path to checkpoint for continuing training
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimization
- `--seed`: Random seed for reproducibility

### Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

## Inference

Hanasu Converter provides several inference modes for different use cases.

### Single File Conversion

Convert a single audio file:

```bash
python inference.py --config config.json --checkpoint outputs/checkpoints/model_latest.pt \
    convert --source test.wav --target_embedding target_embedding.pth \
    --output output_audio.wav
```

### Directory Conversion

Convert all audio files in a directory:

```bash
python inference.py --config config.json --checkpoint path/to/checkpoint.pth \
    convert_dir --source_dir path/to/source_dir --target_embedding path/to/target_embedding.pth \
    --output_dir path/to/output_dir
```

### Extract Speaker Embedding

Extract speaker embedding from reference audio:

```bash
python inference.py --config config.json --checkpoint outputs/checkpoints/model_latest.pt \
    extract --reference emb.wav --output target_embedding.pth
```

### Batch Conversion

Perform batch conversion between multiple speakers:

```bash
python inference.py --config config.json --checkpoint path/to/checkpoint.pth \
    batch --source_dir path/to/source_dir --reference_dir path/to/reference_dir \
    --output_dir path/to/output_dir
```

### Inference Options

- `--config`: Path to the configuration file
- `--checkpoint`: Path to the model checkpoint
- `--device`: Device to use (cuda, mps, cpu)
- `--tau`: Temperature parameter for conversion (default: 0.3)

## Configuration

Hanasu Converter uses a JSON configuration file to control various aspects of the system. The default configuration is provided in `config.json`, but you can modify it to suit your needs.

### Data Configuration

```json
"data": {
  "sampling_rate": 48000,
  "filter_length": 2048,
  "hop_length": 512,
  "win_length": 2048,
  "n_mel_channels": 128,
  "mel_fmin": 0.0,
  "mel_fmax": 24000.0
}
```

### Model Configuration

```json
"model": {
  "inter_channels": 192,
  "hidden_channels": 192,
  "filter_channels": 768,
  "resblock": "1",
  "resblock_kernel_sizes": [3, 7, 11],
  "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
  "upsample_rates": [8, 8, 2, 2],
  "upsample_initial_channel": 512,
  "upsample_kernel_sizes": [16, 16, 4, 4],
  "gin_channels": 256
}
```

### Training Configuration

```json
"train": {
  "log_interval": 100,
  "eval_interval": 1000,
  "seed": 1234,
  "epochs": 100,
  "learning_rate": 2e-4,
  "betas": [0.8, 0.99],
  "eps": 1e-9,
  "batch_size": 16,
  "fp16_run": true,
  "lr_decay": 0.999875,
  "segment_size": 16384,
  "c_mel": 45,
  "c_kl": 1.0,
  "grad_clip": 1.0,
  "weight_decay": 0.0,
  "same_speaker_prob": 0.5,
  "num_workers": 4,
  "checkpoint_interval": 5000
}
```

### Inference Configuration

```json
"inference": {
  "tau": 0.3,
  "batch_size": 1,
  "segment_size": 512000
}
```

### Audio Configuration

```json
"audio": {
  "stereo": true,
  "normalize_audio": true,
  "target_level": -27,
  "vad_threshold": -40,
  "min_silence_duration": 500,
  "keep_silence_duration": 200
}
```

### Device Configuration

```json
"device": {
  "auto_detect": true,
  "force_cpu": false,
  "cuda_device": 0,
  "use_fp16": true
}
```

## Advanced Usage

### Multi-GPU Training

To train with multiple GPUs:

```bash
python train.py --config config.json --data_path path/to/data --output_dir path/to/output
```

The system will automatically detect available GPUs and use distributed training if multiple GPUs are available.

### Custom Audio Processing

You can customize audio processing parameters in the configuration file:

```json
"audio": {
  "stereo": true,
  "normalize_audio": true,
  "target_level": -27,
  "vad_threshold": -40,
  "min_silence_duration": 500,
  "keep_silence_duration": 200
}
```

### Fine-tuning for Specific Voices

For better results with specific voice types (e.g., high-pitched anime voices):

1. Prepare a dataset with similar voice characteristics
2. Adjust the mel spectrogram parameters in the configuration:

```json
"data": {
  "sampling_rate": 48000,
  "filter_length": 2048,
  "hop_length": 512,
  "win_length": 2048,
  "n_mel_channels": 128,
  "mel_fmin": 0.0,
  "mel_fmax": 24000.0
}
```

3. Train the model with a smaller learning rate:

```bash
python train.py --config config.json --data_path path/to/data --output_dir path/to/output --learning_rate 1e-4
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

If you encounter CUDA out of memory errors:

1. Reduce the batch size in the configuration file or via command line:

```bash
python train.py --config config.json --data_path path/to/data --output_dir path/to/output --batch_size 8
```

2. Reduce the model size by adjusting the model parameters in the configuration file.

#### Poor Conversion Quality

If the conversion quality is poor:

1. Ensure your training data is high quality and consistent
2. Train for more epochs
3. Adjust the temperature parameter (`tau`) during inference:

```bash
python inference.py --config config.json --checkpoint path/to/checkpoint.pth \
    convert --source path/to/source_audio.wav --target_embedding path/to/target_embedding.pth \
    --output path/to/output_audio.wav --tau 0.4
```

#### CPU Performance Issues

If inference is slow on CPU:

1. Reduce the model size in the configuration file
2. Use a smaller segment size for processing:

```json
"inference": {
  "segment_size": 256000
}
```