# Hanasu
Hanasu is a powerful voice library that allows you to convert text to speech (TTS) and perform voice conversion. It is designed to be easy to use and flexible.

# Hanasu TTS
Hanasu is a human-like TTS model based on the multilingual Himitsu V1 transformer-based encoder and VITS architecture. Hanasu is a Japanese word that means "to speak." This project aims to build a TTS model that can speak multiple languages and mimic human-like prosody.

## Table of Content
- [Hanasu](#hanasu)
- [Hanasu TTS](#hanasu-tts)
  - [Table of Content](#table-of-content)
  - [Installation](#installation)
    - [Requirements](#requirements)
  - [Usage](#usage)
    - [Language Support](#language-support)
    - [CLI](#cli)
    - [Python API](#python-api)
  - [Training](#training)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
- [Hanasu Converter](#hanasu-converter)
  - [Features](#features)
  - [Installation](#installation-1)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Quick Start](#quick-start)
    - [Training a Voice Model](#training-a-voice-model)
    - [Converting Voice](#converting-voice)
  - [Training Details](#training-details)
    - [Data Preparation](#data-preparation)
    - [Training Configuration](#training-configuration)
    - [Training with Limited Data](#training-with-limited-data)
  - [Inference](#inference)
    - [Voice Conversion Options](#voice-conversion-options)
    - [Batch Conversion](#batch-conversion)

## Installation
To install Hanasu, you can follow the instructions below:

```bash
git clone https://github.com/yukiarimo/hanasu.git
cd hanasu
pip install -e .
python -m unidic download
```

You can download the pre-trained models and encoders from the HF: [Hanasu V1 Encoder](https://huggingface.co/yukiarimo/yuna-ai-hanasu-v1).

### Requirements
- Python 3.8+
- PyTorch 1.9+
- torchaudio 0.9+
- transformers 4.9+
- librosa 0.8+
- unidic 1.0+
- 4GB+ GPU memory (for inference)
- 8GB+ GPU memory (for training)
- Supported GPUs: NVIDIA and Apple Silicon

## Usage
Hanasu can be used in two ways: CLI and Python API. The CLI is more user-friendly, while the Python API is more flexible.

### Language Support
Languages supported by Hanasu TTS:

- English (EN)
- Spanish (ES)
- French (FR)
- Chinese (ZH)
- Japanese (JP)
- Korean (KR)
- Russian (RU)

Additional languages can be added by transliterating the text to IPA.

### CLI
You may use the Hanasu CLI to interact with Hanasu. The CLI may be invoked using either `hanasu` or `hanasu`. Here are some examples:

**Read English text:**
```bash
hanasu "Text to read" output.wav
```

**Specify a language:**
```bash
hanasu "Text to read" output.wav --language EN
```

**Specify a speaker:**
```bash
hanasu "Text to read" output.wav --language EN --speaker Yuna
```

**Specify a speed:**
```bash
hanasu "Text to read" output.wav --language EN --speaker Yuna --speed 1.5
```

**Load from a file:**
```bash
hanasu file.txt out.wav --file
```

**Fuse models:**
```bash
# Basic Model Fusion: To fuse two models with default equal weighting:
python -m hanasu.fuse \
    --model_dirs /path/to/model1 /path/to/model2 \
    --model_steps 8600 108000 \
    --output_dir /path/to/output \
    --fallback_to_non_zero
```

**Fuse models with custom weighting:**
```bash
# Custom Model Fusion: Advanced Component-Specific Fusion with Three Models:
python -m hanasu.fuse \
    --model_dirs /path/to/model1 /path/to/model2 /path/to/model3 \
    --model_steps 8600 108000 21600 \
    --output_dir /path/to/output \
    --encoder_ratios 0.7 0.1 0.2 \
    --decoder_ratios 0.4 0.3 0.3 \
    --flow_attention_ratios 0.3 0.5 0.2 \
    --flow_other_ratios 0.3 0.4 0.3 \
    --duration_ratios 0.4 0.1 0.5 \
    --other_ratios 0.4 0.2 0.4 \
    --fallback_to_non_zero
```

**Fusion and Audio Generation:**
```bash
# Generate Audio from Fused Model:
python -m hanasu.fuse \
    --model_dirs /path/to/model1 /path/to/model2 \
    --model_steps 8600 108000 \
    --output_dir /path/to/output \
    --config_path /path/to/config.json \
    --generate_audio \
    --text "This is a test of the fused model synthesis." \
    --speaker_id 0 \
    --device cuda \
    --sdp_ratio 0.2 \
    --noise_scale 0.6 \
    --noise_scale_w 0.8 \
    --speed 1.0
```

### Python API
You may also use the Hanasu Python API to interact with Hanasu. Here is an example:

```python
from hanasu.api import TTS

# Speed is adjustable
text = "In a quiet neighborhood just west of Embassy Row in Washington, there exists a medieval-style walled garden whose roses, it is said, spring from twelfth-century plants. The garden's Carderock gazebo, known as Shadow House, sits elegantly amid meandering pathways of stones dug from George Washington's private quarry."
model = TTS(language='EN', device='cpu', use_hf=False, config_path="config.json", ckpt_path="G_100.pth")
model.tts_to_file(text=text, speaker_id=0, output_path='en-default.wav', sdp_ratio=0.8, noise_scale=0, noise_scale_w=0.2, speed=1.0, quiet=True)

"""
def tts_to_file(
 text,          # The input text to convert to speech
 speaker_id,    # ID of the speaker voice to use
 output_path,   # Where to save the audio file (optional)
 sdp_ratio,     # Controls the "cleanness" of the voice (0.0-1.0)
 noise_scale,   # Controls the variation in voice (0.0-1.0)
 noise_scale_w, # Controls the variation in speaking pace (0.0-1.0)
 speed,         # Speaking speed multiplier (1.0 = normal speed)
 pbar,          # Custom progress bar (optional)
 format,        # Audio format to save as (optional)
 position,      # Progress bar position (optional)
 quiet,         # Suppress progress output if True
):

- Higher sdp_ratio: Cleaner but more robotic voice
- Higher noise_scale: More variation/expressiveness but potentially less stable
- Higher noise_scale_w: More varied pacing but could sound unnatural
"""
```

## Training
To train Hanasu or its encoder, follow the steps in the `notebooks` directory. The training process is easy to follow and can be done on a single GPU. The training data is not included in this repository, but you can use your own data or download a dataset from a TTS dataset repository.

> Note 1: If you want to fine-tune the TTS model instead of training from scratch, place the pre-trained models into the `hanasu/logs/Yuna` directory.

> Note 2: If you plan to fine-tune or change the encoder, you must retrain the TTS models from scratch.

> Note 3: The TTS model has not been released yet, but you can train it yourself using the provided encoder. It will be released in the future once we have donations to support the project.

## Contributing
Contributions are welcome! Please open an issue in the repository for feature requests, bug reports, or other issues. If you want to contribute code, please fork the repository and submit a pull request.

## License
Hanasu is distributed under the OSI-approved [GNU Affero General Public License v3.0 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.en.html); see `LICENSE.md` for more information. Additionally, on Hugging Face, encoder Hanasu and Himitsu are licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/), meaning they can only be used for non-commercial purposes without modification and with proper attribution. However, using these models for TTS as encoders are subject to the terms of the AGPLv3 license!

## Contact
For questions or support, please open an issue in the repository or contact the author at yukiarimo@gmail.com.

# Hanasu Converter
Hanasu Converter is a high-quality voice-to-voice conversion system that transforms audio from one voice to another. It's designed to work with high-pitched anime voices and supports training on as little as 30 minutes of data.

## Features
- **Direct Voice-to-Voice Conversion**: Convert between voices without intermediate text representation
- **Retrieval-Based Approach**: Convert any input voice to trained voice models
- **High-Quality Audio**: Works with 48kHz stereo audio for high-pitched anime voice recordings
- **Cross-Platform Support**: Automatically detects and uses CPU, CUDA, or MPS (Apple Silicon)
- **Minimal Training Data**: Works with as little as 30 minutes of training data

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

2. Create a conda environment (optional but recommended):

```bash
conda create -n hanasu python=3.8
conda activate hanasu
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Voice Model

1. Prepare your dataset:
   - Organize audio files in a directory structure with speaker subdirectories
   - Create a dataset CSV file:

```bash
python -c "from hanasuconverter.utils import create_dataset_csv; create_dataset_csv('dataset', 'dataset.csv')"
```

2. Train the model:

```bash
python train.py --config config.json --data_path dataset.csv --output_dir ./outputs
```

### Converting Voice

1. Convert a single audio file:

```bash
python inference.py --config config.json --model outputs/checkpoints/model_latest.pt convert --source test.wav --output output.wav
```

2. Convert all files in a directory:

```bash
python inference.py --config config.json --model path/to/model.pt convert_dir --source_dir path/to/source_dir --output_dir path/to/output_dir
```

## Training Details

### Data Preparation

Hanasu Converter works best with clean audio data. For optimal results:

1. Use high-quality recordings (48kHz stereo recommended)
2. Remove background noise and reverb
3. Normalize audio levels
4. Split long recordings into 3-10 second segments

You can preprocess your dataset using the utility functions:

```bash
python -c "from hanasuconverter.utils import preprocess_dataset; preprocess_dataset('dataset.csv', 'path/to/processed_data')"
```

### Training Configuration

The `config.json` file contains all adjustable parameters. Key settings include:

```json
{
  "model": {
    "hidden_channels": 256,
    "gin_channels": 256
  },
  "train": {
    "learning_rate": 5e-5,
    "batch_size": 8,
    "weight_decay": 0.01
  }
}
```

### Training with Limited Data

For training with as little as 30 minutes of data:

1. Use the provided `test_minimal_data.py` script to create a small dataset:

```bash
python test_minimal_data.py --audio_dir path/to/audio --output_dir ./test_output --epochs 100
```

2. Use data augmentation techniques:
   - Pitch shifting
   - Time stretching
   - Adding small amounts of noise

## Inference

### Voice Conversion Options

Hanasu Converter uses a retrieval-based approach, where any input voice can be converted to the trained voice model:

```bash
python inference.py --config config.json --model path/to/model.pt convert --source path/to/source.wav --output path/to/output.wav
```

The `--tau` parameter controls the temperature of the conversion (default: 0.7):
- Lower values (0.3-0.5): More stable but less expressive
- Higher values (0.7-1.0): More expressive but potentially less stable

### Batch Conversion

Convert multiple files using multiple trained models:

```bash
python inference.py --config config.json batch --source_dir path/to/source_dir --model_dir path/to/model_dir --output_dir path/to/output_dir
```