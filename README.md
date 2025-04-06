# Hanasu
Welcome to Hanasu, a human-like TTS model based on the multilingual Hanasu V1 BERT encoder and VITS architecture. Hanasu is a Japanese word that means "to speak." This project aims to build a TTS model that can speak multiple languages and mimic human-like prosody.

## Table of Content
- [Hanasu](#hanasu)
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
hanasu "Text to read" output.wav --speaker EN-US
```

**Specify a speaker:**
```bash
hanasu "Text to read" output.wav --speaker EN-US --speaker Yuna
```

**Specify a speed:**
```bash
hanasu "Text to read" output.wav --speaker EN-US --speaker Yuna --speed 1.5
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
Here’s the updated single-paragraph text with links added for both the **AGPLv3** and **CC BY-NC-ND 4.0** licenses:

---

## License
Hanasu is distributed under the OSI-approved [GNU Affero General Public License v3.0 (AGPLv3)](https://www.gnu.org/licenses/agpl-3.0.en.html); see `LICENSE.md` for more information. Additionally, on Hugging Face, encoder Hanasu and Himitsu are licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/), meaning they can only be used for non-commercial purposes without modification and with proper attribution. However, using these models for TTS as encoders are subject to the terms of the AGPLv3 license!

## Contact
For questions or support, please open an issue in the repository or contact the author at yukiarimo@gmail.com.
