# Yuna TTS
Welcome to Yuna TTS, a leading text-to-speech (TTS) engine designed to provide a natural and expressive voice for your applications. Yuna TTS is a deep learning-based TTS engine that can generate high-quality speech from text. It is designed to be easy to use and integrate into your applications, with support for multiple languages and voices.

## Table of Content
- [Yuna TTS](#yuna-tts)
  - [Table of Content](#table-of-content)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Language Support](#language-support)
    - [CLI](#cli)
    - [Python API](#python-api)
  - [Training](#training)
    - [Data Preparation](#data-preparation)
    - [Training](#training-1)
    - [Inference](#inference)

## Installation
The repo is developed and tested on `Ubuntu 20.04` and `Python 3.9`. To install YunaTTS, you can follow the instructions below:

```bash
git clone https://github.com/yukiarimo/yunatts.git
cd yunatts
pip install -e .
python -m unidic download
```

## Usage

### Language Support
Languages supported by Yuna TTS:

- English (EN)
- Spanish (ES)
- French (FR)
- Chinese (ZH)
- Japanese (JP)
- Korean (KR)

### CLI
You may use the YunaTTS CLI to interact with YunaTTS. The CLI may be invoked using either `yunatts` or `yuna`. Here are some examples:

**Read English text:**

```bash
yuna "Text to read" output.wav
```

**Specify a language:**

```bash
yuna "Text to read" output.wav --language EN
```

**Specify a speaker:**

```bash
yuna "Text to read" output.wav --language EN --speaker EN-US
yuna "Text to read" output.wav --language EN --speaker EN-AU
```

**Specify a speed:**

```bash
yuna "Text to read" output.wav --language EN --speaker EN-US --speed 1.5
yuna "Text to read" output.wav --speed 1.5
```

**Load from a file:**

```bash
yuna file.txt out.wav --file
```

### Python API

```python
from yuna.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto' # Will automatically use GPU if available
language = 'EN'

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language=language, device=device)
output_path = 'en.wav'
model.tts_to_file(text, 0, output_path, speed=speed)
```

## Training
Before training, please install YunaTTS in dev mode and go to the `yuna` folder.

```
pip install -e .
cd yuna
```

### Data Preparation
To train a TTS model, we need to prepare the audio files and a metadata file.

```
path/to/audio_001.wav|<speaker_name>|<language_code>|<text_001>
path/to/audio_002.wav|<speaker_name>|<language_code>|<text_002>
```

We can then run the preprocessing code:

```
python preprocess_text.py --metadata data/example/metadata.list 
```

A config file `data/example/config.json` will be generated. Feel free to edit some hyper-parameters in that config file (for example, you may decrease the batch size if you have encountered the CUDA out-of-memory issue).

### Training
The training can be launched by:

```
bash train.sh <path/to/config.json> <num_of_gpus>
```

### Inference
Simply run:

```
python infer.py --text "<some text here>" -m /path/to/checkpoint/G_<iter>.pth -o <output_dir>
```