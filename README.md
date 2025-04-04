# VITS Model Modifications Documentation

This document outlines the modifications made to the VITS text-to-speech model as requested.

## Overview of Changes

1. **Model Parameter Updates**
   - Increased n_flows from 4 to 8
   - Increased hidden_channels from 192 to 256
   - Increased inter_channels from 192 to 256

2. **Text Processing Simplification**
   - Removed language_emb and tone_emb
   - Implemented raw character processing instead of phonemes
   - Added transliteration to English for all languages (Japanese, Russian, etc.)
   - Limited input to lowercase letters, numbers, and basic punctuation (!?.,)

3. **Embedding System Replacement**
   - Removed bert_proj and ja_bert_proj
   - Implemented Llama-3.2-1B embeddings for all languages
   - Added handling for potential size mismatches between text and embeddings

4. **Audio Quality Improvements**
   - Added support for 48kHz, 16-bit audio
   - Implemented stereo audio support for both training and generation

5. **Training Simplification**
   - Removed multi-GPU training code
   - Optimized for single GPU training
   - Added compatibility for both NVIDIA GPUs and Apple Silicon (M1, M2, M3)

## Usage Instructions

### Training

To train the model:

```bash
python hanasu/train.py
```

The training script will automatically detect and use the appropriate device (CUDA for NVIDIA GPUs, MPS for Apple Silicon, or CPU as fallback).

### Inference

To generate audio using the trained model:

```bash
python test_model.py --model /path/to/model.pth --config /path/to/config.json --text "Your text here" --speaker 0 --language EN --output output.wav
```

Parameters:
- `--model`: Path to the model checkpoint
- `--config`: Path to the configuration file
- `--text`: Text to synthesize
- `--speaker`: Speaker ID (default: 0)
- `--language`: Language code (EN, JP, RU, etc.) (default: EN)
- `--output`: Output audio file path (default: output.wav)

## Technical Details

### Text Processing

The text processing pipeline has been simplified to use raw characters instead of phonemes:
- Input is limited to lowercase letters, numbers, and basic punctuation (!?,.)
- Non-English text is transliterated to English characters
- Numbers are expected to be written as words

### Llama Embeddings

The model now uses Llama-3.2-1B embeddings instead of BERT:
- Embedding size is 8192
- Embeddings are generated before text processing
- The model handles potential size mismatches between text and embeddings

### Stereo Audio Support

The model now supports stereo audio:
- Audio processing functions handle both mono and stereo inputs
- The Generator class can output either mono or stereo audio
- The data loading pipeline properly handles stereo audio files

### Device Compatibility

The model is now compatible with:
- NVIDIA GPUs (using CUDA)
- Apple Silicon (using MPS)
- CPU (as fallback)

## File Structure

- `hanasu/models.py`: Contains the main model architecture
- `hanasu/modules.py`: Contains model components
- `hanasu/mel_processing.py`: Audio processing functions
- `hanasu/train.py`: Training script
- `hanasu/text/`: Text processing components
- `hanasu/llama_utils.py`: Llama embedding utilities
- `test_model.py`: Test script for inference

## Dependencies

- PyTorch (with CUDA or MPS support)
- Transformers (for Llama model)
- Librosa (for audio processing)
- SoundFile (for stereo audio output)
- NumPy
- tqdm

## Notes

- The model is optimized for high-pitched seiyuu anime girl voice
- For best results, use 48kHz, 16-bit stereo audio for training
- When using Apple Silicon, make sure to use PyTorch with MPS support
