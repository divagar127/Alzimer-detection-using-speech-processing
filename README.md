# Alzheimer's Detection using Speech Processing

This repository contains dataset segmentation files, metadata, and task structures for the **ADReSSo Challenge (Alzheimer's Dementia Recognition through Speech Tasks)**.

## 📌 Overview

The ADReSSo dataset focuses on automated speech and language processing for Alzheimer's Dementia recognition and cognitive decline prediction.

### Dataset Structure

- `diagnosis_train/`: Training data for Alzheimer's Diagnosis classification (AD vs. CN).
  - Contains utterance segmentation CSV files and label directory structures.
- `progression_train/`: Training data for Cognitive Decline Progression (`decline` vs. `no_decline`).
  - Contains segmentation CSV files and ground-truth metadata.
- `progression_test/`: Test set distribution for cognitive decline evaluation.

## 📁 Dataset & Audio Files Note

> **Note:** Audio files (`.wav` format, ~6.5 GB total) are ignored via `.gitignore` to comply with GitHub repository size and bandwidth limits.

To use audio files with this codebase, place your downloaded `.wav` files into the corresponding directories:
- `diagnosis_train/train/audio/ad/` and `diagnosis_train/train/audio/cn/`
- `progression_train/ADReSSo21/progression/train/audio/decline/` and `no_decline/`
- `progression_test/ADReSSo21/progression/test-dist/audio/`

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/divagar127/Alzimer-detection-using-speech-processing.git
   cd Alzimer-detection-using-speech-processing
   ```
2. Add your audio data into the designated audio folders.
3. Utilize the CSV segmentation files for audio processing, feature extraction (MFCCs, spectrograms, eGeMAPS), or linguistic modeling.

## 📄 License & Citation
Dataset courtesy of the ADReSSo Challenge (Interspeech 2021).
