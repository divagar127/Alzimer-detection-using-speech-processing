# Alzheimer's Detection using Speech Processing

This repository contains an end-to-end Machine Learning and Deep Learning pipeline for automated **Alzheimer's Dementia Recognition** and **Cognitive Decline Prediction** based on the **IS2021 ADReSSo Challenge** speech dataset.

---

## 🏗️ 5-Stage System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Audio Files (.wav)                    │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA PREPROCESSING                        │
│ • Load .wav files (librosa)  • Noise reduction (noisereduce)    │
│ • Resampling to 16kHz        • VAD Silence Removal              │
│ • Save processed files to new directory (processed_audio/)      │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. FEATURE EXTRACTION                         │
│ ACOUSTIC FEATURES (123-dim):     DEEP EMBEDDINGS:               │
│ • MFCC (40 + Delta + Delta2)     • Whisper Embeddings (1024-dim)│
│ • Pitch F0 (Mean, Std) & HNR     • Wav2Vec2 Embeddings (768-dim)│
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3. FEATURE FUSION (318-dim)                   │
│ • Concatenation: 123 Acoustic + 1024 Deep Embeddings = 1147-dim │
│ • Dimensionality Reduction: PCA / PyTorch Autoencoder           │
│ • Final Feature Vector: ~318 dimensions                         │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     4. CLASSIFICATION                           │
│ MACHINE LEARNING:                DEEP LEARNING:                 │
│ • SVM (RBF Kernel)               • PyTorch DNN (3 Hidden Layers)│
│ • Random Forest                  • PyTorch 1D-CNN               │
│ • Logistic Regression                                           │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     5. EVALUATION                               │
│ • 5-Fold Stratified Cross-Validation                            │
│ • Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC       │
│ • Saved Artifacts: Confusion Matrices & ROC-AUC Curve Plots     │
│ • Comparison Table with Baseline Papers                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Experimental Results (5-Fold Stratified Cross-Validation)

| Classifier Model | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **70.52% ± 8.06%** | **0.7217** | **0.7111** | **0.7117** | **0.7815** |
| **PyTorch DNN (3-Layer)** | **70.48% ± 9.07%** | **0.7208** | **0.7235** | **0.7209** | **0.7568** |
| **SVM (RBF Kernel)** | **66.29% ± 9.54%** | 0.6665 | 0.7346 | 0.6950 | 0.7440 |
| **Random Forest** | **54.80% ± 8.67%** | 0.5566 | 0.6562 | 0.5967 | 0.5285 |
| **PyTorch 1D-CNN** | **54.21% ± 7.80%** | 0.5644 | 0.6458 | 0.5924 | 0.5732 |

---

## 📈 Comparison with ADReSSo Challenge Baselines

| Method / Benchmark Study | Feature Representation | Accuracy (%) | F1-Score |
| :--- | :--- | :---: | :---: |
| **Acoustic Baseline (IS2021 ADReSSo)** | eGeMAPS (Acoustic) | 65.1% | 0.640 |
| **Linguistic Baseline (IS2021 ADReSSo)** | Transcripts (BERT) | 76.7% | 0.765 |
| **Wav2Vec2 Fine-tuned (Papasavvas et al.)** | Wav2Vec2 | 78.2% | 0.779 |
| **Our Proposed Pipeline (Fusion + Logistic Reg)** | Acoustic (123) + Deep (512) | **70.52%** | **0.7117** |
| **Our Proposed Pipeline (Fusion + DNN)** | Acoustic (123) + Deep (512) | **70.48%** | **0.7209** |
| **Our Proposed Pipeline (Fusion + SVM)** | Acoustic (123) + Deep (512) | **66.29%** | **0.6950** |

---

## 📁 Repository Structure & Dataset Note

> **Note:** Audio files (`.wav` format, ~6.5 GB total) are ignored via `.gitignore` to comply with GitHub repository size and bandwidth limits.

```
├── diagnosis_train/                # Alzheimer's Diagnosis training files & labels
├── progression_train/              # Progression training files & labels
├── progression_test/               # Test set distribution & evaluation CSVs
├── src/
│   ├── preprocessing.py            # Stage 1: Noise reduction, 16kHz resampling, VAD
│   ├── feature_extraction.py       # Stage 2: 123 Acoustic + Deep Embeddings
│   ├── feature_fusion.py           # Stage 3: Concatenation & PCA/Autoencoder reduction
│   ├── models.py                   # Stage 4: SVM, RF, Logistic Reg, PyTorch DNN & 1D-CNN
│   └── evaluation.py               # Stage 5: 5-Fold Stratified CV, metrics, plot generation
├── results/
│   ├── model_comparison.csv        # Metrics summary CSV
│   └── plots/                      # Saved confusion matrices & ROC curves
├── requirements.txt                # Python dependencies
├── run_pipeline.py                 # Main execution script
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/divagar127/Alzimer-detection-using-speech-processing.git
   cd Alzimer-detection-using-speech-processing
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline:**
   ```bash
   python run_pipeline.py
   ```

---

## 📄 License & Citation
Dataset courtesy of the **ADReSSo Challenge (IS2021)**.
