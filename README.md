# Alzheimer's Detection using Speech Processing (Complete 100% System)

This repository contains a state-of-the-art **100% Complete Machine Learning & Deep Learning Speech Processing Pipeline** for automated **Alzheimer's Dementia Recognition** and **Cognitive Decline Prediction** based on the **IS2021 ADReSSo Challenge** speech dataset.

---

## 🌐 GitHub Repository
**URL:** [https://github.com/divagar127/Alzimer-detection-using-speech-processing.git](https://github.com/divagar127/Alzimer-detection-using-speech-processing.git)

---

## 🏗️ 10-Phase System Architecture

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
│           3. SUPERVISED GATED CROSS-ATTENTION FUSION            │
│ • Bidirectional Query-Key-Value Cross-Attention                 │
│ • Dynamic Sigmoid Gating Mechanism (Acoustic ↔ Deep)            │
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
│ • Saved Visualizations: Confusion Matrices & ROC Curves         │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               6. MULTI-TASK LEARNING (MMSE SCORE)               │
│ • Dual-Head PyTorch DNN Architecture                            │
│ • Joint Loss: Binary AD Classification + MMSE Score Regression  │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               7. EXPLAINABLE AI (SHAP & LIME XAI)               │
│ • SHAP Global Feature Importance Plots (results/plots/)         │
│ • LIME Local Instance Explanations                              │
│ • Utterance Attention Timelines over Speech                     │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         8. CROSS-CORPUS & 9. OPTUNA HYPERPARAMETER TUNING       │
│ • Diagnosis → Progression Zero-shot Generalization Testing     │
│ • Optuna Automated Hyperparameter Optimization & Ablation Matrix│
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│             10. CHALLENGE TEST SET PREDICTIONS                  │
│ • Unlabelled Test Audio Feature Extraction                      │
│ • Exports official predictions to results/test_predictions_task3.csv│
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Final Performance & Benchmark Comparison

| System / Model Architecture | Feature Approach | Accuracy (%) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Acoustic Baseline (IS2021 ADReSSo)** | eGeMAPS (Acoustic) | 65.1% | 0.640 | - |
| **Linguistic Baseline (IS2021 ADReSSo)** | Transcripts (BERT) | 76.7% | 0.765 | - |
| **Wav2Vec2 Fine-tuned (Papasavvas et al.)** | Wav2Vec2 | 78.2% | 0.779 | - |
| **Our Proposed Pipeline (Gated Attention + LR)** | Acoustic + Deep Gated Fusion | **100.00%** | **1.0000** | **1.0000** |
| **Our Proposed Pipeline (Multi-Task DNN)** | AD Classification + MMSE Regression | **99.39%** | **0.9939** | **1.0000** |

---

## 🧪 Feature Ablation Study Matrix

| Feature Representation | Model | Accuracy (%) | F1-Score |
| :--- | :--- | :---: | :---: |
| **Acoustic Only (123-dim)** | Logistic Regression | 61.46% ± 4.68% | 0.6390 |
| **Acoustic Only (123-dim)** | SVM (RBF) | 52.99% ± 2.67% | 0.6617 |
| **Deep Embeddings Only (512-dim)** | Logistic Regression | 73.51% ± 10.34% | 0.7396 |
| **Deep Embeddings Only (512-dim)** | SVM (RBF) | 54.22% ± 3.38% | 0.6803 |
| **Supervised Gated Attention Fused** | Logistic Regression | **100.00% ± 0.00%** | **1.0000** |
| **Supervised Gated Attention Fused** | SVM (RBF) | **99.39% ± 1.21%** | **0.9943** |

---

## 📁 Repository Structure & Dataset Note

> **Note:** Raw audio files (`.wav` format, ~6.5 GB total) are ignored via `.gitignore` to comply with GitHub repository size and bandwidth limits.

```
├── diagnosis_train/                # Alzheimer's Diagnosis training files & labels
├── progression_train/              # Progression training files & labels
├── progression_test/               # Challenge test set audio & submission template
├── src/
│   ├── preprocessing.py            # Stage 1: Noise reduction, 16kHz resampling, VAD
│   ├── feature_extraction.py       # Stage 2: 123 Acoustic + Deep Embeddings
│   ├── advanced_fusion.py          # Stage 3: Supervised Gated Cross-Attention Fusion
│   ├── models.py                   # Stage 4: SVM, RF, Logistic Reg, PyTorch DNN & 1D-CNN
│   ├── evaluation.py               # Stage 5: 5-Fold Stratified CV, metrics, plot generation
│   ├── multi_task_model.py         # Phase 6: Multi-Task Learning (AD + MMSE Regression)
│   ├── explainability.py           # Phase 7: Explainable AI (SHAP & LIME)
│   ├── cross_validation.py         # Phase 8: Cross-Corpus Validation
│   └── optimization.py             # Phase 9: Optuna Optimization & Ablations
├── results/
│   ├── model_comparison.csv        # Metrics summary CSV
│   ├── ablation_study.csv          # Feature ablation study CSV
│   ├── test_predictions_task3.csv  # Challenge test set predictions
│   └── plots/                      # Saved SHAP, LIME, ROC, CM, and timeline plots
├── requirements.txt                # Python dependencies
├── run_pipeline.py                 # Main execution script
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/divagar127/Alzimer-detection-using-speech-processing.git
   cd Alzimer-detection-using-speech-processing
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete 100% pipeline:**
   ```bash
   python run_pipeline.py
   ```

---

## 📄 License & Citation
Dataset courtesy of the **ADReSSo Challenge (IS2021)**.
