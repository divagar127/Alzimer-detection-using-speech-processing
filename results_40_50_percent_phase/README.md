# Results & Evaluation Artifacts - 40% to 50% Project Phase

This directory contains all experimental evaluation metrics, confusion matrix plots, ROC-AUC curve plots, and comparison tables achieved during the **40% to 50% completion phase** of the Alzheimer's Detection Speech Processing Project.

---

## 📌 Phase Overview & Methodology

In the 40-50% phase, the pipeline implemented:
1. **Stage 1 (Preprocessing):** Audio loading, 16kHz resampling, spectral gating noise reduction, and VAD silence removal.
2. **Stage 2 (Feature Extraction):** 123 Acoustic Features (40 MFCCs + 40 Deltas + 40 Delta-Deltas, Pitch F0 Mean/Std, HNR) + Deep Embeddings (Whisper 1024-dim).
3. **Stage 3 (Feature Fusion):** Concatenation (123 + 1024 = 1147-dim) standardized and reduced via PCA to **318 dimensions**.
4. **Stage 4 & 5 (Classification & 5-Fold CV):** Evaluated across 5 Stratified Cross-Validation Folds using SVM (RBF), Random Forest, Logistic Regression, PyTorch DNN, and PyTorch 1D-CNN.

---

## 📊 5-Fold Stratified Cross-Validation Results Table (40-50% Phase)

| Model Architecture | Feature Fusion Method | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | PCA Reduction (318-dim) | **70.52% ± 8.06%** | **0.7217** | **0.7111** | **0.7117** | **0.7815** |
| **PyTorch DNN (3-Layer)** | PCA Reduction (318-dim) | **70.48% ± 9.07%** | **0.7208** | **0.7235** | **0.7209** | **0.7568** |
| **SVM (RBF Kernel)** | PCA Reduction (318-dim) | **66.29% ± 9.54%** | 0.6665 | 0.7346 | 0.6950 | 0.7440 |
| **Random Forest** | PCA Reduction (318-dim) | **54.80% ± 8.67%** | 0.5566 | 0.6562 | 0.5967 | 0.5285 |
| **PyTorch 1D-CNN** | PCA Reduction (318-dim) | **54.21% ± 7.80%** | 0.5644 | 0.6458 | 0.5924 | 0.5732 |

---

## 📈 Benchmark Comparison with Official ADReSSo Challenge Baselines

| Benchmark Study / Model | Acoustic / Feature Input | Accuracy (%) | F1-Score |
| :--- | :--- | :---: | :---: |
| **Official Acoustic Baseline (IS2021 ADReSSo)** | eGeMAPS (Acoustic) | 65.1% | 0.640 |
| **Official Linguistic Baseline (IS2021 ADReSSo)** | Transcripts (BERT) | 76.7% | 0.765 |
| **Wav2Vec2 Fine-tuned (Papasavvas et al.)** | Wav2Vec2 | 78.2% | 0.779 |
| **Our 40-50% Pipeline (PCA + Logistic Regression)** | Acoustic (123) + Whisper (1024) | **70.52%** | **0.7117** |
| **Our 40-50% Pipeline (PCA + PyTorch DNN)** | Acoustic (123) + Whisper (1024) | **70.48%** | **0.7209** |
| **Our 40-50% Pipeline (PCA + SVM RBF)** | Acoustic (123) + Whisper (1024) | **66.29%** | **0.6950** |

---

## 🖼️ Saved Plot Visualizations (40-50% Phase)

All 5 confusion matrix heatmaps and the ROC-AUC curve plot for the 40-50% phase are stored in `results_40_50_percent_phase/plots/`:

- 📈 **ROC-AUC Curves Plot:** [`plots/roc_auc_curves_phase1.png`](file:///d:/ADReSSo_Recovered/results_40_50_percent_phase/plots/roc_auc_curves_phase1.png)
- 🔷 **Logistic Regression Confusion Matrix:** [`plots/cm_logistic_regression_phase1.png`](file:///d:/ADReSSo_Recovered/results_40_50_percent_phase/plots/cm_logistic_regression_phase1.png)
- 🧠 **PyTorch DNN Confusion Matrix:** [`plots/cm_dnn_phase1.png`](file:///d:/ADReSSo_Recovered/results_40_50_percent_phase/plots/cm_dnn_phase1.png)
- ⚡ **SVM RBF Confusion Matrix:** [`plots/cm_svm_rbf_phase1.png`](file:///d:/ADReSSo_Recovered/results_40_50_percent_phase/plots/cm_svm_rbf_phase1.png)
- 🌲 **Random Forest Confusion Matrix:** [`plots/cm_random_forest_phase1.png`](file:///d:/ADReSSo_Recovered/results_40_50_percent_phase/plots/cm_random_forest_phase1.png)
- 🌊 **PyTorch 1D-CNN Confusion Matrix:** [`plots/cm_1dcnn_phase1.png`](file:///d:/ADReSSo_Recovered/results_40_50_percent_phase/plots/cm_1dcnn_phase1.png)
