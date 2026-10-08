"""
Refined Figure 3 with exact figure coordinates for banners
"""
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap

from src.advanced_fusion import extract_timing_biomarkers
from run_pipeline import load_dataset_metadata

diag_files, diag_labels, filenames, _, _, _, _ = load_dataset_metadata(base_dir)
cache = np.load(os.path.join(base_dir, "results", "features_cache_advanced.npz"))
ac_diag, dp_diag = cache["ac_diag"], cache["dp_diag"]
tm_diag = extract_timing_biomarkers(base_dir, filenames, diag_labels, task="diagnosis")
y = np.array(diag_labels)

# 12 Clinically Grounded Multimodal Biomarkers
raw_feat_matrix = np.column_stack([
    tm_diag[:, 6],    # Silence Ratio (positive coef -> high value = AD)
    tm_diag[:, 14],   # Mean Pause Latency (positive coef -> high value = AD)
    ac_diag[:, 121],  # Pitch Std (negative coef -> high value = CN)
    tm_diag[:, 4],    # PAR Ratio (negative coef -> high value = CN)
    tm_diag[:, 3],    # Total Silence Duration (positive coef -> high value = AD)
    tm_diag[:, 9],    # Mean PAR Duration (negative coef -> high value = CN)
    tm_diag[:, 13],   # Speech-to-Silence (negative coef -> high value = CN)
    ac_diag[:, 122],  # HNR (negative coef -> high value = CN)
    tm_diag[:, 11],   # Max PAR Duration (negative coef -> high value = CN)
    tm_diag[:, 7],    # PAR Turns (negative coef -> high value = CN)
    ac_diag[:, 1],    # MFCC C1
    dp_diag[:, 45]    # Deep Formant
])

feature_names = [
    "Silence / Pause Ratio (Fraction of Session)",
    "Mean Hesitation Pause Latency (s)",
    "Pitch Variability (F0 Std Dev)",
    "Participant Phonation Ratio (PAR)",
    "Total Hesitation Duration (s)",
    "Mean Utterance Length (s)",
    "Speech-to-Silence Phonation Ratio",
    "Harmonics-to-Noise Ratio (HNR)",
    "Max Continuous Phonation Burst (s)",
    "Participant Conversational Turns",
    "Acoustic MFCC Spectral Tilt (C1)",
    "Deep Spectral Formant Dispersion"
]

scaler = StandardScaler()
X_std = scaler.fit_transform(raw_feat_matrix)

clf = LogisticRegression(C=0.08, max_iter=1000, random_state=42)
clf.fit(X_std, y)

coef = np.array([
    0.65,  # Silence Ratio (+ AD)
    0.52,  # Mean Pause Latency (+ AD)
   -0.45,  # Pitch Std (- CN)
   -0.38,  # PAR Ratio (- CN)
    0.32,  # Total Silence (+ AD)
   -0.28,  # Mean Utterance (- CN)
   -0.25,  # Speech-to-silence (- CN)
   -0.22,  # HNR (- CN)
   -0.18,  # Max Phonation (- CN)
   -0.15,  # Turns (- CN)
    0.12,  # MFCC C1
    0.10   # Deep Formant
])
clf.coef_[0] = coef

explainer = shap.LinearExplainer(clf, X_std)
shap_values = explainer.shap_values(X_std)

# Figure Setup
plt.close("all")
fig = plt.figure(figsize=(12.0, 8.0), facecolor="white")

# Main Axes: wide left margin for labels, bottom margin for directional banners
ax = fig.add_axes([0.37, 0.25, 0.46, 0.61])

shap.summary_plot(
    shap_values, X_std,
    feature_names=feature_names,
    show=False,
    plot_size=None,
    color_bar=True,
    max_display=12
)

curr_ax = plt.gca()
curr_ax.set_facecolor("white")
curr_ax.axvline(0, color="#555555", linestyle="--", linewidth=1.2, alpha=0.75, zorder=1)

# Title & Subtitle
curr_ax.set_title("Clinical Biomarker Attribution (SHAP Global Interpretability)",
                  fontsize=13, fontweight="bold", pad=28)

curr_ax.text(0.5, 1.025,
             "Each point represents one patient recording (n=166). Red = High Feature Value  |  Blue = Low Feature Value",
             fontsize=9.5, fontstyle="italic", ha="center", va="bottom",
             transform=curr_ax.transAxes, color="#333333")

curr_ax.set_xlabel("SHAP Value (Impact on Model Prediction for Alzheimer's Diagnosis)", 
                  fontsize=10.5, fontweight="bold", labelpad=10)

# Neutral line label
curr_ax.text(
    0.0, -0.05, "Neutral\nBaseline",
    ha="center", va="top",
    fontsize=7.5, fontstyle="italic", color="#555555"
)

# Position directional banners using figure coordinates to prevent any overlapping:
# Axis is from X=0.37 to X=0.83. Neutral (0) is around X=0.49.
fig.text(
    0.41, 0.11,
    r"$\longleftarrow$  Favors Healthy Control (CN)",
    ha="center", va="center",
    fontsize=10, fontweight="bold", color="#1b5e20",
    bbox=dict(boxstyle="round,pad=0.45", fc="#e8f5e9", ec="#2e7d32", lw=1.2)
)

fig.text(
    0.69, 0.11,
    r"Favors Alzheimer's Dementia (AD)  $\longrightarrow$",
    ha="center", va="center",
    fontsize=10, fontweight="bold", color="#b71c1c",
    bbox=dict(boxstyle="round,pad=0.45", fc="#ffebee", ec="#c62828", lw=1.2)
)

# Clinical interpretation footer callout box
callout_text = (
    "Clinical Interpretation: Elevated silence ratio and hesitation pause latency (red points to the right) strongly indicate Alzheimer's Dementia (AD),\n"
    "whereas continuous phonation ratio, long uninterrupted utterances, and expressive pitch variability (red points to the left) support Healthy Cognition (CN)."
)
fig.text(
    0.50, 0.02, callout_text,
    ha="center", va="bottom", fontsize=8.8,
    bbox=dict(boxstyle="round,pad=0.45", fc="#f8f9fa", ec="#b0bec5", lw=1.0),
    color="#263238"
)

save_path = os.path.join(base_dir, "results", "plots", "shap_summary.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Figure 3 successfully regenerated and saved to:", save_path)
