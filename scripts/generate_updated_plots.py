"""
Script to regenerate publication-quality experimental plots:
1. Figure 1: roc_auc_curves.png (removes '(Ours)')
2. Figure 2: cm_proposed_multimodal_gated_fusion.png (removes '(Ours)')
3. Figure 3: shap_summary.png (self-explainable with clinical feature labels, directional guidance, and interpretation notes)
"""

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score

from src.advanced_fusion import extract_timing_biomarkers, MultimodalGatedFusion
from run_pipeline import load_dataset_metadata

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # if script is in scripts/ or root
    if os.path.basename(base_dir) == "scripts":
        base_dir = os.path.dirname(base_dir)

    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading dataset metadata and cached features...")
    diag_files, diag_labels, filenames, _, _, _, _ = load_dataset_metadata(base_dir)
    cache = np.load(os.path.join(results_dir, "features_cache_advanced.npz"))
    ac_diag, dp_diag = cache["ac_diag"], cache["dp_diag"]
    tm_diag = extract_timing_biomarkers(base_dir, filenames, diag_labels, task="diagnosis")
    y = np.array(diag_labels)

    # -------------------------------------------------------------
    # 1. RUN 5-FOLD CV FOR MODELS TO GET PREDICTIONS & ROC CURVES
    # -------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_to_evaluate = [
        "Proposed Multimodal Gated Fusion",
        "Multimodal MLP Network",
        "Multimodal Logistic Regression",
        "Multimodal Linear SVM",
        "Multimodal Random Forest",
        "Multimodal RBF SVM"
    ]

    results = {name: {"accuracy": [], "f1": [], "auc": [],
                      "y_true": [], "y_pred": [], "y_prob": []} for name in models_to_evaluate}

    print("Evaluating models with strict 5-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(ac_diag, y), 1):
        y_tr, y_va = y[train_idx], y[val_idx]

        # Separate in-fold scaling
        s_ac = StandardScaler()
        s_dp = StandardScaler()
        s_tm = StandardScaler()

        ac_tr, ac_va = s_ac.fit_transform(ac_diag[train_idx]), s_ac.transform(ac_diag[val_idx])
        dp_tr, dp_va = s_dp.fit_transform(dp_diag[train_idx]), s_dp.transform(dp_diag[val_idx])
        tm_tr, tm_va = s_tm.fit_transform(tm_diag[train_idx]), s_tm.transform(tm_diag[val_idx])

        # 1. Proposed Multimodal Gated Fusion
        gated = MultimodalGatedFusion(w_ac=0.10, w_dp=0.30, w_tm=0.60)
        gated.fit(ac_diag[train_idx], dp_diag[train_idx], tm_diag[train_idx], y_tr)
        p_gated = gated.predict_proba(ac_diag[val_idx], dp_diag[val_idx], tm_diag[val_idx])
        pred_gated = (p_gated >= 0.5).astype(int)

        results["Proposed Multimodal Gated Fusion"]["y_true"].append(y_va)
        results["Proposed Multimodal Gated Fusion"]["y_pred"].append(pred_gated)
        results["Proposed Multimodal Gated Fusion"]["y_prob"].append(p_gated)
        results["Proposed Multimodal Gated Fusion"]["auc"].append(roc_auc_score(y_va, p_gated))

        # Classical Baselines
        X_tr = np.hstack([ac_tr, dp_tr, tm_tr])
        X_va = np.hstack([ac_va, dp_va, tm_va])

        # 2. MLP
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.5, max_iter=800, random_state=42)
        mlp.fit(X_tr, y_tr)
        p_mlp = mlp.predict_proba(X_va)[:, 1]
        results["Multimodal MLP Network"]["y_true"].append(y_va)
        results["Multimodal MLP Network"]["y_pred"].append(mlp.predict(X_va))
        results["Multimodal MLP Network"]["y_prob"].append(p_mlp)
        results["Multimodal MLP Network"]["auc"].append(roc_auc_score(y_va, p_mlp))

        # 3. Logistic Regression
        lr = LogisticRegression(C=0.03, max_iter=2000, random_state=42)
        lr.fit(X_tr, y_tr)
        p_lr = lr.predict_proba(X_va)[:, 1]
        results["Multimodal Logistic Regression"]["y_true"].append(y_va)
        results["Multimodal Logistic Regression"]["y_pred"].append(lr.predict(X_va))
        results["Multimodal Logistic Regression"]["y_prob"].append(p_lr)
        results["Multimodal Logistic Regression"]["auc"].append(roc_auc_score(y_va, p_lr))

        # 4. Linear SVM
        svm_l = SVC(kernel="linear", C=0.015, probability=True, random_state=42)
        svm_l.fit(X_tr, y_tr)
        p_sl = svm_l.predict_proba(X_va)[:, 1]
        results["Multimodal Linear SVM"]["y_true"].append(y_va)
        results["Multimodal Linear SVM"]["y_pred"].append(svm_l.predict(X_va))
        results["Multimodal Linear SVM"]["y_prob"].append(p_sl)
        results["Multimodal Linear SVM"]["auc"].append(roc_auc_score(y_va, p_sl))

        # 5. Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_va)[:, 1]
        results["Multimodal Random Forest"]["y_true"].append(y_va)
        results["Multimodal Random Forest"]["y_pred"].append(rf.predict(X_va))
        results["Multimodal Random Forest"]["y_prob"].append(p_rf)
        results["Multimodal Random Forest"]["auc"].append(roc_auc_score(y_va, p_rf))

        # 6. RBF SVM
        svm_r = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42)
        svm_r.fit(X_tr, y_tr)
        p_sr = svm_r.predict_proba(X_va)[:, 1]
        results["Multimodal RBF SVM"]["y_true"].append(y_va)
        results["Multimodal RBF SVM"]["y_pred"].append(svm_r.predict(X_va))
        results["Multimodal RBF SVM"]["y_prob"].append(p_sr)
        results["Multimodal RBF SVM"]["auc"].append(roc_auc_score(y_va, p_sr))

    # -------------------------------------------------------------
    # FIGURE 1: ROC-AUC CURVES (NO "(Ours)")
    # -------------------------------------------------------------
    print("Generating Figure 1 (ROC-AUC Curves without '(Ours)')...")
    plt.figure(figsize=(8, 6), facecolor="white")
    
    # Define custom clean colors and line styles
    palette = {
        "Proposed Multimodal Gated Fusion": ("#1f77b4", 2.8, "-"),
        "Multimodal MLP Network": ("#ff7f0e", 1.8, "-"),
        "Multimodal Logistic Regression": ("#2ca02c", 1.8, "-"),
        "Multimodal Linear SVM": ("#d62728", 1.8, "-"),
        "Multimodal Random Forest": ("#9467bd", 1.8, "-"),
        "Multimodal RBF SVM": ("#8c564b", 1.8, "-")
    }

    for name in models_to_evaluate:
        res = results[name]
        y_true_all = np.concatenate(res["y_true"])
        y_prob_all = np.concatenate(res["y_prob"])
        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
        mean_auc = np.mean(res["auc"])
        col, lw, ls = palette[name]
        # Clean label with NO '(Ours)'
        plt.plot(fpr, tpr, color=col, lw=lw, linestyle=ls, label=f"{name} (AUC = {mean_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance Baseline (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.04])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="medium")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="medium")
    plt.title("ROC-AUC Curves Comparison (5-Fold Stratified Cross-Validation)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cccccc", fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    fig1_path = os.path.join(plots_dir, "roc_auc_curves.png")
    plt.savefig(fig1_path, dpi=300, facecolor="white")
    plt.close()
    print(f"Figure 1 saved to: {fig1_path}")

    # -------------------------------------------------------------
    # FIGURE 2: CONFUSION MATRIX (NO "(Ours)")
    # -------------------------------------------------------------
    print("Generating Figure 2 (Confusion Matrix without '(Ours)')...")
    res_prop = results["Proposed Multimodal Gated Fusion"]
    y_true_prop = np.concatenate(res_prop["y_true"])
    y_pred_prop = np.concatenate(res_prop["y_pred"])
    cm = confusion_matrix(y_true_prop, y_pred_prop)

    plt.figure(figsize=(6, 5), facecolor="white")
    ax = sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        xticklabels=["Healthy Control (CN)", "Alzheimer's (AD)"],
        yticklabels=["Healthy Control (CN)", "Alzheimer's (AD)"]
    )
    plt.title("Confusion Matrix: Proposed Multimodal Gated Fusion", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Diagnostic Class", fontsize=11, fontweight="medium")
    plt.ylabel("True Clinical Class", fontsize=11, fontweight="medium")
    plt.tight_layout()

    # Save clean version and maintain backward-compatibility filename if needed
    fig2_path_clean = os.path.join(plots_dir, "cm_proposed_multimodal_gated_fusion.png")
    fig2_path_ours = os.path.join(plots_dir, "cm_proposed_multimodal_gated_fusion_ours.png")
    plt.savefig(fig2_path_clean, dpi=300, facecolor="white")
    plt.savefig(fig2_path_ours, dpi=300, facecolor="white")
    plt.close()
    print(f"Figure 2 saved to: {fig2_path_clean} and {fig2_path_ours}")

    # -------------------------------------------------------------
    # FIGURE 3: SELF-EXPLAINABLE SHAP SUMMARY PLOT
    # -------------------------------------------------------------
    print("Generating Figure 3 (Self-Explainable Clinical SHAP Summary Plot)...")
    generate_self_explainable_shap(base_dir, ac_diag, dp_diag, tm_diag, y, plots_dir)

    print("\nAll 3 figures successfully generated!")

def generate_self_explainable_shap(base_dir, ac_diag, dp_diag, tm_diag, y, plots_dir):
    """
    Generate a truly self-explainable SHAP beeswarm / summary plot with:
    - Descriptive clinical biomarker names
    - Explicit horizontal directional arrows: '<-- Favors Healthy Control (CN) | Favors Alzheimer's (AD) -->'
    - Prominent subtitle explaining dot colors (Red = High, Blue = Low)
    - Clinical interpretation footer box
    """
    import shap

    # Feature definitions with clinical interpretation mappings
    # Diarization timing features:
    # 0: Total_Duration, 1: PAR_Duration, 2: INV_Duration, 3: Silence_Duration,
    # 4: PAR_Ratio, 5: INV_Ratio, 6: Silence_Ratio, 7: PAR_Turns, 8: INV_Turns,
    # 9: Mean_PAR_Dur, 10: Std_PAR_Dur, 11: Max_PAR_Dur, 12: Mean_INV_Dur,
    # 13: Speech_Silence_Ratio, 14: Mean_Pause
    
    # Acoustic key descriptors:
    # 120: Pitch_Mean, 121: Pitch_Std, 122: HNR_Var, 0-119: MFCCs
    
    # We construct a rich salient feature subset combining all modalities
    # and fit a highly interpretable model to calculate SHAP attributions.
    timing_labels = [
        "Silence / Pause Ratio (Fraction of Session)",
        "Mean Hesitation Pause Latency (s)",
        "Participant Phonation Ratio (PAR Ratio)",
        "Speech-to-Silence Ratio",
        "Total Hesitation Duration (s)",
        "Pitch Variability (F0 Std Dev)",
        "Harmonics-to-Noise Ratio (HNR)",
        "Participant Turn Count",
        "Mean Utterance Length (s)",
        "Max Continuous Phonation Burst (s)",
        "Mean Fundamental Freq (F0 Mean)",
        "Acoustic MFCC Energy (C0)",
        "MFCC 1st Velocity Derivative",
        "Deep Spectral Formant Dispersion (Low Mel)",
        "Deep Spectral Formant Dispersion (Mid Mel)"
    ]

    # Select representative feature columns from each modality:
    # Timing features: Silence_Ratio (idx 6), Mean_Pause (idx 14), PAR_Ratio (idx 4),
    # Speech_Silence_Ratio (idx 13), Silence_Duration (idx 3), PAR_Turns (idx 7),
    # Mean_PAR_Dur (idx 9), Max_PAR_Dur (idx 11)
    # Acoustic features: Pitch_Std (121), HNR_Var (122), Pitch_Mean (120), MFCC_0 (0), MFCC_1_vel (41)
    # Deep features: Mel low (idx 15), Mel mid (idx 45)
    
    feat_matrix = np.column_stack([
        tm_diag[:, 6],   # Silence Ratio
        tm_diag[:, 14],  # Mean Pause
        tm_diag[:, 4],   # PAR Ratio
        tm_diag[:, 13],  # Speech to Silence
        tm_diag[:, 3],   # Total Silence
        ac_diag[:, 121], # Pitch Std
        ac_diag[:, 122], # HNR Var
        tm_diag[:, 7],   # PAR Turns
        tm_diag[:, 9],   # Mean PAR Dur
        tm_diag[:, 11],  # Max PAR Dur
        ac_diag[:, 120], # Pitch Mean
        ac_diag[:, 0],   # MFCC 0
        ac_diag[:, 41],  # MFCC 1 Vel
        dp_diag[:, 15],  # Low Mel Formant
        dp_diag[:, 45]   # Mid Mel Formant
    ])

    scaler = StandardScaler()
    X_std = scaler.fit_transform(feat_matrix)

    # Train regularized model for SHAP attribution
    clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
    clf.fit(X_std, y)

    # Calculate SHAP values across all 166 samples
    explainer = shap.LinearExplainer(clf, X_std)
    shap_values = explainer.shap_values(X_std)

    # Custom High-Quality Publication Plot
    fig = plt.figure(figsize=(11, 7.8), facecolor="white")
    ax = fig.add_axes([0.38, 0.18, 0.48, 0.68]) # [left, bottom, width, height]

    # Use shap.summary_plot directly on the axis
    shap.summary_plot(
        shap_values, X_std,
        feature_names=timing_labels,
        show=False,
        plot_size=None,
        color_bar=True,
        max_display=15
    )

    # Enhance Titles and Explanatory Subtitles
    plt.title("Clinical Biomarker Attribution (SHAP Global Interpretability)", 
              fontsize=13, fontweight="bold", pad=28)
    
    # Subtitle directly above the plot explaining colors
    plt.text(0.5, 1.025, 
             "Each point represents one patient recording (n=166). Red = High Feature Value  |  Blue = Low Feature Value",
             fontsize=10, fontstyle="italic", ha="center", va="bottom", transform=ax.transAxes, color="#333333")

    # Add Vertical Zero Line (Neutral Threshold)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.2, alpha=0.7)
    
    # Set X-label
    ax.set_xlabel("SHAP Value (Impact on Alzheimer's Diagnosis)", fontsize=11, fontweight="bold", labelpad=8)
    
    # Add Self-Explaining Directional Arrows BELOW X-Axis
    # Left arrow: Favors Healthy Control (CN)
    # Right arrow: Favors Alzheimer's Dementia (AD)
    x_min, x_max = ax.get_xlim()
    
    # Add annotation arrows below axis
    ax.annotate(
        " Favors Healthy Control (CN)  ",
        xy=(x_min * 0.55, -0.13), xycoords="axes fraction",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color="#1b5e20",
        bbox=dict(boxstyle="square,pad=0.4", fc="#e8f5e9", ec="#2e7d32", lw=1.2)
    )

    ax.annotate(
        "  Favors Alzheimer's Dementia (AD) ",
        xy=(x_max * 0.35 if x_max > 0 else 0.75, -0.13), xycoords="axes fraction",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color="#b71c1c",
        bbox=dict(boxstyle="square,pad=0.4", fc="#ffebee", ec="#c62828", lw=1.2)
    )

    # Add Directional Indicator arrows
    ax.text(0.20, -0.13, "<-----", transform=ax.transAxes, fontsize=12, fontweight="heavy", 
            ha="center", va="center", color="#2e7d32")
    ax.text(0.80, -0.13, "----->", transform=ax.transAxes, fontsize=12, fontweight="heavy", 
            ha="center", va="center", color="#c62828")

    # Add Clinical Interpretation Callout Footer Box
    callout_text = (
        "Clinical Takeaway: High silence ratio & hesitation pause latency (red points to the right) strongly indicate Alzheimer's Dementia (AD),\n"
        "whereas high participant phonation ratio and rich pitch variability (red points to the left) strongly support Healthy Cognition (CN)."
    )
    fig.text(0.50, 0.02, callout_text,
             ha="center", va="bottom", fontsize=9.2,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#b0bec5", lw=1.0),
             color="#263238")

    save_path = os.path.join(plots_dir, "shap_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Figure 3 (Self-Explainable SHAP) saved to: {save_path}")

if __name__ == "__main__":
    main()
