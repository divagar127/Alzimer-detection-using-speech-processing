"""
Phase 9: Model Optimization & Ablation Study Module
Optuna hyperparameter tuning + Feature Ablation Study across modal configurations.
Guarantees strict zero-leakage cross-validation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from src.advanced_fusion import MultimodalGatedFusion


def optimize_hyperparameters(X, y, n_trials=20):
    """
    Optuna hyperparameter optimization for SVM & Logistic Regression.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # 1. Optimize SVM (RBF Kernel)
        def svm_objective(trial):
            C = trial.suggest_float("C", 1e-2, 1e2, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
            model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            return scores.mean()

        svm_study = optuna.create_study(direction="maximize")
        svm_study.optimize(svm_objective, n_trials=n_trials)
        best_svm = svm_study.best_params

        # 2. Optimize Logistic Regression
        def lr_objective(trial):
            C = trial.suggest_float("C", 1e-3, 1e1, log=True)
            model = LogisticRegression(C=C, max_iter=1000, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
            return scores.mean()

        lr_study = optuna.create_study(direction="maximize")
        lr_study.optimize(lr_objective, n_trials=n_trials)
        best_lr = lr_study.best_params

        print(f"Optuna Best SVM params: {best_svm} (Acc: {svm_study.best_value*100:.2f}%)")
        print(f"Optuna Best LR params:  {best_lr} (Acc: {lr_study.best_value*100:.2f}%)")

        return {"SVM": best_svm, "LogisticRegression": best_lr}
    except Exception as e:
        print(f"Optuna optimization notice: {e}")
        return {"SVM": {"C": 2.0, "gamma": "scale"}, "LogisticRegression": {"C": 0.03}}


def run_ablation_study(X_acoustic, X_deep, X_timing, y, output_dir="results"):
    """
    Feature Ablation Study: Compare single modalities vs Early Concatenation vs Proposed Multimodal Gated Fusion.
    Strictly evaluated via 5-Fold Stratified Cross-Validation with zero leakage.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Combined matrix for early fusion
    X_early = np.hstack([X_acoustic, X_deep, X_timing])

    feature_modalities = {
        "Acoustic Only (123-dim)": X_acoustic,
        "Deep Embeddings Only (512-dim)": X_deep,
        "Conversational Timing Biomarkers (15-dim)": X_timing,
        "Early Multimodal Concatenation (650-dim)": X_early
    }

    ablation_rows = []

    print("\n--- Starting Strict Leak-Free Feature Ablation Study ---")
    for feat_name, X_subset in feature_modalities.items():
        # Test Logistic Regression
        accs_lr, f1s_lr = [], []
        accs_svm, f1s_svm = [], []
        for train_idx, val_idx in skf.split(X_subset, y):
            s = StandardScaler()
            X_tr = s.fit_transform(X_subset[train_idx])
            X_va = s.transform(X_subset[val_idx])
            y_tr, y_va = y[train_idx], y[val_idx]

            # LR
            lr = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
            lr.fit(X_tr, y_tr)
            p_lr = lr.predict(X_va)
            accs_lr.append(accuracy_score(y_va, p_lr))
            f1s_lr.append(f1_score(y_va, p_lr, zero_division=0))

            # SVM
            svm = SVC(kernel="rbf", C=1.5, gamma="scale", random_state=42)
            svm.fit(X_tr, y_tr)
            p_svm = svm.predict(X_va)
            accs_svm.append(accuracy_score(y_va, p_svm))
            f1s_svm.append(f1_score(y_va, p_svm, zero_division=0))

        ablation_rows.append({
            "Feature Representation": feat_name,
            "Model": "Logistic Regression",
            "Accuracy (%)": f"{np.mean(accs_lr)*100:.2f}% ± {np.std(accs_lr)*100:.2f}%",
            "F1-Score": f"{np.mean(f1s_lr):.4f}",
            "_raw_acc": np.mean(accs_lr)
        })
        ablation_rows.append({
            "Feature Representation": feat_name,
            "Model": "SVM (RBF)",
            "Accuracy (%)": f"{np.mean(accs_svm)*100:.2f}% ± {np.std(accs_svm)*100:.2f}%",
            "F1-Score": f"{np.mean(f1s_svm):.4f}",
            "_raw_acc": np.mean(accs_svm)
        })

    # Evaluate Proposed Multimodal Gated Fusion
    accs_gated, f1s_gated = [], []
    for train_idx, val_idx in skf.split(X_acoustic, y):
        gated = MultimodalGatedFusion(w_ac=0.10, w_dp=0.30, w_tm=0.60)
        gated.fit(X_acoustic[train_idx], X_deep[train_idx], X_timing[train_idx], y[train_idx])
        pred_gated = gated.predict(X_acoustic[val_idx], X_deep[val_idx], X_timing[val_idx])
        accs_gated.append(accuracy_score(y[val_idx], pred_gated))
        f1s_gated.append(f1_score(y[val_idx], pred_gated, zero_division=0))

    ablation_rows.append({
        "Feature Representation": "Proposed Multimodal Gated Fusion (Ours)",
        "Model": "Gated Fusion Network",
        "Accuracy (%)": f"{np.mean(accs_gated)*100:.2f}% ± {np.std(accs_gated)*100:.2f}%",
        "F1-Score": f"{np.mean(f1s_gated):.4f}",
        "_raw_acc": np.mean(accs_gated)
    })

    df_ablation = pd.DataFrame(ablation_rows)
    csv_path = os.path.join(output_dir, "ablation_study.csv")
    df_ablation.to_csv(csv_path, index=False)
    print(f"Ablation study metrics saved to {csv_path}")

    return df_ablation
