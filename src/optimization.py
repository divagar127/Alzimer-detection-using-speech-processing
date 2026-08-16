"""
Phase 9: Model Optimization & Ablation Study Module
Optuna hyperparameter tuning + Feature Ablation Study (Acoustic-Only vs Deep-Only vs Cross-Attention Fused).
"""

import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


def optimize_hyperparameters(X, y, n_trials=25):
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
            C = trial.suggest_float("C", 1e-3, 1e2, log=True)
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
        return {"SVM": {"C": 1.0, "gamma": "scale"}, "LogisticRegression": {"C": 1.0}}


def run_ablation_study(X_acoustic, X_deep, X_fused, y, output_dir="results"):
    """
    Feature Ablation Study: Compare Acoustic-Only vs Deep-Only vs Gated Attention Fused representations.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42)
    }

    ablation_rows = []

    feature_sets = {
        "Acoustic Only (123-dim)": X_acoustic,
        "Deep Embeddings Only (512-dim)": X_deep,
        "Gated Cross-Attention Fused": X_fused
    }

    print("\n--- Starting Feature Ablation Study ---")
    for feat_name, X_subset in feature_sets.items():
        for model_name, model in models.items():
            accs, f1s = [], []
            for train_idx, val_idx in skf.split(X_subset, y):
                X_tr, X_va = X_subset[train_idx], X_subset[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]

                model.fit(X_tr, y_tr)
                preds = model.predict(X_va)
                accs.append(accuracy_score(y_va, preds))
                f1s.append(f1_score(y_va, preds, zero_division=0))

            ablation_rows.append({
                "Feature Representation": feat_name,
                "Model": model_name,
                "Accuracy (%)": f"{np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%",
                "F1-Score": f"{np.mean(f1s):.4f}",
                "_raw_acc": np.mean(accs)
            })

    df_ablation = pd.DataFrame(ablation_rows)
    csv_path = os.path.join(output_dir, "ablation_study.csv")
    df_ablation.to_csv(csv_path, index=False)
    print(f"Ablation study metrics saved to {csv_path}")

    return df_ablation
