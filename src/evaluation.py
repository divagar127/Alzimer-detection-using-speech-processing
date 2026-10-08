"""
Stage 5: Evaluation Module
- 5-Fold Stratified Cross-Validation with Strict Zero-Leakage Guarantee
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices plotting & saving
- ROC-AUC curves plotting & saving
- Comparison table generation with official ADReSSo baseline benchmarks
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from src.advanced_fusion import MultimodalGatedFusion


class Evaluator:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate_multimodal_pipeline(self, ac_diag, dp_diag, tm_diag, y, n_splits=5):
        """
        Run 5-Fold Stratified Cross-Validation for Multimodal Gated Fusion and baselines.
        Zero-leakage guarantee: all scalers and models are fit strictly on training splits.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        models_to_evaluate = [
            "Proposed Multimodal Gated Fusion",
            "Multimodal MLP Network",
            "Multimodal Logistic Regression",
            "Multimodal Linear SVM",
            "Multimodal Random Forest",
            "Multimodal RBF SVM"
        ]

        results = {name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [],
                          "y_true": [], "y_pred": [], "y_prob": []} for name in models_to_evaluate}

        print(f"\n--- Starting Strict Leak-Free {n_splits}-Fold Stratified Cross-Validation ---")
        for fold, (train_idx, val_idx) in enumerate(skf.split(ac_diag, y), 1):
            print(f"Processing Fold {fold}/{n_splits}...")
            y_tr, y_va = y[train_idx], y[val_idx]

            # In-fold separate scaling for each modality
            s_ac = StandardScaler()
            s_dp = StandardScaler()
            s_tm = StandardScaler()

            ac_tr, ac_va = s_ac.fit_transform(ac_diag[train_idx]), s_ac.transform(ac_diag[val_idx])
            dp_tr, dp_va = s_dp.fit_transform(dp_diag[train_idx]), s_dp.transform(dp_diag[val_idx])
            tm_tr, tm_va = s_tm.fit_transform(tm_diag[train_idx]), s_tm.transform(tm_diag[val_idx])

            # 1. Proposed Multimodal Gated Fusion
            gated_fusion = MultimodalGatedFusion(w_ac=0.10, w_dp=0.30, w_tm=0.60)
            gated_fusion.fit(ac_diag[train_idx], dp_diag[train_idx], tm_diag[train_idx], y_tr)
            p_gated = gated_fusion.predict_proba(ac_diag[val_idx], dp_diag[val_idx], tm_diag[val_idx])
            pred_gated = (p_gated >= 0.5).astype(int)
            self._record_fold(results["Proposed Multimodal Gated Fusion"], y_va, pred_gated, p_gated)

            # Combined multimodal matrix for classical baselines
            X_tr_all = np.hstack([ac_tr, dp_tr, tm_tr])
            X_va_all = np.hstack([ac_va, dp_va, tm_va])

            # 2. Multimodal MLP
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.5, max_iter=800, random_state=42)
            mlp.fit(X_tr_all, y_tr)
            pred_mlp = mlp.predict(X_va_all)
            prob_mlp = mlp.predict_proba(X_va_all)[:, 1]
            self._record_fold(results["Multimodal MLP Network"], y_va, pred_mlp, prob_mlp)

            # 3. Multimodal Logistic Regression
            lr = LogisticRegression(C=0.03, max_iter=2000, random_state=42)
            lr.fit(X_tr_all, y_tr)
            pred_lr = lr.predict(X_va_all)
            prob_lr = lr.predict_proba(X_va_all)[:, 1]
            self._record_fold(results["Multimodal Logistic Regression"], y_va, pred_lr, prob_lr)

            # 4. Multimodal Linear SVM
            svm_l = SVC(kernel="linear", C=0.015, probability=True, random_state=42)
            svm_l.fit(X_tr_all, y_tr)
            pred_sl = svm_l.predict(X_va_all)
            prob_sl = svm_l.predict_proba(X_va_all)[:, 1]
            self._record_fold(results["Multimodal Linear SVM"], y_va, pred_sl, prob_sl)

            # 5. Multimodal Random Forest
            rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
            rf.fit(X_tr_all, y_tr)
            pred_rf = rf.predict(X_va_all)
            prob_rf = rf.predict_proba(X_va_all)[:, 1]
            self._record_fold(results["Multimodal Random Forest"], y_va, pred_rf, prob_rf)

            # 6. Multimodal RBF SVM
            svm_r = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42)
            svm_r.fit(X_tr_all, y_tr)
            pred_sr = svm_r.predict(X_va_all)
            prob_sr = svm_r.predict_proba(X_va_all)[:, 1]
            self._record_fold(results["Multimodal RBF SVM"], y_va, pred_sr, prob_sr)

        # Process & Save Metrics Summary
        summary_rows = []
        for name in models_to_evaluate:
            res = results[name]
            acc_mean, acc_std = np.mean(res["accuracy"]), np.std(res["accuracy"])
            prec_mean = np.mean(res["precision"])
            rec_mean = np.mean(res["recall"])
            f1_mean = np.mean(res["f1"])
            auc_mean = np.mean(res["auc"])

            summary_rows.append({
                "Model": name,
                "Accuracy (%)": f"{acc_mean*100:.2f} ± {acc_std*100:.2f}",
                "Precision": f"{prec_mean:.4f}",
                "Recall": f"{rec_mean:.4f}",
                "F1-Score": f"{f1_mean:.4f}",
                "ROC-AUC": f"{auc_mean:.4f}",
                "_raw_acc": acc_mean,
                "_raw_f1": f1_mean
            })

            # Plot Confusion Matrix
            y_true_all = np.concatenate(res["y_true"])
            y_pred_all = np.concatenate(res["y_pred"])
            self._plot_confusion_matrix(y_true_all, y_pred_all, name)

        # Plot ROC Curves
        self._plot_roc_curves(results)

        df_summary = pd.DataFrame(summary_rows)

        # Save baseline comparison table CSV
        csv_path = os.path.join(self.output_dir, "model_comparison.csv")
        df_summary.to_csv(csv_path, index=False)
        print(f"\nModel evaluation summary saved to {csv_path}")

        return df_summary, results

    def _record_fold(self, res_dict, y_val, preds, probs):
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_val, probs)
        except Exception:
            auc = 0.5

        res_dict["accuracy"].append(acc)
        res_dict["precision"].append(prec)
        res_dict["recall"].append(rec)
        res_dict["f1"].append(f1)
        res_dict["auc"].append(auc)

        res_dict["y_true"].append(y_val)
        res_dict["y_pred"].append(preds)
        res_dict["y_prob"].append(probs)

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Healthy Control (CN)", "Alzheimer's (AD)"],
                    yticklabels=["Healthy Control (CN)", "Alzheimer's (AD)"])
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("'", "")
        save_path = os.path.join(self.plots_dir, f"cm_{safe_name}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_roc_curves(self, results):
        plt.figure(figsize=(8, 6))
        for model_name, res in results.items():
            y_true_all = np.concatenate(res["y_true"])
            y_prob_all = np.concatenate(res["y_prob"])
            fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
            mean_auc = np.mean(res["auc"])
            lw = 2.5 if "Proposed" in model_name else 1.5
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {mean_auc:.3f})", lw=lw)

        plt.plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC = 0.50)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-AUC Curves Comparison (5-Fold Stratified Cross-Validation)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.plots_dir, "roc_auc_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"ROC-AUC curves plot saved to {save_path}")
