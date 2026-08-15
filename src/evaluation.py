"""
Stage 5: Evaluation Module
- 5-Fold Stratified Cross-Validation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices plotting & saving
- ROC-AUC curves plotting & saving
- Comparison table generation with baseline paper benchmarks
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from src.models import ModelFactory


class Evaluator:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def evaluate_pipeline(self, X, y, n_splits=5):
        """
        Run 5-fold Stratified Cross-Validation for all Machine Learning and Deep Learning models.
        :param X: Feature matrix (N, num_features)
        :param y: Label vector (N,)
        :param n_splits: Number of CV folds (default 5)
        :return: DataFrame summarizing all evaluation metrics
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        ml_models = ModelFactory.get_ml_models()
        dl_model_names = ["DNN", "1DCNN"]

        all_model_names = list(ml_models.keys()) + dl_model_names
        results = {name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [],
                          "y_true": [], "y_pred": [], "y_prob": []} for name in all_model_names}

        print(f"\n--- Starting {n_splits}-Fold Stratified Cross-Validation ---")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"Processing Fold {fold}/{n_splits}...")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 1. Evaluate Machine Learning Models
            for name, model in ml_models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else preds

                self._record_fold(results[name], y_val, preds, probs)

            # 2. Evaluate Deep Learning Models
            for dl_name in dl_model_names:
                preds, probs = ModelFactory.train_dl_model(dl_name, X_train, y_train, X_val, input_dim=X.shape[1])
                self._record_fold(results[dl_name], y_val, preds, probs)

        # Process & Save Metrics Summary
        summary_rows = []
        for name in all_model_names:
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

        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
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
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {mean_auc:.3f})", lw=2)

        plt.plot([0, 1], [0, 1], "k--", label="Random Baseline (AUC = 0.50)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-AUC Curves Comparison")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.plots_dir, "roc_auc_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"ROC-AUC curves plot saved to {save_path}")
