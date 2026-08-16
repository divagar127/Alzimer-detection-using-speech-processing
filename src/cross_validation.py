"""
Phase 8: Cross-Corpus Validation Module
Tests generalizability by training on Diagnosis Task (AD vs CN) and evaluating zero-shot on Progression Task (Decline vs No Decline).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.models import ModelFactory


def cross_corpus_evaluation(X_train_diag, y_train_diag, X_test_prog, y_test_prog):
    """
    Train models on Diagnosis task, test zero-shot on Progression task.
    :param X_train_diag: Diagnosis feature matrix (N_diag, num_feats)
    :param y_train_diag: Diagnosis labels (0: CN, 1: AD)
    :param X_test_prog: Progression feature matrix (N_prog, num_feats)
    :param y_test_prog: Progression labels (0: no_decline, 1: decline)
    :return: DataFrame of Cross-Corpus results & degradation analysis
    """
    ml_models = ModelFactory.get_ml_models()
    results_rows = []

    print("\n--- Starting Cross-Corpus Generalization Evaluation ---")
    print(f"Training set (Diagnosis Task):   {X_train_diag.shape[0]} samples")
    print(f"Testing set  (Progression Task): {X_test_prog.shape[0]} samples")

    for name, model in ml_models.items():
        # 1. Fit on Diagnosis dataset
        model.fit(X_train_diag, y_train_diag)

        # 2. In-corpus self-eval score
        diag_preds = model.predict(X_train_diag)
        diag_acc = accuracy_score(y_train_diag, diag_preds)

        # 3. Cross-corpus evaluation on Progression dataset
        prog_preds = model.predict(X_test_prog)
        prog_probs = model.predict_proba(X_test_prog)[:, 1] if hasattr(model, "predict_proba") else prog_preds

        acc = accuracy_score(y_test_prog, prog_preds)
        prec = precision_score(y_test_prog, prog_preds, zero_division=0)
        rec = recall_score(y_test_prog, prog_preds, zero_division=0)
        f1 = f1_score(y_test_prog, prog_preds, zero_division=0)

        try:
            auc = roc_auc_score(y_test_prog, prog_probs)
        except Exception:
            auc = 0.5

        # Degradation percentage
        degradation = max(0.0, (diag_acc - acc) * 100)

        results_rows.append({
            "Model": name,
            "In-Corpus Acc (%)": f"{diag_acc*100:.2f}%",
            "Cross-Corpus Acc (%)": f"{acc*100:.2f}%",
            "Degradation": f"{degradation:.2f}%",
            "Precision": f"{prec:.4f}",
            "Recall": f"{rec:.4f}",
            "F1-Score": f"{f1:.4f}",
            "ROC-AUC": f"{auc:.4f}"
        })

    df_cross = pd.DataFrame(results_rows)
    return df_cross
