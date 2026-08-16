"""
Phase 6: Multi-Task Learning Module
- Shared PyTorch Backbone (256 -> 128 -> 64)
- Head 1: Binary Alzheimer's Classification (AD vs CN)
- Head 2: Continuous MMSE Score Regression (0 - 30 scale)
- Combined Multi-Task Loss: BCE_loss + 0.5 * MSE_loss
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, root_mean_squared_error


def load_mmse_scores(base_dir):
    """
    Parse MMSE clinical scores from adresso-train-mmse-scores.csv
    :param base_dir: Root directory of project
    :return: dict mapping filename -> mmse_score
    """
    csv_path = os.path.join(base_dir, "diagnosis_train", "train", "adresso-train-mmse-scores.csv")
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    # Columns: adressfname, mmse, dx
    score_map = {}
    for idx, row in df.iterrows():
        fname = str(row['adressfname']).strip()
        score = float(row['mmse'])
        score_map[fname] = score
    return score_map


class MultiTaskDNN(nn.Module):
    """Dual-head DNN for simultaneous AD Classification and MMSE Regression."""
    def __init__(self, input_dim=318):
        super(MultiTaskDNN, self).__init__()
        # Shared feature representation backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Head 1: Binary Classification (AD vs CN)
        self.classifier = nn.Linear(64, 1)

        # Head 2: MMSE Score Regression (0 - 30)
        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        shared_feats = self.shared(x)
        cls_logits = self.classifier(shared_feats).squeeze(-1)
        cls_prob = torch.sigmoid(cls_logits)
        mmse_pred = self.regressor(shared_feats).squeeze(-1)
        return cls_prob, mmse_pred


def multi_task_loss(cls_prob, cls_target, mmse_pred, mmse_target, alpha=0.5):
    """Joint Multi-Task Loss: BCE + alpha * MSE"""
    bce_loss = F.binary_cross_entropy(cls_prob, cls_target.float())
    mse_loss = F.mse_loss(mmse_pred, mmse_target.float())
    return bce_loss + alpha * mse_loss, bce_loss, mse_loss


def train_eval_multitask(X, y_cls, y_mmse, n_splits=5, epochs=50, lr=0.001):
    """
    Run 5-Fold Stratified CV for Multi-Task Learning.
    :return: dict of classification & regression evaluation metrics
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accs, precs, recs, f1s, aucs, rmses = [], [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cls), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_c_tr, y_c_val = y_cls[train_idx], y_cls[val_idx]
        y_m_tr, y_m_val = y_mmse[train_idx], y_mmse[val_idx]

        model = MultiTaskDNN(input_dim=X.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # Convert to PyTorch Tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_c_t = torch.tensor(y_c_tr, dtype=torch.float32)
        y_m_t = torch.tensor(y_m_tr, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_c_t, y_m_t)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for bx, bc, bm in loader:
                bx, bc, bm = bx.to(device), bc.to(device), bm.to(device)
                optimizer.zero_grad()
                p_c, p_m = model(bx)
                total_loss, _, _ = multi_task_loss(p_c, bc, p_m, bm)
                total_loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_v_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            p_c_v, p_m_v = model(X_v_t)
            p_c_np = p_c_v.cpu().numpy()
            preds_np = (p_c_np >= 0.5).astype(int)
            p_m_np = p_m_v.cpu().numpy()

        accs.append(accuracy_score(y_c_val, preds_np))
        precs.append(precision_score(y_c_val, preds_np, zero_division=0))
        recs.append(recall_score(y_c_val, preds_np, zero_division=0))
        f1s.append(f1_score(y_c_val, preds_np, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_c_val, p_c_np))
        except Exception:
            aucs.append(0.5)

        rmse = root_mean_squared_error(y_m_val, p_m_np)
        rmses.append(rmse)

    return {
        "Accuracy (%)": f"{np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%",
        "Precision": f"{np.mean(precs):.4f}",
        "Recall": f"{np.mean(recs):.4f}",
        "F1-Score": f"{np.mean(f1s):.4f}",
        "ROC-AUC": f"{np.mean(aucs):.4f}",
        "MMSE RMSE": f"{np.mean(rmses):.2f}",
        "_raw_acc": np.mean(accs),
        "_raw_f1": np.mean(f1s)
    }
