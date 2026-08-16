"""
Phase 5: Advanced Multimodal Fusion Module
Implements Supervised Gated Cross-Attention Fusion for Acoustic (123-dim) and Deep Embeddings (512/1024-dim).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention module for multimodal fusion between acoustic and deep representations.
    """
    def __init__(self, acoustic_dim=123, deep_dim=512, hidden_dim=256, fused_dim=318):
        super(GatedCrossAttention, self).__init__()
        self.hidden_dim = hidden_dim

        # Projection layers to unified hidden space
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.deep_proj = nn.Sequential(
            nn.Linear(deep_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Cross-Attention: Acoustic Query -> Deep Key/Value
        self.q_acoustic = nn.Linear(hidden_dim, hidden_dim)
        self.k_deep = nn.Linear(hidden_dim, hidden_dim)
        self.v_deep = nn.Linear(hidden_dim, hidden_dim)

        # Cross-Attention: Deep Query -> Acoustic Key/Value
        self.q_deep = nn.Linear(hidden_dim, hidden_dim)
        self.k_acoustic = nn.Linear(hidden_dim, hidden_dim)
        self.v_acoustic = nn.Linear(hidden_dim, hidden_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Output projection layer to target fused dimension
        self.fusion_out = nn.Sequential(
            nn.Linear(hidden_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU()
        )

        # Classification head for supervised gating optimization
        self.classifier = nn.Linear(fused_dim, 2)

    def forward(self, acoustic_feats, deep_feats):
        # Project inputs to shared hidden dimension
        a = self.acoustic_proj(acoustic_feats)  # (N, 256)
        d = self.deep_proj(deep_feats)          # (N, 256)

        # 1. Acoustic queries Deep
        qa = self.q_acoustic(a)
        kd = self.k_deep(d)
        vd = self.v_deep(d)
        scores_ad = torch.sum(qa * kd, dim=-1, keepdim=True) / math.sqrt(self.hidden_dim)
        attn_ad = torch.sigmoid(scores_ad)
        attended_deep = attn_ad * vd

        # 2. Deep queries Acoustic
        qd = self.q_deep(d)
        ka = self.k_acoustic(a)
        va = self.v_acoustic(a)
        scores_da = torch.sum(qd * ka, dim=-1, keepdim=True) / math.sqrt(self.hidden_dim)
        attn_da = torch.sigmoid(scores_da)
        attended_acoustic = attn_da * va

        # 3. Gated fusion weighting
        combined = torch.cat([attended_deep, attended_acoustic], dim=-1)
        gate_weights = self.gate(combined)
        fused_hidden = gate_weights * attended_deep + (1 - gate_weights) * attended_acoustic

        # 4. Final projection
        fused_out = self.fusion_out(fused_hidden)
        logits = self.classifier(fused_out)
        return fused_out, logits, gate_weights


class GatedAttentionFusionPipeline:
    def __init__(self, acoustic_dim=123, deep_dim=512, fused_dim=318, epochs=60, lr=0.001):
        self.acoustic_dim = acoustic_dim
        self.deep_dim = deep_dim
        self.fused_dim = fused_dim
        self.epochs = epochs
        self.lr = lr
        self.ac_scaler = StandardScaler()
        self.dp_scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_transform(self, X_acoustic, X_deep, y_labels=None):
        actual_deep_dim = X_deep.shape[1]

        # Standardize inputs
        X_ac_scaled = self.ac_scaler.fit_transform(X_acoustic)
        X_dp_scaled = self.dp_scaler.fit_transform(X_deep)

        target_fused_dim = min(self.fused_dim, X_acoustic.shape[0] - 1)

        self.model = GatedCrossAttention(
            acoustic_dim=self.acoustic_dim,
            deep_dim=actual_deep_dim,
            hidden_dim=256,
            fused_dim=target_fused_dim
        ).to(self.device)

        ac_tensor = torch.tensor(X_ac_scaled, dtype=torch.float32).to(self.device)
        dp_tensor = torch.tensor(X_dp_scaled, dtype=torch.float32).to(self.device)

        if y_labels is not None:
            y_tensor = torch.tensor(y_labels, dtype=torch.long).to(self.device)
            dataset = TensorDataset(ac_tensor, dp_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=min(16, len(X_acoustic)), shuffle=True)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

            # Supervised Gated Attention Training
            self.model.train()
            for epoch in range(self.epochs):
                for batch_ac, batch_dp, batch_y in loader:
                    optimizer.zero_grad()
                    _, logits, _ = self.model(batch_ac, batch_dp)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

        self.model.eval()
        with torch.no_grad():
            fused_out, _, _ = self.model(ac_tensor, dp_tensor)
            return fused_out.cpu().numpy()

    def transform(self, X_acoustic, X_deep):
        """Transform new test/progression data using fitted scalers and model."""
        if self.model is None:
            return self.fit_transform(X_acoustic, X_deep)

        X_ac_scaled = self.ac_scaler.transform(X_acoustic)
        X_dp_scaled = self.dp_scaler.transform(X_deep)

        ac_tensor = torch.tensor(X_ac_scaled, dtype=torch.float32).to(self.device)
        dp_tensor = torch.tensor(X_dp_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            fused_out, _, _ = self.model(ac_tensor, dp_tensor)
            return fused_out.cpu().numpy()
