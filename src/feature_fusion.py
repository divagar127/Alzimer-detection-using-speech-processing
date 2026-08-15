"""
Stage 3: Feature Fusion Module
- Concatenates Acoustic (123-dim) + Deep Embeddings (Whisper 1024-dim = 1147 total)
- Dimensionality reduction via Autoencoder / PCA
- Final feature vector: ~318 dimensions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Autoencoder(nn.Module):
    def __init__(self, input_dim=1147, target_dim=318):
        super(Autoencoder, self).__init__()
        # Encoder: 1147 -> 600 -> 318
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 600),
            nn.BatchNorm1d(600),
            nn.LeakyReLU(0.2),
            nn.Linear(600, target_dim),
            nn.BatchNorm1d(target_dim)
        )
        # Decoder: 318 -> 600 -> 1147
        self.decoder = nn.Sequential(
            nn.Linear(target_dim, 600),
            nn.BatchNorm1d(600),
            nn.LeakyReLU(0.2),
            nn.Linear(600, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class FeatureFusion:
    def __init__(self, target_dim=318, method="pca"):
        """
        Initialize Feature Fusion.
        :param target_dim: Desired fused feature dimension (~318)
        :param method: Reduction method ('pca' or 'autoencoder')
        """
        self.target_dim = target_dim
        self.method = method.lower()
        self.scaler = StandardScaler()
        self.pca = None
        self.autoencoder = None

    def concatenate_features(self, acoustic_feats, deep_feats):
        """
        Concatenate Acoustic (123) and Deep embeddings (1024).
        :param acoustic_feats: numpy array (N, 123)
        :param deep_feats: numpy array (N, 1024)
        :return: numpy array (N, 1147)
        """
        acoustic_feats = np.atleast_2d(acoustic_feats)
        deep_feats = np.atleast_2d(deep_feats)
        fused = np.hstack([acoustic_feats, deep_feats])
        return fused

    def fit_transform(self, X_fused):
        """
        Fit scaler and reduction model (PCA or Autoencoder) and transform X_fused.
        :param X_fused: Raw concatenated features (N, 1147)
        :return: Reduced features (N, 318)
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_fused)
        n_samples, n_features = X_scaled.shape
        actual_target_dim = min(self.target_dim, n_samples - 1, n_features)

        if self.method == "pca" or n_samples < 10:
            self.pca = PCA(n_components=actual_target_dim, random_state=42)
            X_reduced = self.pca.fit_transform(X_scaled)
        elif self.method == "autoencoder":
            self.autoencoder = Autoencoder(input_dim=n_features, target_dim=actual_target_dim)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)

            self.autoencoder.train()
            for epoch in range(50):
                optimizer.zero_grad()
                encoded, decoded = self.autoencoder(X_tensor)
                loss = criterion(decoded, X_tensor)
                loss.backward()
                optimizer.step()

            self.autoencoder.eval()
            with torch.no_grad():
                encoded, _ = self.autoencoder(X_tensor)
                X_reduced = encoded.numpy()

        return X_reduced

    def transform(self, X_fused):
        """
        Transform new data using fitted scaler and reduction model.
        :param X_fused: Concatenated features (N, 1147)
        :return: Reduced features (N, 318)
        """
        X_scaled = self.scaler.transform(X_fused)

        if self.pca is not None:
            return self.pca.transform(X_scaled)
        elif self.autoencoder is not None:
            self.autoencoder.eval()
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                encoded, _ = self.autoencoder(X_tensor)
                return encoded.numpy()
        else:
            return X_scaled[:, :self.target_dim]
