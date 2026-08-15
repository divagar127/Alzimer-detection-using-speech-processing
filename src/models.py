"""
Stage 4: Classification Module
Models:
1. Machine Learning:
   - SVM (RBF kernel)
   - Random Forest
   - Logistic Regression
2. Deep Learning:
   - PyTorch DNN (3 hidden layers with BatchNorm & Dropout)
   - PyTorch 1D-CNN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ==========================================
# 1. DEEP LEARNING MODELS (PyTorch)
# ==========================================

class PyTorchDNN(nn.Module):
    """Deep Neural Network with 3 hidden layers, BatchNorm, ReLU, Dropout."""
    def __init__(self, input_dim=318, num_classes=2):
        super(PyTorchDNN, self).__init__()
        self.net = nn.Sequential(
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
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class PyTorch1DCNN(nn.Module):
    """1D Convolutional Neural Network for feature sequence classification."""
    def __init__(self, input_dim=318, num_classes=2):
        super(PyTorch1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Output feature size after pooling twice
        conv_out_dim = (input_dim // 4) * 64
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (N, 318) -> reshape to (N, 1, 318)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.pool1(self.relu(self.bn1(self.conv1(x))))
        out = self.pool2(self.relu(self.bn2(self.conv2(out))))
        out = out.view(out.size(0), -1)
        out = self.fc(self.dropout(out))
        return out


# ==========================================
# 2. MODEL WRAPPER & FACTORY
# ==========================================

class ModelFactory:
    @staticmethod
    def get_ml_models():
        """Returns standard Scikit-Learn Machine Learning Classifiers."""
        return {
            "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }

    @staticmethod
    def train_dl_model(model_name, X_train, y_train, X_val, input_dim=318, epochs=40, batch_size=16, lr=0.001):
        """
        Train PyTorch Deep Learning model (DNN or 1DCNN) and return predictions + probabilities.
        :param model_name: 'DNN' or '1DCNN'
        :param X_train: numpy array (N_train, feature_dim)
        :param y_train: numpy array (N_train,)
        :param X_val: numpy array (N_val, feature_dim)
        :return: (y_pred, y_prob)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name.upper() == "DNN":
            model = PyTorchDNN(input_dim=input_dim, num_classes=2).to(device)
        else:
            model = PyTorch1DCNN(input_dim=input_dim, num_classes=2).to(device)

        # Convert to Tensors
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # Training loop
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Inference on Validation set
        model.eval()
        X_v_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_v_t)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        return preds, probs
