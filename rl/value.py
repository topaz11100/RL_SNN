import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _CNNFront(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, padding=2, stride=1)

    def forward(self, spike_history: torch.Tensor) -> torch.Tensor:
        if spike_history.dim() != 3:
            raise ValueError("spike_history must have shape (batch, 2, L)")
        x = F.relu(self.conv1(spike_history.float()))
        x = F.relu(self.conv2(x))
        return x.mean(dim=2)


class ValueFunction(nn.Module):
    def __init__(self, extra_feature_dim: int = 0):
        super().__init__()
        self.encoder = _CNNFront()
        input_dim = 16 + extra_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, spike_history: torch.Tensor, extra_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = self.encoder(spike_history)
        if extra_features is not None:
            features = torch.cat([features, extra_features], dim=-1)
        value = self.mlp(features).squeeze(-1)
        return value
