import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple


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


class GaussianPolicy(nn.Module):
    """Gaussian Actor with CNN front-end and MLP head."""

    def __init__(self, sigma: float, extra_feature_dim: int = 0):
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
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    def forward(
        self,
        spike_history: torch.Tensor,
        extra_features: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            spike_history: Tensor of shape (batch, 2, L)
            extra_features: Optional tensor of shape (batch, F)
            actions: Optional actions to evaluate log-prob on.
        Returns:
            action: sampled or provided action Δd
            log_prob: log π(a|z)
            mean: mean action in [-1, 1]
        """

        spike_history = spike_history.float()
        features = self.encoder(spike_history)
        if extra_features is not None:
            features = torch.cat([features, extra_features], dim=-1)

        mean = torch.tanh(self.mlp(features)).squeeze(-1)
        std = self.sigma.expand_as(mean)
        dist = Normal(mean, std)

        if actions is None:
            action = dist.rsample()
        else:
            action = actions
        log_prob = dist.log_prob(action)
        return action, log_prob, mean
