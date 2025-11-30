from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

action = torch.distributions

from .cnn_frontend import SpikeHistoryCNN


@dataclass
class PolicyOutput:
    """Container for actor outputs used in the RL buffer."""

    mean: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor


class ActorNetwork(nn.Module):
    """Gaussian policy network that maps local synapse state to weight updates."""

    def __init__(self, input_dim: int, sigma: float) -> None:
        """Initialize the actor network.

        Args:
            input_dim: Dimensionality of the fused local state vector.
            sigma: Fixed standard deviation for the Gaussian policy.
        """
        super().__init__()
        self.feature_extractor = SpikeHistoryCNN()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.sigma = sigma

    def forward(self, fused_state: torch.Tensor) -> torch.Tensor:
        """Compute the actor mean action for a batch of fused states."""
        return torch.tanh(self.mlp(fused_state))

    def sample_action(self, fused_state: torch.Tensor) -> PolicyOutput:
        """Sample a Gaussian action and return statistics for learning."""
        mean = self.forward(fused_state)
        dist = action.Normal(mean, self.sigma)
        sampled_action = dist.rsample()
        log_prob = dist.log_prob(sampled_action)
        return PolicyOutput(mean=mean, action=sampled_action, log_prob=log_prob)


class CriticNetwork(nn.Module):
    """Value function approximator that predicts expected return from local state."""

    def __init__(self, input_dim: int) -> None:
        """Initialize the critic network."""
        super().__init__()
        self.feature_extractor = SpikeHistoryCNN()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, fused_state: torch.Tensor) -> torch.Tensor:
        """Predict the value for a batch of fused states."""
        return self.mlp(fused_state)


def fuse_state(spike_features: torch.Tensor, weight: torch.Tensor, event_type: torch.Tensor, layer_pos: torch.Tensor | None = None) -> torch.Tensor:
    """Fuse CNN features and low-dimensional metadata into one vector.

    Args:
        spike_features: Tensor of shape ``(batch, 16)`` produced by the CNN front-end.
        weight: Tensor of shape ``(batch, 1)`` containing current synapse weights.
        event_type: Tensor of shape ``(batch, 2)`` with pre/post one-hot encoding.
        layer_pos: Optional tensor of shape ``(batch, 1)`` for normalized layer index
            used in the supervised gradient mimicry scenario.

    Returns:
        Concatenated tensor matching the Theory.md definitions for each scenario.
    """
    parts = [spike_features, weight, event_type]
    if layer_pos is not None:
        parts.append(layer_pos)
    return torch.cat(parts, dim=-1)


def actor_critic_step(
    fused_state: torch.Tensor,
    actor: ActorNetwork,
    critic: CriticNetwork,
    reward: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute losses for one batch of trajectory entries.

    Args:
        fused_state: Local state vector for each event.
        actor: Actor network producing Gaussian actions.
        critic: Critic network estimating returns.
        reward: Scalar reward for the episode broadcast to the batch.

    Returns:
        Tuple of actor loss and critic loss tensors.
    """
    policy = actor.sample_action(fused_state)
    value = critic(fused_state)
    advantage = reward - value
    actor_loss = -(advantage.detach() * policy.log_prob).mean()
    critic_loss = F.mse_loss(value, reward.expand_as(value))
    return actor_loss, critic_loss
