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

    def __init__(self, input_dim_scalars: int) -> None:
        """Initialize the critic network with its own CNN front-end.

        Args:
            input_dim_scalars: Dimensionality of scalar metadata (weight, event type, layer pos).
        """
        super().__init__()
        self.feature_extractor = SpikeHistoryCNN()
        self.mlp = nn.Sequential(
            nn.Linear(16 + input_dim_scalars, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, spike_history: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Predict the value for a batch of spike histories and scalar metadata."""
        cnn_features = self.feature_extractor(spike_history)
        fused = torch.cat([cnn_features, scalars], dim=-1)
        return self.mlp(fused)


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
    spike_history: torch.Tensor,
    scalars: torch.Tensor,
    actor: ActorNetwork,
    critic: CriticNetwork,
    reward: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute losses for one batch of trajectory entries using separate CNNs.

    Args:
        spike_history: Local spike buffer history tensor.
        scalars: Scalar metadata tensor to fuse after CNN features.
        actor: Actor network producing Gaussian actions.
        critic: Critic network estimating returns.
        reward: Scalar reward for the episode broadcast to the batch.

    Returns:
        Tuple of actor loss and critic loss tensors.
    """
    actor_features = actor.feature_extractor(spike_history)
    fused_state = fuse_state(actor_features, scalars[:, :1], scalars[:, 1:3], scalars[:, 3:] if scalars.shape[1] > 3 else None)
    policy = actor.sample_action(fused_state)
    value = critic(spike_history, scalars)
    advantage = reward - value
    actor_loss = -(advantage.detach() * policy.log_prob).mean()
    critic_loss = F.mse_loss(value, reward.expand_as(value))
    return actor_loss, critic_loss
