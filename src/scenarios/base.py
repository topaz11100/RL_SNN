from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import optim

from src.models.actor_critic import ActorNetwork, CriticNetwork, actor_critic_step, fuse_state
from src.models.cnn_frontend import SpikeHistoryCNN
from src.utils.spike_buffer import RollingSpikeBuffer


@dataclass
class TrajectoryEntry:
    """Container storing information for one synapse event during an episode."""

    fused_state: torch.Tensor


class RLScenario:
    """Abstract base scenario implementing the episode loop scaffolding."""

    def __init__(
        self,
        history_length: int,
        sigma_policy: float,
        include_layer_pos: bool = False,
        device: str = "cpu",
    ) -> None:
        """Prepare shared components for all scenarios."""
        self.history_length = history_length
        self.include_layer_pos = include_layer_pos
        self.device = device
        fused_dim = 16 + 1 + 2 + (1 if include_layer_pos else 0)
        self.actor = ActorNetwork(input_dim=fused_dim, sigma=sigma_policy).to(device)
        self.critic = CriticNetwork(input_dim=fused_dim).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

    def build_state(self, buffer: RollingSpikeBuffer, weight: float, event_type: torch.Tensor, layer_pos: Optional[float] = None) -> torch.Tensor:
        """Construct the fused local state tensor for a single event."""
        spike_tensor = buffer.batch(batch_size=1).to(self.device)
        cnn = SpikeHistoryCNN().to(self.device)
        features = cnn(spike_tensor).detach()
        weight_tensor = torch.tensor([[weight]], device=self.device)
        if layer_pos is not None and self.include_layer_pos:
            layer_tensor = torch.tensor([[layer_pos]], device=self.device)
        else:
            layer_tensor = None
        fused = fuse_state(features, weight_tensor, event_type.to(self.device), layer_tensor)
        return fused

    def run_episode(self, episode_data: Dict) -> torch.Tensor:
        """Run one episode and return the scalar reward."""
        raise NotImplementedError

    def optimize_from_trajectory(self, trajectory: List[TrajectoryEntry], reward: torch.Tensor) -> None:
        """Perform actor-critic updates from a stored trajectory."""
        all_states = torch.cat([entry.fused_state for entry in trajectory], dim=0)
        actor_loss, critic_loss = actor_critic_step(all_states, self.actor, self.critic, reward)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

    def clip_weight(self, weight: float, clip_min: float, clip_max: float) -> float:
        """Clip a scalar weight based on the synapse type bounds."""
        return float(max(min(weight, clip_max), clip_min))
