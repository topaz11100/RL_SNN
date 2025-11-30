from __future__ import annotations

from typing import Dict, List

import torch

from src.scenarios.base import RLScenario, TrajectoryEntry
from src.utils.reward import reward_classification
from src.utils.spike_buffer import RollingSpikeBuffer


class SemiSupervisedScenario(RLScenario):
    """Scenario 2: single actor-critic trained with classification reward."""

    def __init__(
        self,
        history_length: int,
        sigma_policy: float,
        beta_margin: float,
        device: str = "cpu",
    ) -> None:
        """Initialize buffers and reward weights for the semi-supervised task."""
        super().__init__(history_length, sigma_policy, include_layer_pos=False, device=device)
        self.buffer = RollingSpikeBuffer(history_length)
        self.beta_margin = beta_margin

    def run_episode(self, episode_data: Dict) -> torch.Tensor:
        """Compute classification reward using firing rates and update networks."""
        spikes: torch.Tensor = episode_data["spikes"]
        label: int = int(episode_data["label"])
        weight: float = float(episode_data.get("weight", 0.0))
        clip_min, clip_max = episode_data.get("clip", (-1.0, 1.0))
        trajectory: List[TrajectoryEntry] = []

        for t in range(spikes.shape[0]):
            pre = spikes[t]
            post = spikes[t]
            self.buffer.push(float(pre.mean().item()), float(post.mean().item()))
            event_type = torch.tensor([[1.0, 0.0]]) if pre.sum() > 0 else torch.tensor([[0.0, 1.0]])
            fused_state = self.build_state(self.buffer, weight, event_type)
            trajectory.append(TrajectoryEntry(fused_state=fused_state))
            weight = self.clip_weight(weight, clip_min, clip_max)

        firing_rates = spikes.float().mean(dim=0)
        predicted = int(torch.argmax(firing_rates).item())
        margin = float((firing_rates[label] - torch.max(torch.cat([firing_rates[:label], firing_rates[label + 1 :]]))).item())
        reward = reward_classification(predicted == label, margin, self.beta_margin)
        self.optimize_from_trajectory(trajectory, reward)
        return reward.detach()
