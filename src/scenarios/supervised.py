from __future__ import annotations

from typing import Dict, List

import torch

from src.scenarios.base import RLScenario, TrajectoryEntry
from src.utils.reward import reward_mimicry
from src.utils.spike_buffer import RollingSpikeBuffer


class GradientMimicryScenario(RLScenario):
    """Scenario 3: fully supervised gradient mimicry controller."""

    def __init__(
        self,
        history_length: int,
        sigma_policy: float,
        device: str = "cpu",
    ) -> None:
        """Enable layer position input to match supervised state definition."""
        super().__init__(history_length, sigma_policy, include_layer_pos=True, device=device)
        self.buffer = RollingSpikeBuffer(history_length)

    def run_episode(self, episode_data: Dict) -> torch.Tensor:
        """Compare agent updates against teacher deltas and optimize."""
        spikes: torch.Tensor = episode_data["spikes"]
        layer_pos: float = float(episode_data.get("layer_pos", 0.0))
        agent_delta: torch.Tensor = episode_data["agent_delta"]
        teacher_delta: torch.Tensor = episode_data["teacher_delta"]
        weight: float = float(episode_data.get("weight", 0.0))
        clip_min, clip_max = episode_data.get("clip", (-1.0, 1.0))
        trajectory: List[TrajectoryEntry] = []

        for t in range(spikes.shape[0]):
            pre = spikes[t]
            post = spikes[t]
            self.buffer.push(float(pre.mean().item()), float(post.mean().item()))
            event_type = torch.tensor([[1.0, 0.0]]) if pre.sum() > 0 else torch.tensor([[0.0, 1.0]])
            fused_state = self.build_state(self.buffer, weight, event_type, layer_pos=layer_pos)
            trajectory.append(TrajectoryEntry(fused_state=fused_state))
            weight = self.clip_weight(weight, clip_min, clip_max)

        reward = reward_mimicry(agent_delta, teacher_delta)
        self.optimize_from_trajectory(trajectory, reward)
        return reward.detach()
