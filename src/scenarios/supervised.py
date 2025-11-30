from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

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
        # Note: In a real run, agent_delta is accumulated from the actor's outputs.
        # But for reward calculation, we might need the Total Delta over the episode or step-wise.
        # Theory 6.5: R_i = - (Delta w_agent - Delta w_teacher)^2
        # We accumulate agent actions to get total Delta w_agent.

        label: int = int(episode_data.get("label", 0))
        target_signal = torch.tensor([float(label) / 9.0], device=self.device)
        teacher_weight = torch.nn.Parameter(torch.tensor([episode_data.get("weight", 0.0)], device=self.device))
        prediction = (spikes.to(self.device).float().mean() * teacher_weight).unsqueeze(0)
        loss = F.mse_loss(prediction, target_signal)
        loss.backward()
        teacher_delta: torch.Tensor = teacher_weight.grad.detach()
        weight: float = float(episode_data.get("weight", 0.0))
        clip_min, clip_max = episode_data.get("clip", (-1.0, 1.0))
        trajectory: List[TrajectoryEntry] = []

        total_agent_delta = 0.0

        for t in range(spikes.shape[0]):
            pre = spikes[t]
            post = spikes[t]
            self.buffer.push(float(pre.mean().item()), float(post.mean().item()))
            
            events = []
            if pre.sum() > 0: events.append(torch.tensor([[1.0, 0.0]]))
            if post.sum() > 0: events.append(torch.tensor([[0.0, 1.0]]))
            
            for event_type in events:
                spike_history, scalars, actor_state = self.build_state(self.buffer, weight, event_type, layer_pos=layer_pos)

                policy_out = self.actor.sample_action(actor_state)
                value_est = self.critic(spike_history, scalars)
                action_delta = policy_out.action.item()

                trajectory.append(TrajectoryEntry(
                    spike_history=spike_history,
                    scalars=scalars,
                    log_prob=policy_out.log_prob,
                    value=value_est
                ))
                
                weight += action_delta
                weight = self.clip_weight(weight, clip_min, clip_max)
                total_agent_delta += action_delta

        # Theory 6.5: Reward based on difference between Agent's total update and Teacher's update
        agent_delta_tensor = torch.tensor([total_agent_delta], device=self.device)
        reward = reward_mimicry(agent_delta_tensor, teacher_delta.to(self.device))
        
        self.optimize_from_trajectory(trajectory, reward)
        return reward.detach()
