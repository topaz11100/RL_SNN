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

        # Iterate over time steps (T_semi)
        for t in range(spikes.shape[0]):
            pre = spikes[t]
            post = spikes[t] # Simplified: assuming auto-association or layer input-output for demo
            
            self.buffer.push(float(pre.mean().item()), float(post.mean().item()))
            
            # Identify events
            events = []
            if pre.sum() > 0: events.append(torch.tensor([[1.0, 0.0]]))
            if post.sum() > 0: events.append(torch.tensor([[0.0, 1.0]]))
            
            for event_type in events:
                spike_history, scalars, actor_state = self.build_state(self.buffer, weight, event_type)

                # Actor-Critic Interaction
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

        # Theory 5.4: Reward Calculation
        firing_rates = spikes.float().mean(dim=0)
        predicted = int(torch.argmax(firing_rates).item())
        
        # Calculate Margin
        if len(firing_rates) > 1:
            other_rates = torch.cat([firing_rates[:label], firing_rates[label + 1 :]])
            max_wrong = torch.max(other_rates)
            margin = float((firing_rates[label] - max_wrong).item())
        else:
            margin = 0.0 # Single neuron case
            
        reward = reward_classification(predicted == label, margin, self.beta_margin)
        
        self.optimize_from_trajectory(trajectory, reward)
        return reward.detach()
