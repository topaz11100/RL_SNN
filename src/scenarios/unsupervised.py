from __future__ import annotations

from typing import Dict, List

import torch

from src.scenarios.base import RLScenario, TrajectoryEntry
from src.utils.reward import WinnerTracker, reward_diversity, reward_sparse, reward_stability
from src.utils.spike_buffer import RollingSpikeBuffer


class UnsupervisedSinglePolicy(RLScenario):
    """Scenario 1.1: single actor-critic shared by all synapses."""

    def __init__(
        self,
        history_length: int,
        sigma_policy: float,
        rho_target: float,
        alpha_sparse: float,
        alpha_div: float,
        alpha_stab: float,
        num_exc_neurons: int,
        device: str = "cpu",
    ) -> None:
        """Initialize buffers and reward tracker."""
        super().__init__(history_length, sigma_policy, include_layer_pos=False, device=device)
        self.buffer = RollingSpikeBuffer(history_length)
        self.tracker = WinnerTracker(num_exc_neurons)
        self.rho_target = rho_target
        self.alpha_sparse = alpha_sparse
        self.alpha_div = alpha_div
        self.alpha_stab = alpha_stab

    def run_episode(self, episode_data: Dict) -> torch.Tensor:
        """Simulate an episode with actual Actor-Critic interaction."""
        pre_spikes: torch.Tensor = episode_data["pre"]
        post_spikes: torch.Tensor = episode_data["post"]
        weight: float = float(episode_data.get("weight", 0.0))
        clip_min, clip_max = episode_data.get("clip", (-1.0, 1.0))
        trajectory: List[TrajectoryEntry] = []

        # Theory 2.8: Loop through time steps
        for t in range(pre_spikes.shape[0]):
            self.buffer.push(float(pre_spikes[t].item()), float(post_spikes[t].item()))
            
            # Determine event type (simplified for this demo loop)
            # In real SNN, this happens per spike event. Here we assume non-zero pre is pre-event, etc.
            # To strictly follow event-driven, we should act only on spikes. 
            # Assuming 'pre_spikes' is binary.
            is_pre_event = pre_spikes[t] > 0
            is_post_event = post_spikes[t] > 0
            
            events = []
            if is_pre_event:
                events.append(torch.tensor([[1.0, 0.0]]))
            if is_post_event:
                events.append(torch.tensor([[0.0, 1.0]]))
            
            # Process events
            for event_type in events:
                fused_state = self.build_state(self.buffer, weight, event_type)
                
                # Theory 2.6: Sample action Delta d
                policy_out = self.actor.sample_action(fused_state)
                action_delta = policy_out.action.item()
                
                # Theory 2.7: Critic Value
                value_est = self.critic(fused_state)
                
                # Store tuple (s, a_log_prob, V)
                trajectory.append(TrajectoryEntry(
                    fused_state=fused_state,
                    log_prob=policy_out.log_prob,
                    value=value_est
                ))
                
                # Apply weight update (Theory 2.6: action * scale * lr)
                # Assuming simple additive update for this demo scope
                weight += action_delta
                weight = self.clip_weight(weight, clip_min, clip_max)

        # Theory 3.4: Compute Global Reward R
        rates = post_spikes.float().mean()
        mean_rate = rates
        # Find winner (simple argmax over sum for this demo)
        if post_spikes.ndim > 1:
            spike_counts = post_spikes.sum(dim=0)
            current_winner = int(spike_counts.argmax().item())
        else:
             # Scalar case (single neuron demo)
            current_winner = 0 if post_spikes.sum() > 0 else -1 # Placeholder

        sparse_r = reward_sparse(mean_rate, self.rho_target)
        div_r = reward_diversity(self.tracker.histogram())
        stab_r = reward_stability(current_winner, self.tracker)
        
        reward = self.alpha_sparse * sparse_r + self.alpha_div * div_r + self.alpha_stab * stab_r
        
        if current_winner >= 0:
            self.tracker.record(current_winner)
            
        # Optimize
        self.optimize_from_trajectory(trajectory, reward)
        
        return reward.detach()


class UnsupervisedDualPolicy:
    """Scenario 1.2: wrapper maintaining separate policies for exc/inh synapses."""

    def __init__(
        self,
        history_length: int,
        sigma_exc: float,
        sigma_inh: float,
        rho_target: float,
        alpha_sparse: float,
        alpha_div: float,
        alpha_stab: float,
        num_exc_neurons: int,
        device: str = "cpu",
    ) -> None:
        """Instantiate two independent single-policy controllers."""
        self.exc_policy = UnsupervisedSinglePolicy(
            history_length,
            sigma_exc,
            rho_target,
            alpha_sparse,
            alpha_div,
            alpha_stab,
            num_exc_neurons,
            device,
        )
        self.inh_policy = UnsupervisedSinglePolicy(
            history_length,
            sigma_inh,
            rho_target,
            alpha_sparse,
            alpha_div,
            alpha_stab,
            num_exc_neurons,
            device,
        )

    def run_episode(self, episode_data: Dict) -> torch.Tensor:
        """Dispatch episode to excitatory or inhibitory controller."""
        is_inhibitory: bool = episode_data.get("inhibitory", False)
        if is_inhibitory:
            return self.inh_policy.run_episode(episode_data)
        return self.exc_policy.run_episode(episode_data)
