from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from src.scenarios.base import RLScenario, TrajectoryEntry
from src.models.actor_critic import ActorNetwork, CriticNetwork, fuse_state
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
                spike_history, scalars, actor_state = self.build_state(self.buffer, weight, event_type)

                # Theory 2.6: Sample action Delta d
                policy_out = self.actor.sample_action(actor_state)
                action_delta = policy_out.action.item()

                # Theory 2.7: Critic Value (using critic's own CNN)
                value_est = self.critic(spike_history, scalars)

                # Store tuple (s, a_log_prob, V)
                trajectory.append(TrajectoryEntry(
                    spike_history=spike_history,
                    scalars=scalars,
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


class UnsupervisedDualPolicy(RLScenario):
    """Scenario 1.2: single loop that routes events to excitatory/inhibitory policies."""

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
        """Initialize shared critic and two actors for dual-policy control."""
        super().__init__(history_length, sigma_exc, include_layer_pos=False, device=device)
        self.buffer = RollingSpikeBuffer(history_length)
        self.tracker = WinnerTracker(num_exc_neurons)
        self.rho_target = rho_target
        self.alpha_sparse = alpha_sparse
        self.alpha_div = alpha_div
        self.alpha_stab = alpha_stab

        # Actor/Critic networks (Critic shared)
        self.actor_exc = ActorNetwork(input_dim=16 + 3, sigma=sigma_exc).to(device)
        self.actor_inh = ActorNetwork(input_dim=16 + 3, sigma=sigma_inh).to(device)
        self.critic = CriticNetwork(input_dim_scalars=3).to(device)

        self.actor_exc_opt = torch.optim.Adam(self.actor_exc.parameters(), lr=1e-3)
        self.actor_inh_opt = torch.optim.Adam(self.actor_inh.parameters(), lr=1e-3)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def _build_state(
        self, actor: ActorNetwork, weight: float, event_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spike_tensor = self.buffer.batch(batch_size=1).to(self.device)

        weight_tensor = torch.tensor([[weight]], device=self.device)
        scalars = torch.cat([weight_tensor, event_type.to(self.device)], dim=-1)

        actor_features = actor.feature_extractor(spike_tensor)
        fused = fuse_state(actor_features, weight_tensor, event_type.to(self.device), None)
        return spike_tensor, scalars, fused

    def _optimize(self, trajectories_exc: List[TrajectoryEntry], trajectories_inh: List[TrajectoryEntry], reward: torch.Tensor) -> None:
        if not trajectories_exc and not trajectories_inh:
            return

        all_entries = trajectories_exc + trajectories_inh
        values = torch.stack([entry.value for entry in all_entries]).squeeze()
        R_all = reward.to(self.device).expand_as(values)
        critic_loss = F.mse_loss(values, R_all)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        def _actor_step(entries: List[TrajectoryEntry], actor_opt: torch.optim.Optimizer) -> None:
            if not entries:
                return
            log_probs = torch.stack([entry.log_prob for entry in entries]).squeeze()
            local_values = torch.stack([entry.value for entry in entries]).squeeze()
            R_local = reward.to(self.device).expand_as(local_values)
            advantage = R_local - local_values
            actor_loss = -(advantage.detach() * log_probs).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

        _actor_step(trajectories_exc, self.actor_exc_opt)
        _actor_step(trajectories_inh, self.actor_inh_opt)

    def run_episode(self, episode_data: Dict) -> torch.Tensor:
        """Process both synapse types within one episode and update respective actors."""
        pre_exc: torch.Tensor = episode_data.get("pre_exc") or episode_data.get("pre")
        post_exc: torch.Tensor = episode_data.get("post_exc") or episode_data.get("post")
        pre_inh: torch.Tensor = episode_data.get("pre_inh", torch.zeros_like(pre_exc))
        post_inh: torch.Tensor = episode_data.get("post_inh", torch.zeros_like(post_exc))

        weight_exc: float = float(episode_data.get("weight_exc", episode_data.get("weight", 0.0)))
        weight_inh: float = float(episode_data.get("weight_inh", episode_data.get("weight", 0.0)))
        clip_exc = episode_data.get("clip_exc", episode_data.get("clip", (-1.0, 1.0)))
        clip_inh = episode_data.get("clip_inh", episode_data.get("clip", (-1.0, 1.0)))

        trajectory_exc: List[TrajectoryEntry] = []
        trajectory_inh: List[TrajectoryEntry] = []

        for t in range(pre_exc.shape[0]):
            # Combine excitatory and inhibitory history into a single buffer stream
            combined_pre = float(pre_exc[t].item()) - float(pre_inh[t].item())
            combined_post = float(post_exc[t].item()) - float(post_inh[t].item())
            self.buffer.push(combined_pre, combined_post)

            events = []
            if pre_exc[t] > 0:
                events.append(("exc", torch.tensor([[1.0, 0.0]])))
            if post_exc[t] > 0:
                events.append(("exc", torch.tensor([[0.0, 1.0]])))
            if pre_inh[t] > 0:
                events.append(("inh", torch.tensor([[1.0, 0.0]])))
            if post_inh[t] > 0:
                events.append(("inh", torch.tensor([[0.0, 1.0]])))

            for synapse_type, event_type in events:
                actor = self.actor_exc if synapse_type == "exc" else self.actor_inh
                spike_history, scalars, actor_state = self._build_state(actor, weight_exc if synapse_type == "exc" else weight_inh, event_type)

                policy_out = actor.sample_action(actor_state)
                value_est = self.critic(spike_history, scalars)
                action_delta = policy_out.action.item()

                entry = TrajectoryEntry(
                    spike_history=spike_history,
                    scalars=scalars,
                    log_prob=policy_out.log_prob,
                    value=value_est,
                )

                if synapse_type == "exc":
                    trajectory_exc.append(entry)
                    weight_exc += action_delta
                    weight_exc = float(max(min(weight_exc, clip_exc[1]), clip_exc[0]))
                else:
                    trajectory_inh.append(entry)
                    weight_inh += action_delta
                    weight_inh = float(max(min(weight_inh, clip_inh[1]), clip_inh[0]))

        rates = post_exc.float().mean()
        mean_rate = rates
        if post_exc.ndim > 1:
            spike_counts = post_exc.sum(dim=0)
            current_winner = int(spike_counts.argmax().item())
        else:
            current_winner = 0 if post_exc.sum() > 0 else -1

        sparse_r = reward_sparse(mean_rate, self.rho_target)
        div_r = reward_diversity(self.tracker.histogram())
        stab_r = reward_stability(current_winner, self.tracker)
        reward = self.alpha_sparse * sparse_r + self.alpha_div * div_r + self.alpha_stab * stab_r

        if current_winner >= 0:
            self.tracker.record(current_winner)

        self._optimize(trajectory_exc, trajectory_inh, reward)
        return reward.detach()
