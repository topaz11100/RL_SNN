from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from src.scenarios.base import RLScenario, TrajectoryEntry
from src.models.lif_neuron import LIFNeuron, LIFParameters
from src.utils.reward import reward_mimicry
from src.utils.spike_buffer import RollingSpikeBuffer
from src.utils.poisson_encoding import poisson_encode


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
        """Compare agent updates against teacher deltas using a differentiable LIF path."""
        image: torch.Tensor = episode_data["image"].to(self.device).view(-1)
        steps: int = int(episode_data.get("steps", self.history_length))
        layer_pos: float = float(episode_data.get("layer_pos", 0.0))

        # Teacher forward pass: Poisson input -> LIF with surrogate spikes -> loss
        input_spikes = poisson_encode(image, steps).to(self.device)
        lif = LIFNeuron(
            LIFParameters(tau_m=20.0, v_threshold=1.0, v_reset=0.0, dt=1.0),
            soft_reset=True,
        )
        teacher_weight = torch.nn.Parameter(
            torch.full((image.numel(),), float(episode_data.get("weight", 0.0)), device=self.device)
        )
        state = lif.initial_state((1,), device=self.device)
        output_spikes: List[torch.Tensor] = []

        for t in range(steps):
            synaptic_current = torch.dot(input_spikes[t], teacher_weight).view_as(state.voltage)
            state = lif.step_surrogate(state, synaptic_current, slope=2.0)
            output_spikes.append(state.spike)

        output_tensor = torch.stack(output_spikes)  # (T, 1)
        target_signal = torch.tensor([float(int(episode_data.get("label", 0))) / 9.0], device=self.device)
        prediction = output_tensor.float().mean(dim=0)
        loss = F.mse_loss(prediction, target_signal)
        loss.backward()
        teacher_delta: torch.Tensor = teacher_weight.grad.detach().mean().unsqueeze(0)

        # Agent policy loop now uses the differentiable spikes produced above
        clip_min, clip_max = episode_data.get("clip", (-1.0, 1.0))
        weight: float = float(episode_data.get("weight", 0.0))
        trajectory: List[TrajectoryEntry] = []
        total_agent_delta = 0.0

        for t in range(steps):
            pre = input_spikes[t]
            post = output_tensor[t]
            self.buffer.push(float(pre.mean().item()), float(post.mean().item()))

            events = []
            if pre.sum() > 0:
                events.append(torch.tensor([[1.0, 0.0]]))
            if post.sum() > 0:
                events.append(torch.tensor([[0.0, 1.0]]))

            for event_type in events:
                spike_history, scalars, actor_state = self.build_state(
                    self.buffer, weight, event_type, layer_pos=layer_pos
                )

                policy_out = self.actor.sample_action(actor_state)
                value_est = self.critic(spike_history, scalars)
                action_delta = policy_out.action.item()

                trajectory.append(
                    TrajectoryEntry(
                        spike_history=spike_history,
                        scalars=scalars,
                        log_prob=policy_out.log_prob,
                        value=value_est,
                    )
                )

                weight += action_delta
                weight = self.clip_weight(weight, clip_min, clip_max)
                total_agent_delta += action_delta

        # Theory 6.5: Reward based on difference between Agent's total update and Teacher's update
        agent_delta_tensor = torch.tensor([total_agent_delta], device=self.device)
        reward = reward_mimicry(agent_delta_tensor, teacher_delta.to(self.device))

        self.optimize_from_trajectory(trajectory, reward)
        return reward.detach()
