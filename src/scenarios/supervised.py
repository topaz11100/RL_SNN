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
        alpha_align: float = 1.0,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        lif_tau_m: float = 20.0,
        lif_v_threshold: float = 1.0,
        lif_v_reset: float = 0.0,
        dt: float = 1.0,
        log_gradient_stats: bool = False,
        run_name: str = "default",
        device: str = "cpu",
    ) -> None:
        """Enable layer position input to match supervised state definition."""
        super().__init__(
            history_length,
            sigma_policy,
            include_layer_pos=True,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            run_name=run_name,
            device=device,
        )
        self.buffer = RollingSpikeBuffer(history_length)
        self.alpha_align = alpha_align
        self.log_gradient_stats = log_gradient_stats
        self.lif_params = LIFParameters(
            tau_m=lif_tau_m,
            v_threshold=lif_v_threshold,
            v_reset=lif_v_reset,
            dt=dt,
        )

    def run_episode(self, episode_data: Dict) -> torch.Tensor:
        """Compare agent updates against teacher deltas using a differentiable LIF path."""
        image: torch.Tensor = episode_data["image"].to(self.device).view(-1)
        steps: int = int(episode_data.get("steps", self.history_length))
        layer_pos: float = float(episode_data.get("layer_pos", 0.0))
        alpha_align = float(episode_data.get("alpha_align", self.alpha_align))

        # Teacher forward pass: Poisson input -> LIF with surrogate spikes -> loss
        input_spikes = poisson_encode(image, steps).to(self.device)
        lif = LIFNeuron(self.lif_params, soft_reset=True)
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
        teacher_delta: torch.Tensor = -alpha_align * teacher_weight.grad.detach()

        # Agent policy loop now uses the differentiable spikes produced above
        clip_min, clip_max = episode_data.get("clip", (-1.0, 1.0))
        weight = torch.full_like(teacher_weight, float(episode_data.get("weight", 0.0)))
        trajectory: List[TrajectoryEntry] = []
        total_agent_delta = torch.zeros_like(weight)

        lif_agent = LIFNeuron(self.lif_params, soft_reset=True)
        agent_state = lif_agent.initial_state((1,), device=self.device)

        for t in range(steps):
            pre = input_spikes[t]
            synaptic_current = torch.dot(pre, weight).view_as(agent_state.voltage)
            agent_state = lif_agent.step(agent_state, synaptic_current)
            post = agent_state.spike
            self.buffer.push(float(pre.mean().item()), float(post.mean().item()))

            events = []
            if pre.sum() > 0:
                events.append(torch.tensor([[1.0, 0.0]]))
            if post.sum() > 0:
                events.append(torch.tensor([[0.0, 1.0]]))

            for event_type in events:
                spike_history, scalars, actor_state = self.build_state(
                    self.buffer, float(weight.mean().item()), event_type, layer_pos=layer_pos
                )

                policy_out = self.actor.sample_action(actor_state)
                value_est = self.critic(spike_history, scalars)
                action_delta = torch.full_like(weight, float(policy_out.action.item()))

                trajectory.append(
                    TrajectoryEntry(
                        spike_history=spike_history,
                        scalars=scalars,
                        log_prob=policy_out.log_prob,
                        value=value_est,
                    )
                )

                weight = weight + action_delta
                weight = torch.clamp(weight, min=clip_min, max=clip_max)
                total_agent_delta += action_delta

        # Theory 6.5: Reward based on difference between Agent's total update and Teacher's update
        agent_delta_tensor = total_agent_delta.to(self.device)
        teacher_delta_device = teacher_delta.to(self.device)
        reward = reward_mimicry(agent_delta_tensor, teacher_delta_device)

        if self.log_gradient_stats:
            with torch.no_grad():
                print(
                    f"[GradientStats:{self.run_name}] teacher_delta mean={teacher_delta_device.mean().item():.4f} "
                    f"std={teacher_delta_device.std(unbiased=False).item():.4f} "
                    f"agent_delta mean={agent_delta_tensor.mean().item():.4f} std={agent_delta_tensor.std(unbiased=False).item():.4f}"
                )

        self.optimize_from_trajectory(trajectory, reward)
        return reward.detach()
