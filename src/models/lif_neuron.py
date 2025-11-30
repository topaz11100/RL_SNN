from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import autograd
import math


@dataclass
class LIFParameters:
    """Container for LIF neuron parameters used during simulation."""

    tau_m: float
    v_threshold: float
    v_reset: float
    dt: float = 1.0


@dataclass
class LIFState:
    """State of a single LIF neuron including membrane voltage and spike flag."""

    voltage: torch.Tensor
    spike: torch.Tensor


class LIFNeuron:
    """Discrete-time LIF neuron supporting soft and hard resets."""

    def __init__(self, params: LIFParameters, soft_reset: bool = False) -> None:
        """Initialize the neuron with parameters and reset policy."""
        self.params = params
        self.soft_reset = soft_reset

    def _surrogate_spike(self, voltage: torch.Tensor, slope: float = 1.0) -> torch.Tensor:
        """Compute a surrogate spike using an atan-based gradient approximation.

        The forward behaves like a Heaviside step function while the backward
        pass returns a smooth derivative to keep gradients flowing through the
        thresholding operation.
        """
        return SurrogateHeaviside.apply(voltage - self.params.v_threshold, torch.tensor(slope, device=voltage.device))

    def step(self, state: LIFState, synaptic_current: torch.Tensor) -> LIFState:
        """Advance the neuron state by one time step.

        Args:
            state: Current neuron state containing voltage and spike flag.
            synaptic_current: Input current for the time step.

        Returns:
            Updated neuron state after integrating the current and applying the
            firing rule.
        """
        dv = (-(state.voltage) + synaptic_current) * (self.params.dt / self.params.tau_m)
        voltage = state.voltage + dv
        spike = (voltage >= self.params.v_threshold).float()
        if self.soft_reset:
            voltage = voltage - spike * self.params.v_threshold
        else:
            voltage = torch.where(spike.bool(), torch.full_like(voltage, self.params.v_reset), voltage)
        return LIFState(voltage=voltage, spike=spike)

    def step_surrogate(self, state: LIFState, synaptic_current: torch.Tensor, slope: float = 1.0) -> LIFState:
        """Advance the neuron state while using a surrogate gradient for spikes."""
        dv = (-(state.voltage) + synaptic_current) * (self.params.dt / self.params.tau_m)
        voltage = state.voltage + dv
        spike = self._surrogate_spike(voltage, slope)
        if self.soft_reset:
            voltage = voltage - spike * self.params.v_threshold
        else:
            voltage = torch.where(spike.bool(), torch.full_like(voltage, self.params.v_reset), voltage)
        return LIFState(voltage=voltage, spike=spike)

    def initial_state(self, shape: Tuple[int, ...], device: torch.device | None = None) -> LIFState:
        """Create an initial state with zero voltage and spike tensors."""
        voltage = torch.zeros(shape, device=device)
        spike = torch.zeros(shape, device=device)
        return LIFState(voltage=voltage, spike=spike)


class SurrogateHeaviside(autograd.Function):
    """Heaviside step function with atan surrogate gradient."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input, slope)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        input, slope = ctx.saved_tensors
        # atan surrogate derivative: d/dx step(x) ≈ 1 / (1 + (pi * slope * x)^2)
        surrogate_grad = slope / (1 + (math.pi * slope * input).pow(2))
        return grad_output * surrogate_grad, None
