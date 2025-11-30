from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


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

    def initial_state(self, shape: Tuple[int, ...]) -> LIFState:
        """Create an initial state with zero voltage and spike tensors."""
        voltage = torch.zeros(shape)
        spike = torch.zeros(shape)
        return LIFState(voltage=voltage, spike=spike)
