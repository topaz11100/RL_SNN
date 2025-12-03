from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
from torch import nn


@dataclass
class LIFParams:
    tau: float = 20.0
    v_th: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    dt: float = 1.0
    R: float = 1.0


def lif_dynamics(
    v: Tensor,
    I: Tensor,
    params: LIFParams,
    *,
    surrogate: bool = False,
    slope: float = 5.0,
) -> Tuple[Tensor, Tensor]:
    """공통 LIF 동역학 (Theory 2.2).

    Args:
        v: Membrane potential. Shape: (batch, neurons)
        I: Synaptic input current. Shape: (batch, neurons)
        params: LIF parameters.
        surrogate: Whether to use surrogate gradient spikes.
        slope: Surrogate sigmoid slope when surrogate=True.

    Returns:
        Tuple of (next membrane potential, spikes).
    """
    if v.shape != I.shape:
        raise ValueError(f"v and I must have the same shape, got {v.shape} and {I.shape}")

    dt_over_tau = params.dt / params.tau

    dv = (-(v - params.v_rest) + params.R * I) * dt_over_tau
    v_next = v + dv

    if surrogate:
        spikes = torch.sigmoid(slope * (v_next - params.v_th))
    else:
        spikes = (v_next >= params.v_th).to(v_next.dtype)

    spikes_detached = spikes.detach()
    v_reset = torch.as_tensor(params.v_reset, device=v_next.device, dtype=v_next.dtype)
    v_next = v_next * (1.0 - spikes_detached) + v_reset * spikes_detached

    return v_next, spikes


def lif_step(v: Tensor, I_syn: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]:
    """Single LIF update step (Theory 2.2 hard Heaviside)."""
    return lif_dynamics(v, I_syn, params, surrogate=False)


def lif_forward(I: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]:
    """Vectorized LIF simulation over time (Theory 2.2).

    Args:
        I: Input current of shape (batch, neurons, T) or (batch, neurons).
        params: LIF parameters.

    Returns:
        V: Membrane potentials of shape (batch, neurons, T)
        S: Spike trains of shape (batch, neurons, T)
    """
    if I.dim() == 2:
        I = I.unsqueeze(-1)
    if I.dim() != 3:
        raise ValueError("Input current must have shape (batch, neurons, T) or (batch, neurons)")

    batch_size, num_neurons, T = I.shape
    device = I.device
    V = torch.zeros((batch_size, num_neurons, T), device=device, dtype=I.dtype)
    S = torch.zeros_like(V)

    v = torch.full((batch_size, num_neurons), fill_value=params.v_rest, device=device, dtype=I.dtype)
    for t in range(T):
        v, spikes = lif_step(v, I[:, :, t], params)
        V[:, :, t] = v
        S[:, :, t] = spikes
    return V, S


class LIFCell(nn.Module):
    """LIF 동역학을 수행하는 단일 스텝 셀 (Theory 2.2)."""

    def __init__(self, params: LIFParams, surrogate: bool = False, slope: float = 5.0):
        super().__init__()
        self.params = params
        self.surrogate = surrogate
        self.slope = slope

    def forward(self, v: Tensor, I: Tensor) -> Tuple[Tensor, Tensor]:
        return lif_dynamics(v, I, self.params, surrogate=self.surrogate, slope=self.slope)
