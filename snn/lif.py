from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class LIFParams:
    tau: float = 20.0
    v_th: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    dt: float = 1.0
    R: float = 1.0


def lif_step(v: Tensor, I_syn: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]:
    """Single LIF update step.

    Args:
        v: Membrane potential at current step. Shape: (batch, neurons)
        I_syn: Synaptic current input. Shape: (batch, neurons)
        params: LIF parameters.

    Returns:
        Updated membrane potential and spike tensor (0/1) for this step.
    """
    dv = (-(v - params.v_rest) + params.R * I_syn) * (params.dt / params.tau)
    v_next = v + dv
    spikes = (v_next >= params.v_th).to(v_next.dtype)
    v_next = torch.where(spikes.bool(), torch.as_tensor(params.v_reset, device=v_next.device, dtype=v_next.dtype), v_next)
    return v_next, spikes


def lif_forward(I: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]:
    """Vectorized LIF simulation over time.

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
