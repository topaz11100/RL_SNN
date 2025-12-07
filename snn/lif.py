from typing import Tuple

import torch
from torch import Tensor, nn


@torch.jit.script
class LIFParams(object):
    def __init__(
        self,
        tau: float = 20.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        dt: float = 1.0,
        R: float = 1.0,
    ):
        self.tau = tau
        self.v_th = v_th
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.dt = dt
        self.R = R


# ==========================================
# Part A: JIT-Compatible LIF (For Unsup/Semi)
# ==========================================

def lif_dynamics(
    v: Tensor,
    I: Tensor,
    params: LIFParams,
) -> Tuple[Tensor, Tensor]:
    """Hard spike LIF dynamics for JIT execution."""
    if v.shape != I.shape:
        raise ValueError(f"v and I must have the same shape, got {v.shape} and {I.shape}")

    dt_over_tau = params.dt / params.tau
    dv = (-(v - params.v_rest) + params.R * I) * dt_over_tau
    v_next = v + dv

    spikes = (v_next >= params.v_th).to(v_next.dtype)

    v_reset = torch.full_like(v_next, params.v_reset)
    v_after_reset = torch.where(spikes.to(torch.bool), v_reset, v_next)

    return v_after_reset, spikes


lif_dynamics_script = torch.jit.script(lif_dynamics)


def lif_step(v: Tensor, I_syn: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]:
    """Single LIF update step (Theory 2.2 hard Heaviside)."""
    return lif_dynamics(v, I_syn, params)


lif_step_script = torch.jit.script(lif_step)


def lif_forward(I: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]:
    """Vectorized LIF simulation over time (Theory 2.2)."""
    if I.dim() == 2:
        I = I.unsqueeze(-1)
    if I.dim() != 3:
        raise ValueError("Input current must have shape (batch, neurons, T) or (batch, neurons)")

    batch_size, num_neurons, T = I.shape
    v = torch.full((batch_size, num_neurons), fill_value=params.v_rest, device=I.device, dtype=I.dtype)

    v_hist: Tuple[Tensor, ...] = ()
    s_hist: Tuple[Tensor, ...] = ()

    for t in range(T):
        v, spikes = lif_step(v, I[:, :, t], params)
        v_hist = v_hist + (v,)
        s_hist = s_hist + (spikes,)

    V = torch.stack(v_hist, dim=2)
    S = torch.stack(s_hist, dim=2)
    return V, S


class LIFCell(nn.Module):
    """JIT-compatible LIF Cell."""

    def __init__(self, params: LIFParams):
        super().__init__()
        self.params = params

    @torch.jit.export
    def forward(self, v: Tensor, I: Tensor) -> Tuple[Tensor, Tensor]:
        return lif_dynamics(v, I, self.params)


# ==========================================
# Part B: Autograd-Compatible LIF (For Supervised BPTT)
# ==========================================


def lif_dynamics_bptt(
    v: Tensor,
    I: Tensor,
    params: LIFParams,
    slope: float = 25.0,
) -> Tuple[Tensor, Tensor]:
    """LIF dynamics with surrogate gradient using PyTorch primitive ops (vmap compatible)."""

    # 1. Integrate voltage
    dt_over_tau = params.dt / params.tau
    dv = (-(v - params.v_rest) + params.R * I) * dt_over_tau
    v_next = v + dv

    # 2. Surrogate Spike (Detach Trick)
    surrogate = torch.sigmoid(slope * (v_next - params.v_th))
    spike_hard = (v_next >= params.v_th).float()
    spikes = (spike_hard - surrogate).detach() + surrogate

    # 3. Soft Reset
    v_reset = torch.full_like(v_next, params.v_reset)
    v_after_reset = torch.where(spikes.detach().to(torch.bool), v_reset, v_next)

    return v_after_reset, spikes
