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


def lif_dynamics(
    v: Tensor,
    I: Tensor,
    params: LIFParams,
    surrogate: bool = False,
    slope: float = 5.0,
) -> Tuple[Tensor, Tensor]:
    """공통 LIF 동역학 (Theory 2.2)."""
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
    v_reset = torch.full_like(v_next, params.v_reset)
    # Hard reset to v_reset when a spike occurs (Theory.md §2.2 hard reset rule).
    v_after_reset = torch.where(spikes_detached.bool(), v_reset, v_next)

    return v_after_reset, spikes


# Optimized: Provide both non-JIT and scripted variants to avoid vmap/JIT conflicts while
# still enabling scripted kernels where safe.
lif_dynamics_script = torch.jit.script(lif_dynamics)


def lif_step(v: Tensor, I_syn: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]:
    """Single LIF update step (Theory 2.2 hard Heaviside)."""
    return lif_dynamics(v, I_syn, params, surrogate=False)


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
    """LIF 동역학을 수행하는 단일 스텝 셀 (Theory 2.2)."""

    def __init__(self, params: LIFParams, surrogate: bool = False, slope: float = 5.0):
        super().__init__()
        self.params = params
        self.surrogate = surrogate
        self.slope = slope

    @torch.jit.export
    def forward(self, v: Tensor, I: Tensor) -> Tuple[Tensor, Tensor]:
        return lif_dynamics(v, I, self.params, surrogate=self.surrogate, slope=self.slope)
