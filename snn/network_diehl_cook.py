from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from snn.lif import LIFParams, lif_step_script


@torch.jit.script
def _diehl_cook_forward_script(
    input_spikes: Tensor,
    w_input_exc: Tensor,
    w_inh_exc: Tensor,
    exc_params: LIFParams,
    inh_params: LIFParams,
    weight_ei: float,
    inh_exc_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    batch_size, _, T = input_spikes.shape
    dtype = input_spikes.dtype

    v_exc = torch.full((batch_size, w_input_exc.size(1)), exc_params.v_rest, device=input_spikes.device, dtype=dtype)
    v_inh = torch.full((batch_size, w_inh_exc.size(0)), inh_params.v_rest, device=input_spikes.device, dtype=dtype)

    s_exc_prev = torch.zeros_like(v_exc)
    s_inh_prev = torch.zeros_like(v_inh)

    exc_hist: Tuple[Tensor, ...] = ()
    inh_hist: Tuple[Tensor, ...] = ()

    w_inh_exc_masked = torch.relu(w_inh_exc) * inh_exc_mask

    I_exc_all = torch.matmul(input_spikes.permute(0, 2, 1), torch.relu(w_input_exc))

    for t in range(T):
        I_exc = I_exc_all[:, t, :] - torch.matmul(s_inh_prev, w_inh_exc_masked)
        v_exc, s_exc = lif_step_script(v_exc, I_exc, exc_params)

        I_inh = weight_ei * s_exc_prev
        v_inh, s_inh = lif_step_script(v_inh, I_inh, inh_params)

        exc_hist = exc_hist + (s_exc,)
        inh_hist = inh_hist + (s_inh,)

        s_exc_prev = s_exc
        s_inh_prev = s_inh

    exc_spikes = torch.stack(exc_hist, dim=2)
    inh_spikes = torch.stack(inh_hist, dim=2)

    return exc_spikes, inh_spikes


class DiehlCookNetwork(nn.Module):
    def __init__(
        self,
        n_input: int = 784,
        n_exc: int = 100,
        n_inh: int = 100,
        weight_ei: float = 50.0,
        exc_params: Optional[LIFParams] = None,
        inh_params: Optional[LIFParams] = None,
    ):
        super().__init__()
        assert n_exc == n_inh, "Diehl–Cook network assumes n_exc == n_inh for 1:1 E→I mapping"
        self.n_input = n_input
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.weight_ei = weight_ei
        self.exc_params = exc_params if exc_params is not None else LIFParams()
        self.inh_params = inh_params if inh_params is not None else LIFParams()

        self.w_input_exc = nn.Parameter(torch.rand(n_input, n_exc) * 0.1)
        self.w_inh_exc = nn.Parameter(torch.rand(n_inh, n_exc) * 0.1)
        inh_exc_mask = torch.ones(n_inh, n_exc)
        inh_exc_mask = inh_exc_mask - torch.eye(n_inh)
        self.register_buffer("inh_exc_mask", inh_exc_mask)
        with torch.no_grad():
            self.w_inh_exc.mul_(self.inh_exc_mask)

    @torch.jit.export
    def forward(self, input_spikes: Tensor) -> Tuple[Tensor, Tensor]:
        """Simulate the E/I network for given input spike trains."""
        if input_spikes.dim() != 3 or input_spikes.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must have shape (batch, {self.n_input}, T)")

        # Optimized: use scripted functional core for fused time-loop on GPU/CPU.
        return _diehl_cook_forward_script(
            input_spikes,
            self.w_input_exc,
            self.w_inh_exc,
            self.exc_params,
            self.inh_params,
            self.weight_ei,
            self.inh_exc_mask,
        )
