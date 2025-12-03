from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from snn.lif import LIFParams, lif_step


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

    def forward(self, input_spikes: Tensor) -> Tuple[Tensor, Tensor]:
        """Simulate the E/I network for given input spike trains.

        Args:
            input_spikes: Tensor of shape (batch, 784, T)

        Returns:
            exc_spikes: Tensor of shape (batch, n_exc, T)
            inh_spikes: Tensor of shape (batch, n_inh, T)
        """
        if input_spikes.dim() != 3 or input_spikes.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must have shape (batch, {self.n_input}, T)")
        batch_size, _, T = input_spikes.shape
        device = input_spikes.device
        dtype = input_spikes.dtype

        exc_spikes = torch.zeros((batch_size, self.n_exc, T), device=device, dtype=dtype)
        inh_spikes = torch.zeros((batch_size, self.n_inh, T), device=device, dtype=dtype)

        v_exc = torch.full((batch_size, self.n_exc), self.exc_params.v_rest, device=device, dtype=dtype)
        v_inh = torch.full((batch_size, self.n_inh), self.inh_params.v_rest, device=device, dtype=dtype)

        s_exc_prev = torch.zeros_like(v_exc)
        s_inh_prev = torch.zeros_like(v_inh)

        for t in range(T):
            x_t = input_spikes[:, :, t]
            w_inh_exc = torch.relu(self.w_inh_exc) * self.inh_exc_mask
            I_exc = torch.matmul(x_t, torch.relu(self.w_input_exc)) - torch.matmul(s_inh_prev, w_inh_exc)
            v_exc, s_exc = lif_step(v_exc, I_exc, self.exc_params)

            I_inh = self.weight_ei * s_exc_prev
            v_inh, s_inh = lif_step(v_inh, I_inh, self.inh_params)

            exc_spikes[:, :, t] = s_exc
            inh_spikes[:, :, t] = s_inh

            s_exc_prev = s_exc
            s_inh_prev = s_inh

        return exc_spikes, inh_spikes
