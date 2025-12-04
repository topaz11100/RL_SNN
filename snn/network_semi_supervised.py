from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from snn.lif import LIFCell, LIFParams


class SemiSupervisedNetwork(nn.Module):
    def __init__(
        self,
        n_input: int = 784,
        n_hidden: int = 256,
        n_output: int = 10,
        hidden_params: Optional[LIFParams] = None,
        output_params: Optional[LIFParams] = None,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden_params = hidden_params if hidden_params is not None else LIFParams()
        self.output_params = output_params if output_params is not None else LIFParams()

        self.w_input_hidden = nn.Parameter(torch.rand(n_input, n_hidden) * 0.1)
        self.w_hidden_output = nn.Parameter(torch.rand(n_hidden, n_output) * 0.1)

        self.hidden_cell = LIFCell(self.hidden_params, surrogate=False)
        self.output_cell = LIFCell(self.output_params, surrogate=False)

    @torch.jit.export
    def forward(self, input_spikes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Simulate hidden and output LIF layers for encoded input.

        Args:
            input_spikes: Tensor of shape (batch, 784, T)

        Returns:
            hidden_spikes: Tensor of shape (batch, n_hidden, T)
            output_spikes: Tensor of shape (batch, n_output, T)
            firing_rates: Tensor of shape (batch, n_output) averaged over time
        """
        if input_spikes.dim() != 3 or input_spikes.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must have shape (batch, {self.n_input}, T)")
        batch_size, _, T = input_spikes.shape
        dtype = input_spikes.dtype

        v_hidden = torch.full((batch_size, self.n_hidden), self.hidden_params.v_rest, device=input_spikes.device, dtype=dtype)
        v_output = torch.full((batch_size, self.n_output), self.output_params.v_rest, device=input_spikes.device, dtype=dtype)

        s_hidden_prev = torch.zeros_like(v_hidden)
        s_output_prev = torch.zeros_like(v_output)

        hidden_hist: Tuple[Tensor, ...] = ()
        output_hist: Tuple[Tensor, ...] = ()

        for t in range(T):
            x_t = input_spikes[:, :, t]
            I_hidden = torch.matmul(x_t, torch.relu(self.w_input_hidden))
            v_hidden, s_hidden = self.hidden_cell(v_hidden, I_hidden)

            I_output = torch.matmul(s_hidden_prev, torch.relu(self.w_hidden_output))
            v_output, s_output = self.output_cell(v_output, I_output)

            hidden_hist = hidden_hist + (s_hidden,)
            output_hist = output_hist + (s_output,)

            s_hidden_prev = s_hidden
            s_output_prev = s_output

        hidden_spikes = torch.stack(hidden_hist, dim=2)
        output_spikes = torch.stack(output_hist, dim=2)

        firing_rates = output_spikes.mean(dim=2)
        return hidden_spikes, output_spikes, firing_rates
