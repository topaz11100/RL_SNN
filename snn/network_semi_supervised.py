from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from snn.lif import LIFParams, lif_step


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

    def forward(self, input_spikes: Tensor) -> Tuple[Tensor, Tensor]:
        """Simulate hidden and output LIF layers for encoded input.

        Args:
            input_spikes: Tensor of shape (batch, 784, T)

        Returns:
            output_spikes: Tensor of shape (batch, n_output, T)
            firing_rates: Tensor of shape (batch, n_output) averaged over time
        """
        if input_spikes.dim() != 3 or input_spikes.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must have shape (batch, {self.n_input}, T)")
        batch_size, _, T = input_spikes.shape
        device = input_spikes.device
        dtype = input_spikes.dtype

        hidden_spikes = torch.zeros((batch_size, self.n_hidden, T), device=device, dtype=dtype)
        output_spikes = torch.zeros((batch_size, self.n_output, T), device=device, dtype=dtype)

        v_hidden = torch.full((batch_size, self.n_hidden), self.hidden_params.v_rest, device=device, dtype=dtype)
        v_output = torch.full((batch_size, self.n_output), self.output_params.v_rest, device=device, dtype=dtype)

        s_hidden_prev = torch.zeros_like(v_hidden)
        s_output_prev = torch.zeros_like(v_output)

        for t in range(T):
            x_t = input_spikes[:, :, t]
            I_hidden = torch.matmul(x_t, torch.relu(self.w_input_hidden))
            v_hidden, s_hidden = lif_step(v_hidden, I_hidden, self.hidden_params)

            I_output = torch.matmul(s_hidden_prev, torch.relu(self.w_hidden_output))
            v_output, s_output = lif_step(v_output, I_output, self.output_params)

            hidden_spikes[:, :, t] = s_hidden
            output_spikes[:, :, t] = s_output

            s_hidden_prev = s_hidden
            s_output_prev = s_output

        firing_rates = output_spikes.mean(dim=2)
        return output_spikes, firing_rates
