from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from snn.lif import LIFParams


def _surrogate_heaviside(x: Tensor, slope: float = 5.0) -> Tensor:
    return torch.sigmoid(slope * x)


class GradMimicryNetwork(nn.Module):
    """SNN for gradient mimicry with surrogate gradients."""

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
        if input_spikes.dim() != 3 or input_spikes.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must have shape (batch, {self.n_input}, T)")
        batch_size, _, T = input_spikes.shape
        device = input_spikes.device
        dtype = input_spikes.dtype

        hidden_spikes = torch.zeros((batch_size, self.n_hidden, T), device=device, dtype=dtype)
        output_spikes = torch.zeros((batch_size, self.n_output, T), device=device, dtype=dtype)

        v_hidden = torch.full((batch_size, self.n_hidden), self.hidden_params.v_rest, device=device, dtype=dtype)
        v_output = torch.full((batch_size, self.n_output), self.output_params.v_rest, device=device, dtype=dtype)

        for t in range(T):
            x_t = input_spikes[:, :, t]
            I_hidden = torch.matmul(x_t, torch.relu(self.w_input_hidden))
            dv_hidden = (-(v_hidden - self.hidden_params.v_rest) + self.hidden_params.R * I_hidden) * (
                self.hidden_params.dt / self.hidden_params.tau
            )
            v_hidden = v_hidden + dv_hidden
            s_hidden = _surrogate_heaviside(v_hidden - self.hidden_params.v_th)

            I_output = torch.matmul(s_hidden, torch.relu(self.w_hidden_output))
            dv_out = (-(v_output - self.output_params.v_rest) + self.output_params.R * I_output) * (
                self.output_params.dt / self.output_params.tau
            )
            v_output = v_output + dv_out
            s_output = _surrogate_heaviside(v_output - self.output_params.v_th)

            hidden_spikes[:, :, t] = s_hidden
            output_spikes[:, :, t] = s_output

        firing_rates = output_spikes.mean(dim=2)
        return output_spikes, firing_rates
