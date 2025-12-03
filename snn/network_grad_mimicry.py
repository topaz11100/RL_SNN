from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from snn.lif import LIFParams


def _surrogate_heaviside(x: Tensor, slope: float = 5.0) -> Tensor:
    return torch.sigmoid(slope * x)


class GradMimicryNetwork(nn.Module):
    """SNN for gradient mimicry with multiple hidden layers and surrogate gradients."""

    def __init__(
        self,
        n_input: int = 784,
        hidden_sizes: Optional[List[int]] = None,
        n_output: int = 10,
        hidden_params: Optional[LIFParams] = None,
        output_params: Optional[LIFParams] = None,
    ):
        super().__init__()
        self.n_input = n_input
        self.hidden_sizes = hidden_sizes or [256, 128, 64, 32]
        self.n_output = n_output
        self.hidden_params = hidden_params if hidden_params is not None else LIFParams()
        self.output_params = output_params if output_params is not None else LIFParams()

        layer_sizes = [n_input] + self.hidden_sizes + [n_output]
        weights = []
        for i in range(len(layer_sizes) - 1):
            weights.append(nn.Parameter(torch.rand(layer_sizes[i], layer_sizes[i + 1]) * 0.1))
        self.w_layers = nn.ParameterList(weights)

    @property
    def w_input_hidden(self) -> nn.Parameter:
        return self.w_layers[0]

    @property
    def w_hidden_output(self) -> nn.Parameter:
        return self.w_layers[-1]

    def forward(self, input_spikes: Tensor) -> Tuple[List[Tensor], Tensor, Tensor]:
        if input_spikes.dim() != 3 or input_spikes.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must have shape (batch, {self.n_input}, T)")
        batch_size, _, T = input_spikes.shape
        device = input_spikes.device
        dtype = input_spikes.dtype

        n_hidden_layers = len(self.hidden_sizes)
        hidden_spikes = [torch.zeros((batch_size, h, T), device=device, dtype=dtype) for h in self.hidden_sizes]
        output_spikes = torch.zeros((batch_size, self.n_output, T), device=device, dtype=dtype)

        v_states = [torch.full((batch_size, h), self.hidden_params.v_rest, device=device, dtype=dtype) for h in self.hidden_sizes]
        v_output = torch.full((batch_size, self.n_output), self.output_params.v_rest, device=device, dtype=dtype)

        s_prev = [torch.zeros_like(v_states[i]) for i in range(n_hidden_layers)]

        for t in range(T):
            x_t = input_spikes[:, :, t]
            current = torch.matmul(x_t, torch.relu(self.w_layers[0]))
            dv = (-(v_states[0] - self.hidden_params.v_rest) + self.hidden_params.R * current) * (
                self.hidden_params.dt / self.hidden_params.tau
            )
            v_states[0] = v_states[0] + dv
            s_hidden = _surrogate_heaviside(v_states[0] - self.hidden_params.v_th)
            s_reset = s_hidden.detach()
            v_states[0] = v_states[0] * (1.0 - s_reset) + self.hidden_params.v_reset * s_reset
            hidden_spikes[0][:, :, t] = s_hidden

            for li in range(1, n_hidden_layers):
                current = torch.matmul(s_prev[li - 1], torch.relu(self.w_layers[li]))
                dv = (-(v_states[li] - self.hidden_params.v_rest) + self.hidden_params.R * current) * (
                    self.hidden_params.dt / self.hidden_params.tau
                )
                v_states[li] = v_states[li] + dv
                s_curr = _surrogate_heaviside(v_states[li] - self.hidden_params.v_th)
                s_reset = s_curr.detach()
                v_states[li] = v_states[li] * (1.0 - s_reset) + self.hidden_params.v_reset * s_reset
                hidden_spikes[li][:, :, t] = s_curr
                s_hidden = s_curr

            current_out = torch.matmul(s_prev[-1], torch.relu(self.w_layers[-1])) if n_hidden_layers > 0 else torch.matmul(
                s_hidden, torch.relu(self.w_layers[-1])
            )
            dv_out = (-(v_output - self.output_params.v_rest) + self.output_params.R * current_out) * (
                self.output_params.dt / self.output_params.tau
            )
            v_output = v_output + dv_out
            s_output = _surrogate_heaviside(v_output - self.output_params.v_th)
            s_reset_out = s_output.detach()
            v_output = v_output * (1.0 - s_reset_out) + self.output_params.v_reset * s_reset_out
            output_spikes[:, :, t] = s_output

            s_prev[0] = hidden_spikes[0][:, :, t]
            for li in range(1, n_hidden_layers):
                s_prev[li] = hidden_spikes[li][:, :, t]

        firing_rates = output_spikes.mean(dim=2)
        return hidden_spikes, output_spikes, firing_rates

    @property
    def synapse_shapes(self) -> List[Tuple[int, int]]:
        return [(w.shape[0], w.shape[1]) for w in self.w_layers]
