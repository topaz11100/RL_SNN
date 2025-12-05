from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from snn.lif import LIFCell, LIFParams


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
        self.surrogate_slope = 5.0

        layer_sizes = [n_input] + self.hidden_sizes + [n_output]
        weights = [
            nn.Parameter(torch.rand(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
            for i in range(len(layer_sizes) - 1)
        ]
        self.w_layers = nn.ParameterList(weights)

        # Surrogate LIF dynamics shared across layers (Theory 2.2)
        self.hidden_cell = LIFCell(self.hidden_params, surrogate=True, slope=self.surrogate_slope)
        self.output_cell = LIFCell(self.output_params, surrogate=True, slope=self.surrogate_slope)

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

        v_states: Tuple[Tensor, ...] = tuple(
            torch.full((batch_size, h), self.hidden_params.v_rest, device=device, dtype=dtype)
            for h in self.hidden_sizes
        )
        s_prev: Tuple[Tensor, ...] = tuple(torch.zeros_like(v) for v in v_states)
        v_output = torch.full((batch_size, self.n_output), self.output_params.v_rest, device=device, dtype=dtype)

        if T == 0:
            empty_hidden = [torch.empty((batch_size, h, 0), device=device, dtype=dtype) for h in self.hidden_sizes]
            empty_output = torch.empty((batch_size, self.n_output, 0), device=device, dtype=dtype)
            firing_rates = torch.zeros((batch_size, self.n_output), device=device, dtype=dtype)
            return empty_hidden, empty_output, firing_rates

        hidden_spikes_tensor: Tuple[Tensor, ...] = tuple(
            torch.empty((batch_size, h, T), device=device, dtype=dtype) for h in self.hidden_sizes
        )
        output_spikes = torch.empty((batch_size, self.n_output, T), device=device, dtype=dtype)

        if n_hidden_layers == 0:
            for t in range(T):
                x_t = input_spikes[:, :, t]
                current_out = torch.matmul(x_t, torch.relu(self.w_layers[0]))
                v_output, s_output = self.output_cell(v_output, current_out)
                output_spikes[:, :, t] = s_output

            firing_rates = output_spikes.mean(dim=2)
            return [], output_spikes, firing_rates

        for t in range(T):
            x_t = input_spikes[:, :, t]

            current_first = torch.matmul(x_t, torch.relu(self.w_layers[0]))
            v_first_next, s_first = self.hidden_cell(v_states[0], current_first)

            new_v_states = (v_first_next,)
            new_spikes = (s_first,)

            for li in range(1, n_hidden_layers):
                current_hidden = torch.matmul(s_prev[li - 1], torch.relu(self.w_layers[li]))
                v_next, s_next = self.hidden_cell(v_states[li], current_hidden)
                new_v_states = new_v_states + (v_next,)
                new_spikes = new_spikes + (s_next,)

            prev_spikes_for_output = s_prev[-1]
            current_out = torch.matmul(prev_spikes_for_output, torch.relu(self.w_layers[-1]))
            v_output, s_output = self.output_cell(v_output, current_out)

            for li, spike in enumerate(new_spikes):
                hidden_spikes_tensor[li][:, :, t] = spike
            output_spikes[:, :, t] = s_output

            v_states = new_v_states
            s_prev = new_spikes

        hidden_spikes = [tensor for tensor in hidden_spikes_tensor]
        firing_rates = output_spikes.mean(dim=2)
        return hidden_spikes, output_spikes, firing_rates

    @property
    def synapse_shapes(self) -> List[Tuple[int, int]]:
        return [(w.shape[0], w.shape[1]) for w in self.w_layers]
