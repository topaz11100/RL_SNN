from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from snn.lif import LIFParams, lif_dynamics_bptt


def _grad_mimicry_forward(
    input_spikes: Tensor,
    w_layers: List[Tensor],
    hidden_sizes: List[int],
    hidden_params: LIFParams,
    output_params: LIFParams,
    surrogate_slope: float,
) -> Tuple[List[Tensor], Tensor, Tensor]:
    batch_size, _, T = input_spikes.shape
    device = input_spikes.device
    dtype = input_spikes.dtype

    n_hidden_layers = len(hidden_sizes)

    v_states = [torch.full((batch_size, h), hidden_params.v_rest, device=device, dtype=dtype) for h in hidden_sizes]
    s_prev = [torch.zeros((batch_size, h), device=device, dtype=dtype) for h in hidden_sizes]
    v_output = torch.full((batch_size, w_layers[-1].size(1)), output_params.v_rest, device=device, dtype=dtype)

    if T == 0:
        empty_hidden = [torch.empty((batch_size, h, 0), device=device, dtype=dtype) for h in hidden_sizes]
        empty_output = torch.empty((batch_size, w_layers[-1].size(1), 0), device=device, dtype=dtype)
        firing_rates = torch.zeros((batch_size, w_layers[-1].size(1)), device=device, dtype=dtype)
        return empty_hidden, empty_output, firing_rates

    hidden_spikes_tensor = [torch.empty((batch_size, h, T), device=device, dtype=dtype) for h in hidden_sizes]
    output_spikes = torch.empty((batch_size, w_layers[-1].size(1), T), device=device, dtype=dtype)

    relu_w0 = torch.relu(w_layers[0])
    current_input_all = torch.matmul(input_spikes.permute(0, 2, 1), relu_w0)

    if n_hidden_layers == 0:
        for t in range(T):
            current_out = current_input_all[:, t, :]
            v_output, s_output = lif_dynamics_bptt(
                v_output, current_out, output_params, slope=surrogate_slope
            )
            output_spikes[:, :, t] = s_output

        firing_rates = output_spikes.mean(dim=2)
        return [], output_spikes, firing_rates

    for t in range(T):
        current_first = current_input_all[:, t, :]
        v_states[0], s_first = lif_dynamics_bptt(
            v_states[0], current_first, hidden_params, slope=surrogate_slope
        )

        s_current: List[Tensor] = []

        s_current.append(s_first)
        hidden_spikes_tensor[0][:, :, t] = s_first

        for li in range(1, n_hidden_layers):
            current_hidden = torch.matmul(s_prev[li - 1], torch.relu(w_layers[li]))
            v_states[li], s_next = lif_dynamics_bptt(
                v_states[li], current_hidden, hidden_params, slope=surrogate_slope
            )
            s_current.append(s_next)
            hidden_spikes_tensor[li][:, :, t] = s_next

        prev_spikes_for_output = s_prev[-1]
        current_out = torch.matmul(prev_spikes_for_output, torch.relu(w_layers[-1]))
        v_output, s_output = lif_dynamics_bptt(
            v_output, current_out, output_params, slope=surrogate_slope
        )

        output_spikes[:, :, t] = s_output

        s_prev = s_current

    hidden_spikes = hidden_spikes_tensor
    firing_rates = output_spikes.mean(dim=2)
    return hidden_spikes, output_spikes, firing_rates


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

    @property
    def w_input_hidden(self) -> nn.Parameter:
        return self.w_layers[0]

    @property
    def w_hidden_output(self) -> nn.Parameter:
        return self.w_layers[-1]

    def forward(self, input_spikes: Tensor) -> Tuple[List[Tensor], Tensor, Tensor]:
        if input_spikes.dim() != 3 or input_spikes.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must have shape (batch, {self.n_input}, T)")
        return _grad_mimicry_forward(
            input_spikes,
            list(self.w_layers),
            self.hidden_sizes,
            self.hidden_params,
            self.output_params,
            self.surrogate_slope,
        )

    @property
    def synapse_shapes(self) -> List[Tuple[int, int]]:
        return [(w.shape[0], w.shape[1]) for w in self.w_layers]
