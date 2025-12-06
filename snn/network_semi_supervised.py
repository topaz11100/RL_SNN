from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from snn.lif import LIFParams, lif_step_script


@torch.jit.script
def _semi_supervised_forward_script(
    input_spikes: Tensor,
    w_input_hidden: Tensor,
    w_hidden_output: Tensor,
    hidden_params: LIFParams,
    output_params: LIFParams,
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_size, _, T = input_spikes.shape
    dtype = input_spikes.dtype

    v_hidden = torch.full((batch_size, w_input_hidden.size(1)), hidden_params.v_rest, device=input_spikes.device, dtype=dtype)
    v_output = torch.full((batch_size, w_hidden_output.size(1)), output_params.v_rest, device=input_spikes.device, dtype=dtype)

    s_hidden_prev = torch.zeros_like(v_hidden)
    s_output_prev = torch.zeros_like(v_output)

    hidden_hist: Tuple[Tensor, ...] = ()
    output_hist: Tuple[Tensor, ...] = ()

    I_hidden_all = torch.matmul(input_spikes.permute(0, 2, 1), torch.relu(w_input_hidden))

    for t in range(T):
        I_hidden = I_hidden_all[:, t, :]
        v_hidden, s_hidden = lif_step_script(v_hidden, I_hidden, hidden_params)

        I_output = torch.matmul(s_hidden_prev, torch.relu(w_hidden_output))
        v_output, s_output = lif_step_script(v_output, I_output, output_params)

        hidden_hist = hidden_hist + (s_hidden,)
        output_hist = output_hist + (s_output,)

        s_hidden_prev = s_hidden
        s_output_prev = s_output

    hidden_spikes = torch.stack(hidden_hist, dim=2)
    output_spikes = torch.stack(output_hist, dim=2)

    firing_rates = output_spikes.mean(dim=2)
    return hidden_spikes, output_spikes, firing_rates


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

    @torch.jit.export
    def forward(self, input_spikes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Simulate hidden and output LIF layers for encoded input."""
        if input_spikes.dim() != 3 or input_spikes.shape[1] != self.n_input:
            raise ValueError(f"input_spikes must have shape (batch, {self.n_input}, T)")

        # Optimized: use scripted functional core for fused time-loop on GPU/CPU.
        return _semi_supervised_forward_script(
            input_spikes,
            self.w_input_hidden,
            self.w_hidden_output,
            self.hidden_params,
            self.output_params,
        )
