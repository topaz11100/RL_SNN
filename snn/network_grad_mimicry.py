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
        weights = []
        for i in range(len(layer_sizes) - 1):
            weights.append(nn.Parameter(torch.rand(layer_sizes[i], layer_sizes[i + 1]) * 0.1))
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
        
        # [Fix for vmap] In-place 할당을 피하기 위해 리스트에 수집합니다.
        hidden_spikes_list = [[] for _ in self.hidden_sizes]
        output_spikes_list = []

        v_states = [torch.full((batch_size, h), self.hidden_params.v_rest, device=device, dtype=dtype) for h in self.hidden_sizes]
        v_output = torch.full((batch_size, self.n_output), self.output_params.v_rest, device=device, dtype=dtype)

        if n_hidden_layers == 0:
            for t in range(T):
                x_t = input_spikes[:, :, t]
                current_out = torch.matmul(x_t, torch.relu(self.w_layers[0]))
                v_output, s_output = self.output_cell(v_output, current_out)
                output_spikes_list.append(s_output)
            
            output_spikes = torch.stack(output_spikes_list, dim=2)
            firing_rates = output_spikes.mean(dim=2)
            return [], output_spikes, firing_rates

        s_prev = [torch.zeros_like(v) for v in v_states]
        s_hidden = None  # 초기화

        for t in range(T):
            x_t = input_spikes[:, :, t]
            
            # 첫 번째 은닉층
            current = torch.matmul(x_t, torch.relu(self.w_layers[0]))
            v_states[0], s_hidden_0 = self.hidden_cell(v_states[0], current)
            hidden_spikes_list[0].append(s_hidden_0)
            
            # 현재 스텝의 은닉층 출력을 저장할 임시 변수
            s_curr_layer_out = s_hidden_0

            # 나머지 은닉층
            for li in range(1, n_hidden_layers):
                # 이전 층의 '이전 스텝' 스파이크(s_prev)를 입력으로 사용
                current = torch.matmul(s_prev[li - 1], torch.relu(self.w_layers[li]))
                v_states[li], s_curr = self.hidden_cell(v_states[li], current)
                hidden_spikes_list[li].append(s_curr)
                s_curr_layer_out = s_curr

            # 출력층 (마지막 은닉층의 이전 스텝 스파이크 사용)
            prev_spikes_for_output = s_prev[-1]
            current_out = torch.matmul(prev_spikes_for_output, torch.relu(self.w_layers[-1]))
            v_output, s_output = self.output_cell(v_output, current_out)
            output_spikes_list.append(s_output)

            # s_prev 업데이트 (다음 타임스텝에서 사용하기 위해 현재 스파이크 저장)
            # 리스트의 요소를 교체하는 것은 Tensor In-place 연산이 아니므로 안전합니다.
            s_prev[0] = s_hidden_0
            for li in range(1, n_hidden_layers):
                # 주의: hidden_spikes_list[li][-1]은 방금 추가한 현재 스텝 스파이크
                s_prev[li] = hidden_spikes_list[li][-1]

        # 리스트를 텐서로 변환 (Time 축 = dim 2)
        hidden_spikes = [torch.stack(s_list, dim=2) for s_list in hidden_spikes_list]
        output_spikes = torch.stack(output_spikes_list, dim=2)

        firing_rates = output_spikes.mean(dim=2)
        return hidden_spikes, output_spikes, firing_rates

    @property
    def synapse_shapes(self) -> List[Tuple[int, int]]:
        return [(w.shape[0], w.shape[1]) for w in self.w_layers]
