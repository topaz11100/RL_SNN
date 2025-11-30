from collections import deque
from typing import Deque, List

import torch


class RollingSpikeBuffer:
    """Fixed-length buffer that stores binary spike traces for pre/post neurons."""

    def __init__(self, length: int) -> None:
        """Initialize the buffer with the desired history length."""
        self.length = length
        self.pre: Deque[float] = deque(maxlen=length)
        self.post: Deque[float] = deque(maxlen=length)
        self.reset()

    def reset(self) -> None:
        """Clear the buffer and fill with zeros."""
        self.pre.clear()
        self.post.clear()
        for _ in range(self.length):
            self.pre.append(0.0)
            self.post.append(0.0)

    def push(self, pre_spike: float, post_spike: float) -> None:
        """Append a new pair of spikes to the rolling history."""
        self.pre.append(pre_spike)
        self.post.append(post_spike)

    def to_tensor(self) -> torch.Tensor:
        """Convert the buffer contents to a torch tensor of shape (2, L)."""
        pre_tensor = torch.tensor(list(self.pre), dtype=torch.float32)
        post_tensor = torch.tensor(list(self.post), dtype=torch.float32)
        return torch.stack([pre_tensor, post_tensor], dim=0)

    def batch(self, batch_size: int) -> torch.Tensor:
        """Tile the current buffer into a batch tensor for vectorized inference."""
        tensor = self.to_tensor().unsqueeze(0)
        return tensor.repeat(batch_size, 1, 1)
