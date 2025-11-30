import torch
import torch.nn as nn


class SpikeHistoryCNN(nn.Module):
    """1D CNN front-end that embeds pre/post spike histories into a fixed feature vector.

    The module follows the design specified in Theory.md Section 2.4: two Conv1d
    layers with ReLU activations followed by global average pooling over the
    temporal dimension. Input tensors are expected in the shape
    ``(batch, channels=2, length=L)`` where the two channels correspond to the
    pre- and post-synaptic spike traces.
    """

    def __init__(self) -> None:
        """Initialize the SpikeHistoryCNN with two convolutional blocks."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
        )

    def forward(self, spike_history: torch.Tensor) -> torch.Tensor:
        """Embed spike history into a 16-D feature vector.

        Args:
            spike_history: Tensor of shape ``(batch, 2, L)`` containing binary
                spike traces for the pre- and post-synaptic neurons.

        Returns:
            Tensor of shape ``(batch, 16)`` representing the pooled temporal
            features for each synapse event.
        """
        x = self.features(spike_history)
        pooled = x.mean(dim=-1)
        return pooled
