from typing import Tuple

import torch


def poisson_encode(image: torch.Tensor, steps: int) -> torch.Tensor:
    """Convert a normalized image into Poisson spike trains.

    Args:
        image: Tensor of shape ``(pixels,)`` or ``(H, W)`` with values in ``[0, 1]``.
        steps: Number of time steps ``T`` to simulate for the episode.

    Returns:
        Tensor of shape ``(steps, pixels)`` containing binary spikes.
    """
    flat = image.view(-1)
    rates = flat.clamp(0.0, 1.0)
    probs = rates.unsqueeze(0).repeat(steps, 1)
    return torch.bernoulli(probs)


def spike_delta_times(pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> torch.Tensor:
    """Compute time difference matrix for logging Δt-Δd scatter plots."""
    pre_times = (pre_spikes.nonzero(as_tuple=False)[:, 0]).float()
    post_times = (post_spikes.nonzero(as_tuple=False)[:, 0]).float()
    if pre_times.numel() == 0 or post_times.numel() == 0:
        return torch.tensor([])
    delta = pre_times.unsqueeze(1) - post_times.unsqueeze(0)
    return delta.reshape(-1)
