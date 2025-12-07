import torch
from torch import Tensor


def poisson_encode(images: Tensor, T: int, max_rate: float = 1.0) -> Tensor:
    """Convert images to Poisson spike trains.

    Args:
        images: Tensor of shape (batch, 1, 28, 28) with values in [0,1] or [0,255].
        T: Number of simulation timesteps.
        max_rate: Maximum firing probability per timestep.

    Returns:
        Tensor of shape (batch, 784, T) with 0/1 spikes.
    """
    if images.dim() != 4 or images.shape[1] != 1:
        raise ValueError("Images must have shape (batch, 1, 28, 28)")
    imgs = images.float()
    if imgs.max() > 1.0:
        imgs = imgs / 255.0
    intensities = imgs.flatten(start_dim=1)
    probs = torch.clamp(intensities * max_rate, 0.0, 1.0)
    batch_size, num_pixels = probs.shape
    rand_vals = torch.rand((batch_size, num_pixels, T), device=probs.device, dtype=probs.dtype)
    spikes = (rand_vals < probs.unsqueeze(-1)).to(probs.dtype)
    return spikes
