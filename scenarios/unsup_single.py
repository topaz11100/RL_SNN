import os
from typing import Tuple

import torch

from data.mnist import get_mnist_dataloaders
from rl.buffers import EpisodeBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.network_diehl_cook import DiehlCookNetwork


def _ensure_metrics_file(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("epoch\tR_sparse\tR_div\tR_stab\tR_total\n")


def _compute_rewards(exc_spikes: torch.Tensor, rho_target: float, alpha_sparse: float, alpha_div: float, alpha_stab: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # exc_spikes: (batch, n_exc, T)
    firing_rates = exc_spikes.mean(dim=2)  # (batch, n_exc)
    r_sparse = -(firing_rates.mean(dim=1) - rho_target).abs()
    r_div = firing_rates.std(dim=1)
    temporal_diff = exc_spikes[:, :, 1:] - exc_spikes[:, :, :-1]
    r_stab = -temporal_diff.abs().mean(dim=(1, 2))
    total = alpha_sparse * r_sparse + alpha_div * r_div + alpha_stab * r_stab
    return r_sparse, r_div, r_stab, total


def _prepare_state(input_spikes: torch.Tensor, exc_spikes: torch.Tensor, L: int) -> torch.Tensor:
    # input_spikes: (batch, 784, T), exc_spikes: (batch, n_exc, T)
    pre_hist = input_spikes.mean(dim=1)  # (batch, T)
    post_hist = exc_spikes.mean(dim=1)   # (batch, T)
    T = pre_hist.shape[1]
    if L > T:
        L = T
    pre_seg = pre_hist[:, -L:]
    post_seg = post_hist[:, -L:]
    state = torch.stack([pre_seg, post_seg], dim=1)
    return state


def run_unsup1(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = get_mnist_dataloaders(args.batch_size_images, args.seed)

    network = DiehlCookNetwork().to(device)
    actor = GaussianPolicy(sigma=args.sigma_unsup1).to(device)
    critic = ValueFunction().to(device)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    metrics_path = os.path.join(args.result_dir, "metrics_train.txt")
    _ensure_metrics_file(metrics_path)

    for epoch in range(1, args.num_epochs + 1):
        epoch_sparse, epoch_div, epoch_stab, epoch_total = [], [], [], []
        for images, _ in train_loader:
            images = images.to(device)
            input_spikes = poisson_encode(images, args.T_unsup1, max_rate=args.max_rate).to(device)
            exc_spikes, _ = network(input_spikes)

            state = _prepare_state(input_spikes, exc_spikes, args.spike_array_len)
            action, log_prob, _ = actor(state)
            value = critic(state).squeeze(-1)

            r_sparse, r_div, r_stab, r_total = _compute_rewards(
                exc_spikes, args.rho_target, args.alpha_sparse, args.alpha_div, args.alpha_stab
            )

            # Apply weight update scaled by mean action (placeholder local update)
            with torch.no_grad():
                delta = 0.01 * action.mean()
                network.w_input_exc.data = network.w_input_exc.data + delta

            # Treat each image as an episode
            for i in range(state.size(0)):
                buffer = EpisodeBuffer()
                buffer.append(state[i], action[i], log_prob[i], value[i])
                buffer.finalize(r_total[i])
                ppo_update(
                    actor,
                    critic,
                    buffer,
                    optimizer_actor,
                    optimizer_critic,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, len(buffer)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

            epoch_sparse.append(r_sparse.mean().item())
            epoch_div.append(r_div.mean().item())
            epoch_stab.append(r_stab.mean().item())
            epoch_total.append(r_total.mean().item())

            # Limit iteration per epoch for quick smoke tests
            break

        mean_sparse = sum(epoch_sparse) / len(epoch_sparse)
        mean_div = sum(epoch_div) / len(epoch_div)
        mean_stab = sum(epoch_stab) / len(epoch_stab)
        mean_total = sum(epoch_total) / len(epoch_total)

        with open(metrics_path, "a") as f:
            f.write(f"{epoch}\t{mean_sparse:.6f}\t{mean_div:.6f}\t{mean_stab:.6f}\t{mean_total:.6f}\n")

        logger.info(
            "Epoch %d | R_sparse %.4f | R_div %.4f | R_stab %.4f | R_total %.4f",
            epoch,
            mean_sparse,
            mean_div,
            mean_stab,
            mean_total,
        )
