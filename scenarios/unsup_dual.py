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
    firing_rates = exc_spikes.mean(dim=2)
    r_sparse = -(firing_rates.mean(dim=1) - rho_target).abs()
    r_div = firing_rates.std(dim=1)
    temporal_diff = exc_spikes[:, :, 1:] - exc_spikes[:, :, :-1]
    r_stab = -temporal_diff.abs().mean(dim=(1, 2))
    total = alpha_sparse * r_sparse + alpha_div * r_div + alpha_stab * r_stab
    return r_sparse, r_div, r_stab, total


def _prepare_state(pre_spikes: torch.Tensor, post_spikes: torch.Tensor, L: int) -> torch.Tensor:
    pre_hist = pre_spikes.mean(dim=1)
    post_hist = post_spikes.mean(dim=1)
    T = pre_hist.shape[1]
    if L > T:
        L = T
    pre_seg = pre_hist[:, -L:]
    post_seg = post_hist[:, -L:]
    return torch.stack([pre_seg, post_seg], dim=1)


def run_unsup2(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = get_mnist_dataloaders(args.batch_size_images, args.seed)

    network = DiehlCookNetwork().to(device)

    actor_exc = GaussianPolicy(sigma=args.sigma_unsup2).to(device)
    critic_exc = ValueFunction().to(device)
    optimizer_actor_exc = torch.optim.Adam(actor_exc.parameters(), lr=args.lr_actor)
    optimizer_critic_exc = torch.optim.Adam(critic_exc.parameters(), lr=args.lr_critic)

    actor_inh = GaussianPolicy(sigma=args.sigma_unsup2).to(device)
    critic_inh = ValueFunction().to(device)
    optimizer_actor_inh = torch.optim.Adam(actor_inh.parameters(), lr=args.lr_actor)
    optimizer_critic_inh = torch.optim.Adam(critic_inh.parameters(), lr=args.lr_critic)

    metrics_path = os.path.join(args.result_dir, "metrics_train.txt")
    _ensure_metrics_file(metrics_path)

    for epoch in range(1, args.num_epochs + 1):
        epoch_sparse, epoch_div, epoch_stab, epoch_total = [], [], [], []
        for images, _ in train_loader:
            images = images.to(device)
            input_spikes = poisson_encode(images, args.T_unsup2, max_rate=args.max_rate).to(device)
            exc_spikes, inh_spikes = network(input_spikes)

            state_exc = _prepare_state(input_spikes, exc_spikes, args.spike_array_len)
            state_inh = _prepare_state(inh_spikes, exc_spikes, args.spike_array_len)

            action_exc, logp_exc, _ = actor_exc(state_exc)
            value_exc = critic_exc(state_exc).squeeze(-1)

            action_inh, logp_inh, _ = actor_inh(state_inh)
            value_inh = critic_inh(state_inh).squeeze(-1)

            r_sparse, r_div, r_stab, r_total = _compute_rewards(
                exc_spikes, args.rho_target, args.alpha_sparse, args.alpha_div, args.alpha_stab
            )

            with torch.no_grad():
                network.w_input_exc.data = network.w_input_exc.data + 0.01 * action_exc.mean()
                network.w_inh_exc.data = network.w_inh_exc.data + 0.01 * action_inh.mean()

            for i in range(state_exc.size(0)):
                buffer_exc = EpisodeBuffer()
                buffer_exc.append(state_exc[i], action_exc[i], logp_exc[i], value_exc[i])
                buffer_exc.finalize(r_total[i])
                ppo_update(
                    actor_exc,
                    critic_exc,
                    buffer_exc,
                    optimizer_actor_exc,
                    optimizer_critic_exc,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, len(buffer_exc)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

                buffer_inh = EpisodeBuffer()
                buffer_inh.append(state_inh[i], action_inh[i], logp_inh[i], value_inh[i])
                buffer_inh.finalize(r_total[i])
                ppo_update(
                    actor_inh,
                    critic_inh,
                    buffer_inh,
                    optimizer_actor_inh,
                    optimizer_critic_inh,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, len(buffer_inh)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

            epoch_sparse.append(r_sparse.mean().item())
            epoch_div.append(r_div.mean().item())
            epoch_stab.append(r_stab.mean().item())
            epoch_total.append(r_total.mean().item())

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
