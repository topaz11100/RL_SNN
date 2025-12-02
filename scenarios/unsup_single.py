import os
from typing import Tuple

import torch
import torch.nn.functional as F

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


def _compute_reward_components(exc_spikes: torch.Tensor, rho_target: float) -> Tuple[torch.Tensor, torch.Tensor]:
    firing_rates = exc_spikes.mean(dim=2)
    r_sparse = -(firing_rates.mean(dim=1) - rho_target).abs()
    r_div = firing_rates.std(dim=1)
    return r_sparse, r_div


def _gather_events(
    pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor, L: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pre_spikes.device
    batch_size, n_pre, T = pre_spikes.shape
    n_post = post_spikes.shape[1]
    pad_pre = F.pad(pre_spikes, (L - 1, 0))
    pad_post = F.pad(post_spikes, (L - 1, 0))
    idx_range = torch.arange(L, device=device)

    histories = []
    extras = []
    pre_indices = []
    post_indices = []

    def _append_events(spike_tensor: torch.Tensor, is_pre_event: bool) -> None:
        indices = (spike_tensor == 1).nonzero(as_tuple=False)
        if indices.numel() == 0:
            return
        if is_pre_event:
            batch_idx = indices[:, 0].repeat_interleave(n_post)
            pre_idx = indices[:, 1].repeat_interleave(n_post)
            post_idx = torch.arange(n_post, device=device).repeat(indices.size(0))
            e_type = torch.tensor([1.0, 0.0], device=device)
        else:
            batch_idx = indices[:, 0].repeat_interleave(n_pre)
            post_idx = indices[:, 1].repeat_interleave(n_pre)
            pre_idx = torch.arange(n_pre, device=device).repeat(indices.size(0))
            e_type = torch.tensor([0.0, 1.0], device=device)
        time_idx = indices[:, 2].repeat_interleave(n_post if is_pre_event else n_pre)
        pos = time_idx + (L - 1)
        time_indices = pos.unsqueeze(1) - idx_range.view(1, -1)
        pre_hist = pad_pre[batch_idx.unsqueeze(1), pre_idx.unsqueeze(1), time_indices]
        post_hist = pad_post[batch_idx.unsqueeze(1), post_idx.unsqueeze(1), time_indices]
        histories.append(torch.stack([pre_hist, post_hist], dim=1))
        w_vals = weights[pre_idx, post_idx].unsqueeze(1)
        extras.append(torch.cat([w_vals, e_type.expand(w_vals.size(0), -1)], dim=1))
        pre_indices.append(pre_idx)
        post_indices.append(post_idx)

    _append_events(pre_spikes, True)
    _append_events(post_spikes, False)

    if not histories:
        return (
            torch.empty(0, 2, L, device=device),
            torch.empty(0, 3, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    return (
        torch.cat(histories, dim=0),
        torch.cat(extras, dim=0),
        torch.cat(pre_indices, dim=0),
        torch.cat(post_indices, dim=0),
    )


def _scatter_updates(
    delta: torch.Tensor, pre_idx: torch.Tensor, post_idx: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    delta_matrix = torch.zeros_like(weights)
    delta_matrix.index_put_((pre_idx, post_idx), delta, accumulate=True)
    weights.data.add_(delta_matrix)
    return delta_matrix


def run_unsup1(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = get_mnist_dataloaders(args.batch_size_images, args.seed)

    base_len = len(getattr(train_loader.dataset, "dataset", train_loader.dataset))
    prev_winners = torch.full((base_len,), -1, device=device, dtype=torch.long)

    network = DiehlCookNetwork().to(device)
    actor = GaussianPolicy(sigma=args.sigma_unsup1, extra_feature_dim=3).to(device)
    critic = ValueFunction(extra_feature_dim=3).to(device)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    metrics_path = os.path.join(args.result_dir, "metrics_train.txt")
    _ensure_metrics_file(metrics_path)

    for epoch in range(1, args.num_epochs + 1):
        epoch_sparse, epoch_div, epoch_stab, epoch_total = [], [], [], []
        for images, _, indices in train_loader:
            images = images.to(device)
            indices = indices.to(device)
            input_spikes = poisson_encode(images, args.T_unsup1, max_rate=args.max_rate).to(device)
            exc_spikes, _ = network(input_spikes)

            r_sparse, r_div = _compute_reward_components(exc_spikes, args.rho_target)

            batch_buffer = EpisodeBuffer()

            firing_rates = exc_spikes.mean(dim=2)
            winners = firing_rates.argmax(dim=1)

            for b in range(input_spikes.size(0)):
                prev = prev_winners[indices[b]].item()
                r_stab_value = 0.0 if prev < 0 else (1.0 if winners[b].item() == prev else -1.0)
                prev_winners[indices[b]] = winners[b]

                state, extra, pre_idx, post_idx = _gather_events(
                    input_spikes[b : b + 1], exc_spikes[b : b + 1], network.w_input_exc, args.spike_array_len
                )
                if state.numel() == 0:
                    epoch_sparse.append(r_sparse[b].item())
                    epoch_div.append(r_div[b].item())
                    epoch_stab.append(r_stab_value)
                    total_reward = (
                        args.alpha_sparse * r_sparse[b]
                        + args.alpha_div * r_div[b]
                        + args.alpha_stab * torch.tensor(r_stab_value, device=device)
                    )
                    epoch_total.append(total_reward.item())
                    continue

                action, log_prob, _ = actor(state, extra)
                value = critic(state, extra)

                with torch.no_grad():
                    # Δw_i(t) = η_w * s_scen * Δd_i(t); unsup1 uses s_scen = 1
                    _scatter_updates(args.local_lr * action, pre_idx, post_idx, network.w_input_exc)
                    torch.clamp_(network.w_input_exc, args.exc_clip_min, args.exc_clip_max)

                episode_buffer = EpisodeBuffer()
                for i in range(state.size(0)):
                    episode_buffer.append(state[i], extra[i], action[i], log_prob[i], value[i])
                total_reward = (
                    args.alpha_sparse * r_sparse[b]
                    + args.alpha_div * r_div[b]
                    + args.alpha_stab * torch.tensor(r_stab_value, device=device)
                )
                episode_buffer.finalize(total_reward)
                batch_buffer.extend(episode_buffer)

                epoch_sparse.append(r_sparse[b].item())
                epoch_div.append(r_div[b].item())
                epoch_stab.append(r_stab_value)
                epoch_total.append(total_reward.item())

            if len(batch_buffer) > 0:
                ppo_update(
                    actor,
                    critic,
                    batch_buffer,
                    optimizer_actor,
                    optimizer_critic,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, len(batch_buffer)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

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
