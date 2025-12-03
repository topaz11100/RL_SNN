import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from utils.metrics import compute_neuron_labels, evaluate_labeling, plot_delta_t_delta_d, plot_weight_histograms
from data.mnist import get_mnist_dataloaders
from rl.buffers import EpisodeBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.lif import LIFParams
from snn.network_diehl_cook import DiehlCookNetwork


def _ensure_metrics_file(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("epoch\tR_sparse\tR_div\tR_stab\tR_total\n")


def _ensure_eval_file(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("epoch\taccuracy\n")


def _compute_sparse_reward(exc_spikes: torch.Tensor, rho_target: float) -> Tuple[torch.Tensor, torch.Tensor]:
    firing_rates = exc_spikes.mean(dim=2)
    r_sparse = -((firing_rates.mean(dim=1) - rho_target) ** 2)
    return r_sparse, firing_rates


def _gather_events(
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    weights: torch.Tensor,
    L: int,
    valid_mask: Optional[torch.Tensor] = None,
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
            repeat_count = n_post
        else:
            batch_idx = indices[:, 0].repeat_interleave(n_pre)
            post_idx = indices[:, 1].repeat_interleave(n_pre)
            pre_idx = torch.arange(n_pre, device=device).repeat(indices.size(0))
            e_type = torch.tensor([0.0, 1.0], device=device)
            repeat_count = n_pre
        time_idx = indices[:, 2].repeat_interleave(repeat_count)
        if valid_mask is not None:
            mask_flat = valid_mask[pre_idx, post_idx] > 0.5
            if mask_flat.sum() == 0:
                return
            batch_idx = batch_idx[mask_flat]
            pre_idx = pre_idx[mask_flat]
            post_idx = post_idx[mask_flat]
            time_idx = time_idx[mask_flat]
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
    delta: torch.Tensor,
    pre_idx: torch.Tensor,
    post_idx: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> None:
    delta_matrix = torch.zeros_like(weights)
    delta_matrix.index_put_((pre_idx, post_idx), delta, accumulate=True)
    if valid_mask is not None:
        delta_matrix = delta_matrix * valid_mask
    weights.data.add_(delta_matrix)


def _collect_firing_rates(network: DiehlCookNetwork, loader, device, args):
    network.eval()
    rates, labels = [], []
    with torch.no_grad():
        for images, lbls, _ in loader:
            images = images.to(device)
            lbls = lbls.to(device)
            spikes = poisson_encode(images, args.T_unsup2, max_rate=args.max_rate).to(device)
            exc_spikes, _ = network(spikes)
            rates.append(exc_spikes.mean(dim=2))
            labels.append(lbls)
    network.train()
    return torch.cat(rates, dim=0), torch.cat(labels, dim=0)


def _extract_delta_t(states: torch.Tensor) -> torch.Tensor:
    if states.numel() == 0:
        return torch.empty(0, device=states.device)
    L = states.size(2)
    time_idx = torch.arange(L, device=states.device)
    last_pre = torch.where(states[:, 0, :] > 0, time_idx, torch.full_like(time_idx, -1)).max(dim=1).values
    last_post = torch.where(states[:, 1, :] > 0, time_idx, torch.full_like(time_idx, -1)).max(dim=1).values
    return last_pre - last_post


def run_unsup2(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(args.batch_size_images, args.seed)

    base_len = len(getattr(train_loader.dataset, "dataset", train_loader.dataset))
    prev_winners = torch.full((base_len,), -1, device=device, dtype=torch.long)
    winner_counts = torch.zeros(args.N_E, device=device)
    total_seen = 0.0

    lif_params = LIFParams(dt=args.dt)
    network = DiehlCookNetwork(n_exc=args.N_E, n_inh=args.N_E, exc_params=lif_params, inh_params=lif_params).to(device)

    actor_exc = GaussianPolicy(sigma=args.sigma_unsup2, extra_feature_dim=3).to(device)
    actor_inh = GaussianPolicy(sigma=args.sigma_unsup2, extra_feature_dim=3).to(device)
    critic = ValueFunction(extra_feature_dim=3).to(device)
    optimizer_actor_exc = torch.optim.Adam(actor_exc.parameters(), lr=args.lr_actor)
    optimizer_actor_inh = torch.optim.Adam(actor_inh.parameters(), lr=args.lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    w_input_exc_before = network.w_input_exc.detach().cpu().clone()
    w_inh_exc_before = network.w_inh_exc.detach().cpu().clone()

    metrics_path = os.path.join(args.result_dir, "metrics_train.txt")
    metrics_val = os.path.join(args.result_dir, "metrics_val.txt")
    metrics_test = os.path.join(args.result_dir, "metrics_test.txt")
    _ensure_metrics_file(metrics_path)
    _ensure_eval_file(metrics_val)
    _ensure_eval_file(metrics_test)

    s_scen = 1.0
    delta_t_exc = []
    delta_d_exc = []
    delta_t_inh = []
    delta_d_inh = []
    for epoch in range(1, args.num_epochs + 1):
        epoch_sparse, epoch_div, epoch_stab, epoch_total = [], [], [], []
        for batch_idx, (images, _, indices) in enumerate(train_loader, start=1):
            images = images.to(device)
            indices = indices.to(device)
            input_spikes = poisson_encode(images, args.T_unsup2, max_rate=args.max_rate).to(device)
            exc_spikes, inh_spikes = network(input_spikes)

            r_sparse, firing_rates = _compute_sparse_reward(exc_spikes, args.rho_target)

            batch_buffer_exc = EpisodeBuffer()
            batch_buffer_inh = EpisodeBuffer()

            winners = firing_rates.argmax(dim=1)

            for b in range(input_spikes.size(0)):
                prev = prev_winners[indices[b]].item()
                r_stab_value = 0.0 if prev < 0 else (1.0 if winners[b].item() == prev else -1.0)
                prev_winners[indices[b]] = winners[b]

                winner_counts[winners[b]] += 1
                total_seen += 1
                p = winner_counts / total_seen
                uniform = 1.0 / args.N_E
                r_div_value = -(p - uniform).pow(2).sum()

                total_reward = (
                    args.alpha_sparse * r_sparse[b]
                    + args.alpha_div * r_div_value
                    + args.alpha_stab * torch.tensor(r_stab_value, device=device)
                )

                state_exc, extra_exc, pre_exc, post_exc = _gather_events(
                    input_spikes[b : b + 1], exc_spikes[b : b + 1], network.w_input_exc, args.spike_array_len
                )
                if state_exc.numel() > 0:
                    action_exc, logp_exc, _ = actor_exc(state_exc, extra_exc)
                    value_exc = critic(state_exc, extra_exc)
                    with torch.no_grad():
                        _scatter_updates(args.local_lr * s_scen * action_exc, pre_exc, post_exc, network.w_input_exc)
                        network.w_input_exc.clamp_(args.exc_clip_min, args.exc_clip_max)
                    delta_t_exc.append(_extract_delta_t(state_exc).detach().cpu())
                    delta_d_exc.append(action_exc.detach().cpu())

                    buffer_exc = EpisodeBuffer()
                    for i in range(state_exc.size(0)):
                        buffer_exc.append(state_exc[i], extra_exc[i], action_exc[i], logp_exc[i], value_exc[i])
                    buffer_exc.finalize(total_reward)
                    batch_buffer_exc.extend(buffer_exc)

                state_inh, extra_inh, pre_inh, post_inh = _gather_events(
                    inh_spikes[b : b + 1],
                    exc_spikes[b : b + 1],
                    network.w_inh_exc,
                    args.spike_array_len,
                    valid_mask=network.inh_exc_mask,
                )
                if state_inh.numel() > 0:
                    action_inh, logp_inh, _ = actor_inh(state_inh, extra_inh)
                    value_inh = critic(state_inh, extra_inh)
                    with torch.no_grad():
                        _scatter_updates(
                            args.local_lr * s_scen * action_inh,
                            pre_inh,
                            post_inh,
                            network.w_inh_exc,
                            valid_mask=network.inh_exc_mask,
                        )
                        network.w_inh_exc.clamp_(args.inh_clip_min, args.inh_clip_max)
                    delta_t_inh.append(_extract_delta_t(state_inh).detach().cpu())
                    delta_d_inh.append(action_inh.detach().cpu())

                    buffer_inh = EpisodeBuffer()
                    for i in range(state_inh.size(0)):
                        buffer_inh.append(state_inh[i], extra_inh[i], action_inh[i], logp_inh[i], value_inh[i])
                    buffer_inh.finalize(total_reward)
                    batch_buffer_inh.extend(buffer_inh)

                epoch_sparse.append(r_sparse[b].item())
                epoch_div.append(r_div_value.item())
                epoch_stab.append(r_stab_value)
                epoch_total.append(total_reward.item())

            if len(batch_buffer_exc) > 0:
                ppo_update(
                    actor_exc,
                    critic,
                    batch_buffer_exc,
                    optimizer_actor_exc,
                    optimizer_critic,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, len(batch_buffer_exc)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

            if len(batch_buffer_inh) > 0:
                ppo_update(
                    actor_inh,
                    critic,
                    batch_buffer_inh,
                    optimizer_actor_inh,
                    optimizer_critic,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, len(batch_buffer_inh)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                logger.info(
                    "Epoch %d/%d | Batch %d/%d",
                    epoch,
                    args.num_epochs,
                    batch_idx,
                    len(train_loader),
                )

        mean_sparse = sum(epoch_sparse) / len(epoch_sparse)
        mean_div = sum(epoch_div) / len(epoch_div)
        mean_stab = sum(epoch_stab) / len(epoch_stab)
        mean_total = sum(epoch_total) / len(epoch_total)

        with open(metrics_path, "a") as f:
            f.write(f"{epoch}\t{mean_sparse:.6f}\t{mean_div:.6f}\t{mean_stab:.6f}\t{mean_total:.6f}\n")

        train_rates, train_labels = _collect_firing_rates(network, train_loader, device, args)
        neuron_labels = compute_neuron_labels(train_rates, train_labels, num_classes=10)

        val_rates, val_labels = _collect_firing_rates(network, val_loader, device, args)
        val_acc, _ = evaluate_labeling(val_rates, val_labels, neuron_labels, 10, os.path.join(args.result_dir, f"val_confusion_epoch{epoch}"))
        test_rates, test_labels = _collect_firing_rates(network, test_loader, device, args)
        test_acc, _ = evaluate_labeling(
            test_rates, test_labels, neuron_labels, 10, os.path.join(args.result_dir, f"test_confusion_epoch{epoch}")
        )

        with open(metrics_val, "a") as f:
            f.write(f"{epoch}\t{val_acc:.6f}\n")
        with open(metrics_test, "a") as f:
            f.write(f"{epoch}\t{test_acc:.6f}\n")

        if epoch % args.log_interval == 0:
            logger.info(
                "Epoch %d | R_sparse %.4f | R_div %.4f | R_stab %.4f | R_total %.4f | Val acc %.4f | Test acc %.4f",
                epoch,
                mean_sparse,
                mean_div,
                mean_stab,
                mean_total,
                val_acc,
                test_acc,
            )

    delta_t_exc_concat = torch.cat(delta_t_exc, dim=0) if delta_t_exc else torch.empty(0)
    delta_d_exc_concat = torch.cat(delta_d_exc, dim=0) if delta_d_exc else torch.empty(0)
    if delta_t_exc_concat.numel() > 0 and delta_d_exc_concat.numel() > 0:
        plot_delta_t_delta_d(
            delta_t_exc_concat, delta_d_exc_concat, os.path.join(args.result_dir, "delta_t_delta_d_exc.png")
        )

    delta_t_inh_concat = torch.cat(delta_t_inh, dim=0) if delta_t_inh else torch.empty(0)
    delta_d_inh_concat = torch.cat(delta_d_inh, dim=0) if delta_d_inh else torch.empty(0)
    if delta_t_inh_concat.numel() > 0 and delta_d_inh_concat.numel() > 0:
        plot_delta_t_delta_d(
            delta_t_inh_concat, delta_d_inh_concat, os.path.join(args.result_dir, "delta_t_delta_d_inh.png")
        )

    plot_weight_histograms(
        w_input_exc_before, network.w_input_exc.detach().cpu(), os.path.join(args.result_dir, "hist_input_exc.png")
    )
    plot_weight_histograms(
        w_inh_exc_before, network.w_inh_exc.detach().cpu(), os.path.join(args.result_dir, "hist_inh_exc.png")
    )
