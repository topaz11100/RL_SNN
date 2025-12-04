import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from utils.metrics import compute_neuron_labels, evaluate_labeling, plot_delta_t_delta_d, plot_weight_histograms
from data.mnist import get_mnist_dataloaders

from rl.buffers import EventBatchBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update_events
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pre_spikes.device
    batch_size, n_pre, T = pre_spikes.shape
    n_post = post_spikes.shape[1]

    pad_pre = F.pad(pre_spikes, (L - 1, 0))
    pad_post = F.pad(post_spikes, (L - 1, 0))
    pre_windows = pad_pre.unfold(2, L, 1)
    post_windows = pad_post.unfold(2, L, 1)

    states_list = []
    extras_list = []
    pre_indices_list = []
    post_indices_list = []
    batch_indices_list = []

    # Optimized: build events only for real spikes to avoid large expanded tensors.
    pre_events = pre_spikes.nonzero(as_tuple=False)
    if pre_events.numel() > 0:
        batch_pre = pre_events[:, 0].repeat_interleave(n_post)
        pre_idx = pre_events[:, 1].repeat_interleave(n_post)
        time_idx = pre_events[:, 2].repeat_interleave(n_post)
        post_idx = torch.arange(n_post, device=device).repeat(pre_events.size(0))

        if valid_mask is not None:
            keep_mask = valid_mask[pre_idx, post_idx]
            keep = keep_mask.nonzero(as_tuple=False).squeeze(1)
            batch_pre = batch_pre.index_select(0, keep)
            pre_idx = pre_idx.index_select(0, keep)
            time_idx = time_idx.index_select(0, keep)
            post_idx = post_idx.index_select(0, keep)

        if pre_idx.numel() > 0:
            pre_hist = pre_windows[batch_pre, pre_idx, time_idx]
            post_hist = post_windows[batch_pre, post_idx, time_idx]
            states_list.append(torch.stack([pre_hist, post_hist], dim=1))

            weights_pre = weights[pre_idx, post_idx].unsqueeze(1)
            event_type = torch.tensor([1.0, 0.0], device=device, dtype=weights.dtype).expand(weights_pre.size(0), -1)
            extras_list.append(torch.cat([weights_pre, event_type], dim=1))

            pre_indices_list.append(pre_idx)
            post_indices_list.append(post_idx)
            batch_indices_list.append(batch_pre)

    post_events = post_spikes.nonzero(as_tuple=False)
    if post_events.numel() > 0:
        batch_post = post_events[:, 0].repeat_interleave(n_pre)
        post_idx = post_events[:, 1].repeat_interleave(n_pre)
        time_idx = post_events[:, 2].repeat_interleave(n_pre)
        pre_idx = torch.arange(n_pre, device=device).repeat(post_events.size(0))

        if valid_mask is not None:
            keep_mask = valid_mask[pre_idx, post_idx]
            keep = keep_mask.nonzero(as_tuple=False).squeeze(1)
            batch_post = batch_post.index_select(0, keep)
            pre_idx = pre_idx.index_select(0, keep)
            time_idx = time_idx.index_select(0, keep)
            post_idx = post_idx.index_select(0, keep)

        if pre_idx.numel() > 0:
            pre_hist = pre_windows[batch_post, pre_idx, time_idx]
            post_hist = post_windows[batch_post, post_idx, time_idx]
            states_list.append(torch.stack([pre_hist, post_hist], dim=1))

            weights_post = weights[pre_idx, post_idx].unsqueeze(1)
            event_type = torch.tensor([0.0, 1.0], device=device, dtype=weights.dtype).expand(weights_post.size(0), -1)
            extras_list.append(torch.cat([weights_post, event_type], dim=1))

            pre_indices_list.append(pre_idx)
            post_indices_list.append(post_idx)
            batch_indices_list.append(batch_post)

    if not states_list:
        empty_state = torch.empty((0, 2, L), device=device, dtype=pre_spikes.dtype)
        empty_extras = torch.empty((0, 3), device=device, dtype=weights.dtype)
        empty_index = torch.empty((0,), device=device, dtype=torch.long)
        return empty_state, empty_extras, empty_index, empty_index, empty_index

    return (
        torch.cat(states_list, dim=0),
        torch.cat(extras_list, dim=0),
        torch.cat(pre_indices_list, dim=0),
        torch.cat(post_indices_list, dim=0),
        torch.cat(batch_indices_list, dim=0),
    )


def _forward_in_event_batches(actor, critic, states, extras, batch_size):
    actions, log_probs, values = [], [], []
    extras_available = extras.numel() > 0
    for start in range(0, states.size(0), batch_size):
        end = start + batch_size
        states_mb = states[start:end]
        extras_mb = extras[start:end] if extras_available else None
        action_mb, log_prob_mb, _ = actor(states_mb, extras_mb)
        value_mb = critic(states_mb, extras_mb)
        actions.append(action_mb)
        log_probs.append(log_prob_mb)
        values.append(value_mb)
    return torch.cat(actions, dim=0), torch.cat(log_probs, dim=0), torch.cat(values, dim=0)


def _apply_weight_updates(delta, connection_ids, pre_idx, post_idx, network, args):
    with torch.no_grad():
        for conn_id, weight, clip_min, clip_max, mask in [
            (0, network.w_input_exc, args.exc_clip_min, args.exc_clip_max, None),
            (1, network.w_inh_exc, args.inh_clip_min, args.inh_clip_max, network.inh_exc_mask),
        ]:
            event_mask = connection_ids == conn_id
            if event_mask.sum() == 0:
                continue
            delta_conn = delta[event_mask]
            pre_conn = pre_idx[event_mask]
            post_conn = post_idx[event_mask]
            delta_matrix = torch.zeros_like(weight)
            delta_matrix.index_put_((pre_conn, post_conn), delta_conn, accumulate=True)
            if mask is not None:
                delta_matrix = delta_matrix * mask
            weight.add_(delta_matrix)
            weight.clamp_(clip_min, clip_max)


def _collect_firing_rates(network: DiehlCookNetwork, loader, device, args):
    network.eval()
    rates, labels = [], []
    with torch.no_grad():
        for images, lbls, _ in loader:
            images = images.to(device)
            lbls = lbls.to(device)
            spikes = poisson_encode(images, args.T_unsup1, max_rate=args.max_rate).to(device)
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


def run_unsup1(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(args.batch_size_images, args.seed)

    base_len = len(getattr(train_loader.dataset, "dataset", train_loader.dataset))
    prev_winners = torch.full((base_len,), -1, device=device, dtype=torch.long)
    winner_counts = torch.zeros(args.N_E, device=device)
    total_seen = 0.0

    lif_params = LIFParams(dt=args.dt)
    network = DiehlCookNetwork(n_exc=args.N_E, n_inh=args.N_E, exc_params=lif_params, inh_params=lif_params).to(device)
    actor = GaussianPolicy(sigma=args.sigma_unsup1, extra_feature_dim=3).to(device)
    critic = ValueFunction(extra_feature_dim=3).to(device)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
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
    delta_t_values = []
    delta_d_values = []
    for epoch in range(1, args.num_epochs + 1):
        epoch_sparse, epoch_div, epoch_stab, epoch_total = [], [], [], []
        for batch_idx, (images, _, indices) in enumerate(train_loader, start=1):
            images = images.to(device)
            indices = indices.to(device)
            input_spikes = poisson_encode(images, args.T_unsup1, max_rate=args.max_rate).to(device)
            exc_spikes, inh_spikes = network(input_spikes)

            r_sparse, firing_rates = _compute_sparse_reward(exc_spikes, args.rho_target)

            winners = firing_rates.argmax(dim=1)
            event_buffer = EventBatchBuffer()

            prev_values = prev_winners[indices]
            stable_mask = prev_values >= 0
            r_stab = torch.where(
                stable_mask & (winners == prev_values),
                torch.tensor(1.0, device=device),
                torch.tensor(0.0, device=device),
            )
            r_stab = torch.where(
                stable_mask & (winners != prev_values), torch.tensor(-1.0, device=device), r_stab
            )
            prev_winners[indices] = winners

            one_hot_winners = F.one_hot(winners, num_classes=args.N_E).to(dtype=winner_counts.dtype)
            cumulative_counts = winner_counts.unsqueeze(0) + torch.cumsum(one_hot_winners, dim=0)
            total_seen_tensor = torch.tensor(total_seen, device=device, dtype=winner_counts.dtype)
            total_seen_per = total_seen_tensor + torch.arange(
                1, winners.size(0) + 1, device=device, dtype=winner_counts.dtype
            )
            uniform = 1.0 / args.N_E
            r_div = -((cumulative_counts / total_seen_per.unsqueeze(1) - uniform).pow(2).sum(dim=1))

            winner_counts += one_hot_winners.sum(dim=0)
            total_seen += winners.size(0)

            total_reward = args.alpha_sparse * r_sparse + args.alpha_div * r_div + args.alpha_stab * r_stab

            state_exc, extra_exc, pre_exc, post_exc, batch_exc = _gather_events(
                input_spikes, exc_spikes, network.w_input_exc, args.spike_array_len
            )
            state_inh, extra_inh, pre_inh, post_inh, batch_inh = _gather_events(
                inh_spikes,
                exc_spikes,
                network.w_inh_exc,
                args.spike_array_len,
                valid_mask=network.inh_exc_mask,
            )

            if state_exc.numel() > 0:
                event_buffer.add(indices[batch_exc], 0, state_exc, extra_exc, pre_exc, post_exc, batch_exc)
            if state_inh.numel() > 0:
                event_buffer.add(indices[batch_inh], 1, state_inh, extra_inh, pre_inh, post_inh, batch_inh)

            epoch_sparse.extend(r_sparse.tolist())
            epoch_div.extend(r_div.tolist())
            epoch_stab.extend(r_stab.tolist())
            epoch_total.extend(total_reward.tolist())

            if len(event_buffer) > 0:
                states, extras, _, connection_ids, pre_idx, post_idx, batch_idx_events = event_buffer.flatten()
                rewards_tensor = total_reward.detach()

                actions, log_probs_old, values_old = _forward_in_event_batches(
                    actor, critic, states, extras, batch_size=args.event_batch_size
                )
                returns = rewards_tensor[batch_idx_events]
                advantages = returns - values_old.detach()

                ppo_update_events(
                    actor,
                    critic,
                    states,
                    extras,
                    actions.detach(),
                    log_probs_old.detach(),
                    returns.detach(),
                    advantages.detach(),
                    optimizer_actor,
                    optimizer_critic,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, states.size(0)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

                delta = args.local_lr * s_scen * actions.detach()
                _apply_weight_updates(delta, connection_ids, pre_idx, post_idx, network, args)
                delta_t_values.append(_extract_delta_t(states).detach().cpu())
                delta_d_values.append(actions.detach().cpu())

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

    delta_t_concat = torch.cat(delta_t_values, dim=0) if delta_t_values else torch.empty(0)
    delta_d_concat = torch.cat(delta_d_values, dim=0) if delta_d_values else torch.empty(0)
    if delta_t_concat.numel() > 0 and delta_d_concat.numel() > 0:
        plot_delta_t_delta_d(delta_t_concat, delta_d_concat, os.path.join(args.result_dir, "delta_t_delta_d.png"))

    plot_weight_histograms(
        w_input_exc_before, network.w_input_exc.detach().cpu(), os.path.join(args.result_dir, "hist_input_exc.png")
    )
    plot_weight_histograms(
        w_inh_exc_before, network.w_inh_exc.detach().cpu(), os.path.join(args.result_dir, "hist_inh_exc.png")
    )
