import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from data.mnist import get_mnist_dataloaders

from rl.buffers import EventBatchBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update_events
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.lif import LIFParams
from snn.network_diehl_cook import DiehlCookNetwork
from utils.event_utils import gather_events
from utils.metrics import compute_neuron_labels, evaluate_labeling, plot_delta_t_delta_d, plot_weight_histograms


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
            images = images.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            spikes = poisson_encode(images, args.T_unsup2, max_rate=args.max_rate)
            exc_spikes, _ = network(spikes)
            rates.append(exc_spikes.mean(dim=2).detach().cpu())
            labels.append(lbls.detach().cpu())
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
    critic_exc = ValueFunction(extra_feature_dim=3).to(device)
    critic_inh = ValueFunction(extra_feature_dim=3).to(device)
    optimizer_actor_exc = torch.optim.Adam(actor_exc.parameters(), lr=args.lr_actor)
    optimizer_actor_inh = torch.optim.Adam(actor_inh.parameters(), lr=args.lr_actor)
    optimizer_critic_exc = torch.optim.Adam(critic_exc.parameters(), lr=args.lr_critic)
    optimizer_critic_inh = torch.optim.Adam(critic_inh.parameters(), lr=args.lr_critic)

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
            images = images.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            input_spikes = poisson_encode(images, args.T_unsup2, max_rate=args.max_rate)
            exc_spikes, inh_spikes = network(input_spikes)

            r_sparse, firing_rates = _compute_sparse_reward(exc_spikes, args.rho_target)

            winners = firing_rates.argmax(dim=1)

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

            state_exc, extra_exc, pre_exc, post_exc, batch_exc = gather_events(
                input_spikes, exc_spikes, network.w_input_exc, args.spike_array_len
            )
            state_inh, extra_inh, pre_inh, post_inh, batch_inh = gather_events(
                inh_spikes,
                exc_spikes,
                network.w_inh_exc,
                args.spike_array_len,
                valid_mask=network.inh_exc_mask,
            )

            event_buffer = EventBatchBuffer()
            if state_exc.numel() > 0:
                event_buffer.add(indices[batch_exc], 0, state_exc, extra_exc, pre_exc, post_exc, batch_exc)
            if state_inh.numel() > 0:
                event_buffer.add(indices[batch_inh], 1, state_inh, extra_inh, pre_inh, post_inh, batch_inh)

            epoch_sparse.append(r_sparse.detach())
            epoch_div.append(r_div.detach())
            epoch_stab.append(r_stab.detach())
            epoch_total.append(total_reward.detach())

            if len(event_buffer) > 0:
                states, extras, _, connection_ids, pre_idx, post_idx, batch_idx_events = event_buffer.flatten()
                rewards_tensor = total_reward.detach()

                # Excitatory pathway
                exc_mask = connection_ids == 0
                if exc_mask.any():
                    states_exc = states[exc_mask]
                    extras_exc = extras[exc_mask]
                    batch_exc_events = batch_idx_events[exc_mask]
                    pre_exc_events = pre_idx[exc_mask]
                    post_exc_events = post_idx[exc_mask]
                    actions_exc, logp_exc, _ = actor_exc(states_exc, extras_exc)
                    values_exc = critic_exc(states_exc, extras_exc)
                    returns_exc = rewards_tensor[batch_exc_events]
                    adv_exc = returns_exc - values_exc.detach()

                    ppo_update_events(
                        actor_exc,
                        critic_exc,
                        states_exc,
                        extras_exc,
                        actions_exc.detach(),
                        logp_exc.detach(),
                        returns_exc.detach(),
                        adv_exc.detach(),
                        optimizer_actor_exc,
                        optimizer_critic_exc,
                        ppo_epochs=args.ppo_epochs,
                        batch_size=min(args.ppo_batch_size, states_exc.size(0)),
                        eps_clip=args.ppo_eps,
                        c_v=1.0,
                    )

                    with torch.no_grad():
                        delta_exc = args.local_lr * s_scen * actions_exc.detach()
                        _scatter_updates(delta_exc, pre_exc_events, post_exc_events, network.w_input_exc)
                        network.w_input_exc.clamp_(args.exc_clip_min, args.exc_clip_max)
                    delta_t_exc.append(_extract_delta_t(states_exc).detach())
                    delta_d_exc.append(actions_exc.detach())

                # Inhibitory pathway
                inh_mask = connection_ids == 1
                if inh_mask.any():
                    states_inh = states[inh_mask]
                    extras_inh = extras[inh_mask]
                    batch_inh_events = batch_idx_events[inh_mask]
                    pre_inh_events = pre_idx[inh_mask]
                    post_inh_events = post_idx[inh_mask]
                    actions_inh, logp_inh, _ = actor_inh(states_inh, extras_inh)
                    values_inh = critic_inh(states_inh, extras_inh)
                    returns_inh = rewards_tensor[batch_inh_events]
                    adv_inh = returns_inh - values_inh.detach()

                    ppo_update_events(
                        actor_inh,
                        critic_inh,
                        states_inh,
                        extras_inh,
                        actions_inh.detach(),
                        logp_inh.detach(),
                        returns_inh.detach(),
                        adv_inh.detach(),
                        optimizer_actor_inh,
                        optimizer_critic_inh,
                        ppo_epochs=args.ppo_epochs,
                        batch_size=min(args.ppo_batch_size, states_inh.size(0)),
                        eps_clip=args.ppo_eps,
                        c_v=1.0,
                    )

                    with torch.no_grad():
                        delta_inh = args.local_lr * s_scen * actions_inh.detach()
                        _scatter_updates(
                            delta_inh,
                            pre_inh_events,
                            post_inh_events,
                            network.w_inh_exc,
                            valid_mask=network.inh_exc_mask,
                        )
                        network.w_inh_exc.clamp_(args.inh_clip_min, args.inh_clip_max)
                    delta_t_inh.append(_extract_delta_t(states_inh).detach())
                    delta_d_inh.append(actions_inh.detach())

            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                logger.info(
                    "Epoch %d/%d | Batch %d/%d",
                    epoch,
                    args.num_epochs,
                    batch_idx,
                    len(train_loader),
                )

        sparse_tensor = torch.cat(epoch_sparse) if epoch_sparse else torch.empty(0, device=device)
        div_tensor = torch.cat(epoch_div) if epoch_div else torch.empty(0, device=device)
        stab_tensor = torch.cat(epoch_stab) if epoch_stab else torch.empty(0, device=device)
        total_tensor = torch.cat(epoch_total) if epoch_total else torch.empty(0, device=device)
        mean_sparse = sparse_tensor.mean().item() if sparse_tensor.numel() > 0 else 0.0
        mean_div = div_tensor.mean().item() if div_tensor.numel() > 0 else 0.0
        mean_stab = stab_tensor.mean().item() if stab_tensor.numel() > 0 else 0.0
        mean_total = total_tensor.mean().item() if total_tensor.numel() > 0 else 0.0

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

    delta_t_exc_concat = torch.cat(delta_t_exc, dim=0).cpu() if delta_t_exc else torch.empty(0)
    delta_d_exc_concat = torch.cat(delta_d_exc, dim=0).cpu() if delta_d_exc else torch.empty(0)
    if delta_t_exc_concat.numel() > 0 and delta_d_exc_concat.numel() > 0:
        plot_delta_t_delta_d(
            delta_t_exc_concat, delta_d_exc_concat, os.path.join(args.result_dir, "delta_t_delta_d_exc.png")
        )

    delta_t_inh_concat = torch.cat(delta_t_inh, dim=0).cpu() if delta_t_inh else torch.empty(0)
    delta_d_inh_concat = torch.cat(delta_d_inh, dim=0).cpu() if delta_d_inh else torch.empty(0)
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
