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




def _forward_in_event_batches(actor, critic, states, extras, batch_size):
    actions_cat, log_probs_cat, values_cat = None, None, None
    extras_available = extras.numel() > 0
    for start in range(0, states.size(0), batch_size):
        end = start + batch_size
        states_mb = states[start:end]
        extras_mb = extras[start:end] if extras_available else None
        action_mb, log_prob_mb, _ = actor(states_mb, extras_mb)
        log_prob_mb = log_prob_mb.detach()
        value_mb = critic(states_mb, extras_mb)
        if actions_cat is None:
            actions_cat = action_mb
            log_probs_cat = log_prob_mb
            values_cat = value_mb
        else:
            actions_cat = torch.cat((actions_cat, action_mb), dim=0)
            log_probs_cat = torch.cat((log_probs_cat, log_prob_mb), dim=0)
            values_cat = torch.cat((values_cat, value_mb), dim=0)
    return actions_cat, log_probs_cat, values_cat


def _scatter_updates(
    delta: torch.Tensor,
    pre_idx: torch.Tensor,
    post_idx: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> None:
    if delta.numel() == 0:
        return
    if valid_mask is not None:
        delta = delta * valid_mask[pre_idx, post_idx]
    weights.index_put_((pre_idx, post_idx), delta, accumulate=True)


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


def analyze_stdp_profile(
    network: DiehlCookNetwork,
    actor_exc: GaussianPolicy,
    actor_inh: GaussianPolicy,
    critic_exc: ValueFunction,
    critic_inh: ValueFunction,
    loader,
    args,
    device: torch.device,
) -> None:
    network_state = network.training
    actor_exc_state = actor_exc.training
    actor_inh_state = actor_inh.training
    critic_exc_state = critic_exc.training
    critic_inh_state = critic_inh.training
    network.eval()
    actor_exc.eval()
    actor_inh.eval()
    critic_exc.eval()
    critic_inh.eval()

    try:
        images, _, _ = next(iter(loader))
    except StopIteration:
        return

    estimated_events = args.batch_size_images * args.spike_array_len * (784 + 2 * args.N_E)
    event_buffer = EventBatchBuffer(initial_capacity=max(100_000, estimated_events))

    def _plot(delta_t: torch.Tensor, delta_d: torch.Tensor, suffix: str) -> None:
        if delta_t.numel() > 0 and delta_d.numel() > 0:
            plot_delta_t_delta_d(delta_t.cpu(), delta_d.cpu(), os.path.join(args.result_dir, suffix))

    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        input_spikes = poisson_encode(images, args.T_unsup2, max_rate=args.max_rate)
        exc_spikes, inh_spikes = network(input_spikes)

        event_buffer.reset()
        padded_cache = {}

        def _get_padded(spikes: torch.Tensor) -> torch.Tensor:
            key = id(spikes)
            if key not in padded_cache:
                padded_cache[key] = F.pad(spikes, (args.spike_array_len - 1, 0))
            return padded_cache[key]

        gather_events(
            input_spikes,
            exc_spikes,
            network.w_input_exc,
            args.spike_array_len,
            event_buffer,
            0,
            padded_pre=_get_padded(input_spikes),
            padded_post=_get_padded(exc_spikes),
        )
        gather_events(
            inh_spikes,
            exc_spikes,
            network.w_inh_exc,
            args.spike_array_len,
            event_buffer,
            1,
            valid_mask=network.inh_exc_mask,
            padded_pre=_get_padded(inh_spikes),
            padded_post=_get_padded(exc_spikes),
        )

        if len(event_buffer) > 0:
            states, extras, connection_ids, _, _, _ = event_buffer.flatten()

            exc_mask = connection_ids == 0
            inh_mask = connection_ids == 1

            if exc_mask.any():
                actions_exc, _, _ = _forward_in_event_batches(
                    actor_exc,
                    critic_exc,
                    states[exc_mask],
                    extras[exc_mask],
                    batch_size=args.event_batch_size,
                )
                _plot(_extract_delta_t(states[exc_mask]), actions_exc.detach(), "delta_t_delta_d_exc.png")

            if inh_mask.any():
                actions_inh, _, _ = _forward_in_event_batches(
                    actor_inh,
                    critic_inh,
                    states[inh_mask],
                    extras[inh_mask],
                    batch_size=args.event_batch_size,
                )
                _plot(_extract_delta_t(states[inh_mask]), actions_inh.detach(), "delta_t_delta_d_inh.png")

    if network_state:
        network.train()
    if actor_exc_state:
        actor_exc.train()
    if actor_inh_state:
        actor_inh.train()
    if critic_exc_state:
        critic_exc.train()
    if critic_inh_state:
        critic_inh.train()


def run_unsup2(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(args.batch_size_images, args.seed)

    prev_winners = torch.full((60000,), -1, device=device, dtype=torch.long)
    winner_counts = torch.zeros(args.N_E, device=device)
    total_seen = torch.zeros((), device=device, dtype=winner_counts.dtype)

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

    estimated_events = args.batch_size_images * args.spike_array_len * (784 + 2 * args.N_E)
    event_buffer = EventBatchBuffer(initial_capacity=max(100_000, estimated_events))

    s_scen = 1.0
    pos_reward = torch.tensor(1.0, device=device)
    neg_reward = torch.tensor(-1.0, device=device)
    for epoch in range(1, args.num_epochs + 1):
        total_sparse = torch.zeros((), device=device)
        total_div = torch.zeros((), device=device)
        total_stab = torch.zeros((), device=device)
        total_reward_sum = torch.zeros((), device=device)
        total_samples = torch.zeros((), device=device)
        for batch_idx, (images, _, indices) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            input_spikes = poisson_encode(images, args.T_unsup2, max_rate=args.max_rate)
            exc_spikes, inh_spikes = network(input_spikes)

            r_sparse, firing_rates = _compute_sparse_reward(exc_spikes, args.rho_target)

            winners = firing_rates.argmax(dim=1)

            prev_values = prev_winners[indices]
            first_visit = prev_values == -1
            stable_mask = ~first_visit
            r_stab = torch.zeros_like(prev_values, dtype=torch.float32, device=device)
            r_stab = torch.where(stable_mask & (winners == prev_values), pos_reward, r_stab)
            r_stab = torch.where(stable_mask & (winners != prev_values), neg_reward, r_stab)
            prev_winners[indices] = winners

            one_hot_winners = F.one_hot(winners, num_classes=args.N_E).to(dtype=winner_counts.dtype)
            cumulative_counts = winner_counts.unsqueeze(0) + torch.cumsum(one_hot_winners, dim=0)
            total_seen_per = total_seen + torch.arange(
                1, winners.size(0) + 1, device=device, dtype=winner_counts.dtype
            )
            uniform = 1.0 / args.N_E
            r_div = -((cumulative_counts / total_seen_per.unsqueeze(1) - uniform).pow(2).sum(dim=1))

            winner_counts += one_hot_winners.sum(dim=0)
            total_seen.add_(winners.size(0))

            total_reward = args.alpha_sparse * r_sparse + args.alpha_div * r_div + args.alpha_stab * r_stab

            event_buffer.reset()

            padded_cache = {}

            def _get_padded(spikes: torch.Tensor) -> torch.Tensor:
                key = id(spikes)
                if key not in padded_cache:
                    padded_cache[key] = F.pad(spikes, (args.spike_array_len - 1, 0))
                return padded_cache[key]

            gather_events(
                input_spikes,
                exc_spikes,
                network.w_input_exc,
                args.spike_array_len,
                event_buffer,
                0,
                padded_pre=_get_padded(input_spikes),
                padded_post=_get_padded(exc_spikes),
            )
            gather_events(
                inh_spikes,
                exc_spikes,
                network.w_inh_exc,
                args.spike_array_len,
                event_buffer,
                1,
                valid_mask=network.inh_exc_mask,
                padded_pre=_get_padded(inh_spikes),
                padded_post=_get_padded(exc_spikes),
            )

            batch_size = winners.numel()
            sample_increment = winners.new_full((), batch_size, dtype=torch.float32)
            total_samples = total_samples + sample_increment
            total_sparse = total_sparse + r_sparse.sum()
            total_div = total_div + r_div.sum()
            total_stab = total_stab + r_stab.sum()
            total_reward_sum = total_reward_sum + total_reward.sum()

            if len(event_buffer) > 0:
                states, extras, connection_ids, pre_idx, post_idx, batch_idx_events = event_buffer.flatten()
                rewards_tensor = total_reward.detach()

                # Excitatory pathway
                exc_mask = connection_ids == 0
                if exc_mask.any():
                    states_exc = states[exc_mask]
                    extras_exc = extras[exc_mask]
                    batch_exc_events = batch_idx_events[exc_mask]
                    pre_exc_events = pre_idx[exc_mask]
                    post_exc_events = post_idx[exc_mask]
                    # Actor sees extras built from the simulation-time weights captured
                    # in gather_events, ensuring on-policy consistency when evaluating
                    # actions after the episode rollout.
                    actions_exc, logp_exc, values_exc = _forward_in_event_batches(
                        actor_exc,
                        critic_exc,
                        states_exc,
                        extras_exc,
                        batch_size=args.event_batch_size,
                    )
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

                # Inhibitory pathway
                inh_mask = connection_ids == 1
                if inh_mask.any():
                    states_inh = states[inh_mask]
                    extras_inh = extras[inh_mask]
                    batch_inh_events = batch_idx_events[inh_mask]
                    pre_inh_events = pre_idx[inh_mask]
                    post_inh_events = post_idx[inh_mask]
                    actions_inh, logp_inh, values_inh = _forward_in_event_batches(
                        actor_inh,
                        critic_inh,
                        states_inh,
                        extras_inh,
                        batch_size=args.event_batch_size,
                    )
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

            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                logger.info(
                    "Epoch %d/%d | Batch %d/%d",
                    epoch,
                    args.num_epochs,
                    batch_idx,
                    len(train_loader),
                )

        mean_sparse = (total_sparse / total_samples).item() if total_samples.item() > 0 else 0.0
        mean_div = (total_div / total_samples).item() if total_samples.item() > 0 else 0.0
        mean_stab = (total_stab / total_samples).item() if total_samples.item() > 0 else 0.0
        mean_total = (total_reward_sum / total_samples).item() if total_samples.item() > 0 else 0.0

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

    plot_weight_histograms(
        w_input_exc_before, network.w_input_exc.detach().cpu(), os.path.join(args.result_dir, "hist_input_exc.png")
    )
    plot_weight_histograms(
        w_inh_exc_before, network.w_inh_exc.detach().cpu(), os.path.join(args.result_dir, "hist_inh_exc.png")
    )

    analyze_stdp_profile(
        network, actor_exc, actor_inh, critic_exc, critic_inh, train_loader, args, device
    )
