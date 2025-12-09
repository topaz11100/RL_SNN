from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data.mnist import get_mnist_dataloaders

from rl.buffers import StreamingEventBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update_events
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.lif import LIFParams
from snn.network_diehl_cook import DiehlCookNetwork
from utils.event_utils import gather_events
from utils.metrics import (
    compute_neuron_labels,
    evaluate_labeling,
    plot_receptive_fields,
    plot_weight_histograms,
)
from utils.logging import resolve_path


def _ensure_metrics_file(path: str | Path) -> None:
    resolved = resolve_path(path)
    if not resolved.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("w") as f:
            f.write("epoch\tR_sparse\tR_div\tR_stab\tR_total\n")


def _ensure_eval_file(path: str | Path) -> None:
    resolved = resolve_path(path)
    if not resolved.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("w") as f:
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
    args,
    device: torch.device,
) -> None:
    result_dir = resolve_path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
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

    dt_range = torch.arange(-50, 51, device=device)
    window_size = args.spike_array_len
    states = torch.zeros((len(dt_range), 2, window_size), device=device)
    center = window_size // 2
    states[:, 1, center] = 1.0
    for idx, dt in enumerate(dt_range):
        pre_idx = center + int(dt.item())
        if 0 <= pre_idx < window_size:
            states[idx, 0, pre_idx] = 1.0

    weights_to_test = [-1.0, -0.5, 0.0, 0.5, 1.0]
    dt_cpu = dt_range.cpu().numpy()

    def _sweep_and_plot(actor: GaussianPolicy, out_name: str) -> None:
        # 확장자(.png) 제거 (만약 인자로 "stdp_sweep_exc.png"를 받았다면 "stdp_sweep_exc"만 추출)
        base_stem = Path(out_name).stem
        curves: list[tuple[float, torch.Tensor]] = []
        with torch.no_grad():
            for w_val in weights_to_test:
                extras = torch.zeros((len(dt_range), actor.extra_feature_dim), device=device)
                extras[:, 0] = w_val
                actions, _, _ = actor(states, extras)
                action_vals = actions.detach().cpu().view(-1)
                curves.append((w_val, action_vals))

                # [추가됨] 개별 그래프 저장
                plt.figure(figsize=(6, 5), dpi=100)
                plt.plot(dt_cpu, action_vals.numpy(), color='green', label=f"w={w_val}")
                plt.title(f"STDP Profile ({base_stem}, w={w_val})")
                plt.xlabel("Delta t")
                plt.ylabel("Delta w")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                # 파일명 예: stdp_sweep_exc_w_-1.0.png
                plt.savefig(result_dir / f"{base_stem}_w_{w_val}.png")
                plt.close()

        plt.figure(figsize=(8, 6), dpi=200)
        for w_val, action_vals in curves:
            plt.plot(dt_cpu, action_vals.numpy(), label=f"w={w_val}")
        plt.title("Learned STDP Profile by Weight")
        plt.xlabel("Delta t (steps)")
        plt.ylabel("Delta w (Action)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / out_name)
        plt.close()

    _sweep_and_plot(actor_exc, "stdp_sweep_exc.png")
    _sweep_and_plot(actor_inh, "stdp_sweep_inh.png")

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

    result_dir = resolve_path(args.result_dir)
    args.result_dir = str(result_dir)

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

    metrics_path = result_dir / "metrics_train.txt"
    metrics_val = result_dir / "metrics_val.txt"
    metrics_test = result_dir / "metrics_test.txt"
    _ensure_metrics_file(metrics_path)
    _ensure_eval_file(metrics_val)
    _ensure_eval_file(metrics_test)

    estimated_events = args.batch_size_images * args.spike_array_len * (784 + 2 * args.N_E)
    event_buffer = StreamingEventBuffer(
        max_batch_size=args.batch_size_images, max_events_per_image=args.events_per_image, device=device
    )

    s_scen = 1.0
    pos_reward = torch.tensor(1.0, device=device)
    neg_reward = torch.tensor(-1.0, device=device)
    saved_batch_exc: Optional[Tuple[torch.Tensor, ...]] = None
    saved_batch_inh: Optional[Tuple[torch.Tensor, ...]] = None
    for epoch in range(1, args.num_epochs + 1):
        total_sparse = torch.zeros((), device=device)
        total_div = torch.zeros((), device=device)
        total_stab = torch.zeros((), device=device)
        total_reward_sum = torch.zeros((), device=device)
        total_samples = torch.zeros((), device=device)
        saved_batch_exc = None
        saved_batch_inh = None
        for batch_idx, (images, _, indices) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)

            # 수정 후: SNN 실행 부분만 no_grad로 감쌉니다.
            with torch.no_grad():
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
                max_events_per_image=args.events_per_image,
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
                max_events_per_image=args.events_per_image,
                padded_pre=_get_padded(inh_spikes),
                padded_post=_get_padded(exc_spikes),
            )

            batch_size = winners.numel()
            sample_increment = winners.new_full((), batch_size, dtype=torch.float32)
            total_samples = total_samples + sample_increment.detach()
            total_sparse = total_sparse + r_sparse.sum().detach()
            total_div = total_div + r_div.sum().detach()
            total_stab = total_stab + r_stab.sum().detach()
            total_reward_sum = total_reward_sum + total_reward.sum().detach()

            rewards_tensor = total_reward.detach()

            # Reward shifting: use the reward observed at batch k to update the
            # actions sampled at batch k-1, preserving causal credit assignment.
            if saved_batch_exc is not None:
                (
                    prev_states_exc,
                    prev_actions_exc,
                    prev_logp_exc,
                    prev_values_exc,
                    prev_extras_exc,
                    prev_batch_exc_events,
                ) = saved_batch_exc
                returns_exc = rewards_tensor[prev_batch_exc_events]
                adv_exc = returns_exc - prev_values_exc

                ppo_update_events(
                    actor_exc,
                    critic_exc,
                    prev_states_exc,
                    prev_extras_exc,
                    prev_actions_exc,
                    prev_logp_exc,
                    returns_exc.detach(),
                    adv_exc.detach(),
                    optimizer_actor_exc,
                    optimizer_critic_exc,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, prev_states_exc.size(0)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

            if saved_batch_inh is not None:
                (
                    prev_states_inh,
                    prev_actions_inh,
                    prev_logp_inh,
                    prev_values_inh,
                    prev_extras_inh,
                    prev_batch_inh_events,
                ) = saved_batch_inh
                returns_inh = rewards_tensor[prev_batch_inh_events]
                adv_inh = returns_inh - prev_values_inh

                ppo_update_events(
                    actor_inh,
                    critic_inh,
                    prev_states_inh,
                    prev_extras_inh,
                    prev_actions_inh,
                    prev_logp_inh,
                    returns_inh.detach(),
                    adv_inh.detach(),
                    optimizer_actor_inh,
                    optimizer_critic_inh,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, prev_states_inh.size(0)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

            saved_batch_exc = None
            saved_batch_inh = None

            if len(event_buffer) > 0:
                states, extras, connection_ids, pre_idx, post_idx, batch_idx_events = event_buffer.flatten()

                # Excitatory pathway
                exc_mask = connection_ids == 0
                if exc_mask.any():
                    states_exc = states[exc_mask]
                    extras_exc = extras[exc_mask]
                    batch_exc_events = batch_idx_events[exc_mask]
                    pre_exc_events = pre_idx[exc_mask]
                    post_exc_events = post_idx[exc_mask]
                    actions_exc, logp_exc, values_exc = _forward_in_event_batches(
                        actor_exc,
                        critic_exc,
                        states_exc,
                        extras_exc,
                        batch_size=args.event_batch_size,
                    )

                    saved_batch_exc = (
                        states_exc.detach(),
                        actions_exc.detach(),
                        logp_exc.detach(),
                        values_exc.detach(),
                        extras_exc.detach() if extras_exc.numel() > 0 else extras_exc,
                        batch_exc_events.detach(),
                    )

                    with torch.no_grad():
                        delta_exc = args.local_lr * s_scen * actions_exc.detach()
                        _scatter_updates(delta_exc, pre_exc_events, post_exc_events, network.w_input_exc)
                        network.w_input_exc.clamp_(args.exc_clip_min, args.exc_clip_max)
                else:
                    saved_batch_exc = None

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

                    saved_batch_inh = (
                        states_inh.detach(),
                        actions_inh.detach(),
                        logp_inh.detach(),
                        values_inh.detach(),
                        extras_inh.detach() if extras_inh.numel() > 0 else extras_inh,
                        batch_inh_events.detach(),
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
                else:
                    saved_batch_inh = None

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
        val_acc, _ = evaluate_labeling(
            val_rates, val_labels, neuron_labels, 10, result_dir / f"val_confusion_epoch{epoch}"
        )
        test_rates, test_labels = _collect_firing_rates(network, test_loader, device, args)
        test_acc, _ = evaluate_labeling(
            test_rates, test_labels, neuron_labels, 10, result_dir / f"test_confusion_epoch{epoch}"
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

    plot_weight_histograms(w_input_exc_before, network.w_input_exc.detach().cpu(), result_dir / "hist_input_exc.png")
    plot_weight_histograms(w_inh_exc_before, network.w_inh_exc.detach().cpu(), result_dir / "hist_inh_exc.png")

    analyze_stdp_profile(
        network, actor_exc, actor_inh, critic_exc, critic_inh, args, device
    )

    plot_receptive_fields(network.w_input_exc.detach().cpu(), result_dir / "receptive_fields_exc.png")
