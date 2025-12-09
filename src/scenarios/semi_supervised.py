from pathlib import Path
from typing import Tuple

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
from snn.network_semi_supervised import SemiSupervisedNetwork
from utils.event_utils import gather_events
from utils.logging import resolve_path
from utils.metrics import plot_weight_histograms


def _ensure_metrics_file(path: str | Path, header: str) -> None:
    resolved = resolve_path(path)
    if not resolved.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("w") as f:
            f.write(header + "\n")


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


def _scatter_updates(delta: torch.Tensor, pre_idx: torch.Tensor, post_idx: torch.Tensor, weights: torch.Tensor) -> None:
    if delta.numel() == 0:
        return
    weights.index_put_((pre_idx, post_idx), delta, accumulate=True)


def _compute_reward_components(
    firing_rates: torch.Tensor, labels: torch.Tensor, beta_margin: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = firing_rates.argmax(dim=1)
    correct = preds == labels
    ones = torch.ones_like(firing_rates[:, 0])
    r_cls = torch.where(correct, ones, -ones)
    true_rates = firing_rates.gather(1, labels.view(-1, 1)).squeeze(1)
    masked_rates = firing_rates.clone()
    masked_rates.scatter_(1, labels.view(-1, 1), -1e9)
    max_other, _ = masked_rates.max(dim=1)
    margin = true_rates - max_other
    r_margin = beta_margin * margin
    total = r_cls + r_margin
    return r_cls, r_margin, total


def _evaluate(network: SemiSupervisedNetwork, loader, device, args) -> Tuple[float, float, float]:
    network.eval()
    correct = torch.zeros((), device=device)
    margin_sum = torch.zeros((), device=device)
    reward_sum = torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            spikes = poisson_encode(images, args.T_semi, max_rate=args.max_rate)
            _, output_spikes, rates = network(spikes)
            r_cls, r_margin, r_total = _compute_reward_components(rates, labels, args.beta_margin)
            preds = rates.argmax(dim=1)
            correct = correct + (preds == labels).sum()
            margin_sum = margin_sum + r_margin.sum()
            reward_sum = reward_sum + r_total.sum()
            total = total + labels.numel()
    network.train()
    total_count = total.item()
    if total_count > 0:
        return (
            (correct / total).item(),
            (margin_sum / total).item(),
            (reward_sum / total).item(),
        )
    return 0.0, 0.0, 0.0


def _extract_delta_t(states: torch.Tensor) -> torch.Tensor:
    if states.numel() == 0:
        return torch.empty(0, device=states.device)
    L = states.size(2)
    time_idx = torch.arange(L, device=states.device)
    last_pre = torch.where(states[:, 0, :] > 0, time_idx, torch.full_like(time_idx, -1)).max(dim=1).values
    last_post = torch.where(states[:, 1, :] > 0, time_idx, torch.full_like(time_idx, -1)).max(dim=1).values
    return last_pre - last_post


def analyze_stdp_profile(
    network: SemiSupervisedNetwork,
    actor: GaussianPolicy,
    critic: ValueFunction,
    args,
    device: torch.device,
) -> None:
    result_dir = resolve_path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    network_state = network.training
    actor_state = actor.training
    critic_state = critic.training
    network.eval()
    actor.eval()
    critic.eval()

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
    curves: list[tuple[float, torch.Tensor]] = []

    with torch.no_grad():
        for w_val in weights_to_test:
            extras = torch.zeros((len(dt_range), actor.extra_feature_dim), device=device)
            extras[:, 0] = w_val
            actions, _, _ = actor(states, extras)
            
            # 텐서 변환
            action_vals = actions.detach().cpu().view(-1)
            curves.append((w_val, action_vals))

            # [추가됨] w별 개별 그래프 즉시 저장
            plt.figure(figsize=(6, 5), dpi=100)
            plt.plot(dt_cpu, action_vals.numpy(), color='blue', label=f"w={w_val}")
            plt.title(f"STDP Profile (w={w_val})")
            plt.xlabel("Delta t (steps)")
            plt.ylabel("Delta w (Action)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(result_dir / f"stdp_profile_w_{w_val}.png") # 파일명 예: stdp_profile_w_-0.5.png
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
    plt.savefig(result_dir / "stdp_profile_sweep.png")
    plt.close()

    if network_state:
        network.train()
    if actor_state:
        actor.train()
    if critic_state:
        critic.train()


def run_semi(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(args.batch_size_images, args.seed)

    result_dir = resolve_path(args.result_dir)
    args.result_dir = str(result_dir)

    lif_params = LIFParams(dt=args.dt)
    network = SemiSupervisedNetwork(
        n_hidden=args.N_hidden, hidden_params=lif_params, output_params=lif_params
    ).to(device)
    actor = GaussianPolicy(sigma=getattr(args, "sigma_semi", args.sigma_unsup1), extra_feature_dim=3).to(device)
    critic = ValueFunction(extra_feature_dim=3).to(device)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    w_input_hidden_before = network.w_input_hidden.detach().cpu().clone()
    w_hidden_output_before = network.w_hidden_output.detach().cpu().clone()

    metrics_train = result_dir / "metrics_train.txt"
    metrics_val = result_dir / "metrics_val.txt"
    metrics_test = result_dir / "metrics_test.txt"
    _ensure_metrics_file(metrics_train, "epoch\tacc\tmargin\treward")
    _ensure_metrics_file(metrics_val, "epoch\tacc\tmargin\treward")
    _ensure_metrics_file(metrics_test, "epoch\tacc\tmargin\treward")

    estimated_events = args.batch_size_images * args.spike_array_len * (784 + args.N_hidden + 10)
    event_buffer = StreamingEventBuffer(
        max_batch_size=args.batch_size_images, max_events_per_image=args.events_per_image, device=device
    )

    s_scen = 1.0
    w_clip_min = args.w_clip_min
    w_clip_max = args.w_clip_max
    saved_batch: Optional[Tuple[torch.Tensor, ...]] = None
    for epoch in range(1, args.num_epochs + 1):
        total_correct = torch.zeros((), device=device)
        total_margin = torch.zeros((), device=device)
        total_reward = torch.zeros((), device=device)
        total_samples = torch.zeros((), device=device)
        saved_batch = None
        for batch_idx, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            input_spikes = poisson_encode(images, args.T_semi, max_rate=args.max_rate)
            hidden_spikes, output_spikes, firing_rates = network(input_spikes)

            preds = firing_rates.argmax(dim=1)
            batch_size = labels.numel()
            sample_increment = labels.new_full((), batch_size, dtype=torch.float32)
            total_samples = total_samples + sample_increment.detach()
            total_correct = total_correct + (preds == labels).sum().detach()

            r_cls, r_margin, r_total = _compute_reward_components(firing_rates, labels, args.beta_margin)

            event_buffer.reset()

            padded_cache = {}

            def _get_padded(spikes: torch.Tensor) -> torch.Tensor:
                key = id(spikes)
                if key not in padded_cache:
                    padded_cache[key] = F.pad(spikes, (args.spike_array_len - 1, 0))
                return padded_cache[key]
            gather_events(
                input_spikes,
                hidden_spikes,
                network.w_input_hidden,
                args.spike_array_len,
                event_buffer,
                0,
                max_events_per_image=args.events_per_image,
                padded_pre=_get_padded(input_spikes),
                padded_post=_get_padded(hidden_spikes),
            )
            gather_events(
                hidden_spikes,
                output_spikes,
                network.w_hidden_output,
                args.spike_array_len,
                event_buffer,
                1,
                max_events_per_image=args.events_per_image,
                padded_pre=_get_padded(hidden_spikes),
                padded_post=_get_padded(output_spikes),
            )

            rewards_tensor = r_total.detach()

            # Reward shifting: use reward R_k to update actions sampled at batch k-1.
            if saved_batch is not None:
                (
                    prev_states,
                    prev_actions,
                    prev_log_probs,
                    prev_values,
                    prev_extras,
                    prev_batch_indices,
                ) = saved_batch
                returns = rewards_tensor[prev_batch_indices]
                advantages = returns - prev_values

                ppo_update_events(
                    actor,
                    critic,
                    prev_states,
                    prev_extras,
                    prev_actions,
                    prev_log_probs,
                    returns.detach(),
                    advantages.detach(),
                    optimizer_actor,
                    optimizer_critic,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, prev_states.size(0)),
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

            saved_batch = None

            if len(event_buffer) > 0:
                states, extras, connection_ids, pre_idx, post_idx, batch_indices = event_buffer.flatten()
                actions, log_probs_old, values_old = _forward_in_event_batches(
                    actor, critic, states, extras, batch_size=args.event_batch_size
                )

                saved_batch = (
                    states.detach(),
                    actions.detach(),
                    log_probs_old.detach(),
                    values_old.detach(),
                    extras.detach() if extras.numel() > 0 else extras,
                    batch_indices.detach(),
                )

                with torch.no_grad():
                    delta = args.local_lr * s_scen * actions.detach()
                    in_mask = connection_ids == 0
                    if in_mask.any():
                        _scatter_updates(delta[in_mask], pre_idx[in_mask], post_idx[in_mask], network.w_input_hidden)
                        network.w_input_hidden.clamp_(w_clip_min, w_clip_max)
                    out_mask = connection_ids == 1
                    if out_mask.any():
                        _scatter_updates(delta[out_mask], pre_idx[out_mask], post_idx[out_mask], network.w_hidden_output)
                        network.w_hidden_output.clamp_(w_clip_min, w_clip_max)

            total_margin = total_margin + r_margin.sum().detach()
            total_reward = total_reward + r_total.sum().detach()

            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                batch_acc = (preds == labels).float().mean().item()
                batch_rew = r_total.mean().item()
                logger.info(
                    "Epoch %d | Batch %d/%d | Acc %.4f | Reward %.4f",
                    epoch, batch_idx, len(train_loader), batch_acc, batch_rew
                )

        mean_acc = (total_correct / total_samples).item() if total_samples.item() > 0 else 0.0
        mean_margin = (total_margin / total_samples).item() if total_samples.item() > 0 else 0.0
        mean_reward = (total_reward / total_samples).item() if total_samples.item() > 0 else 0.0

        val_acc, val_margin, val_reward = _evaluate(network, val_loader, device, args)
        test_acc, test_margin, test_reward = _evaluate(network, test_loader, device, args)

        with open(metrics_train, "a") as f:
            f.write(f"{epoch}\t{mean_acc:.6f}\t{mean_margin:.6f}\t{mean_reward:.6f}\n")
        with open(metrics_val, "a") as f:
            f.write(f"{epoch}\t{val_acc:.6f}\t{val_margin:.6f}\t{val_reward:.6f}\n")
        with open(metrics_test, "a") as f:
            f.write(f"{epoch}\t{test_acc:.6f}\t{test_margin:.6f}\t{test_reward:.6f}\n")

        if epoch % args.log_interval == 0:
            logger.info(
                "Epoch %d | Train acc %.4f margin %.4f reward %.4f | Val acc %.4f | Test acc %.4f",
                epoch,
                mean_acc,
                mean_margin,
                mean_reward,
                val_acc,
                test_acc,
            )

    plot_weight_histograms(w_input_hidden_before, network.w_input_hidden.detach().cpu(), result_dir / "hist_input_hidden.png")
    plot_weight_histograms(w_hidden_output_before, network.w_hidden_output.detach().cpu(), result_dir / "hist_hidden_output.png")

    analyze_stdp_profile(network, actor, critic, args, device)
