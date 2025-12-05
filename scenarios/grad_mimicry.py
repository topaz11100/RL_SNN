import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from data.mnist import get_mnist_dataloaders
from rl.buffers import EventBatchBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update_events
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.lif import LIFParams
from snn.network_grad_mimicry import GradMimicryNetwork
from utils.event_utils import gather_events
from utils.metrics import plot_delta_t_delta_d, plot_grad_alignment, plot_weight_histograms


def _ensure_metrics_file(path: str, header: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header + "\n")


def _scatter_updates(delta: torch.Tensor, pre_idx: torch.Tensor, post_idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    delta_matrix = torch.zeros_like(weights)
    delta_matrix.index_put_((pre_idx, post_idx), delta, accumulate=True)
    return delta_matrix


def _evaluate(network: GradMimicryNetwork, loader, device, args) -> Tuple[float, float]:
    network.eval()
    accuracies, rewards = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            spikes = poisson_encode(images, args.T_sup, max_rate=args.max_rate)
            _, _, firing_rates = network(spikes)
            preds = firing_rates.argmax(dim=1)
            accuracies.append((preds == labels).float().mean().item())
            true_rates = firing_rates.gather(1, labels.view(-1, 1)).squeeze(1)
            masked_rates = firing_rates.clone()
            masked_rates.scatter_(1, labels.view(-1, 1), -1e9)
            max_other, _ = masked_rates.max(dim=1)
            margin = true_rates - max_other
            rewards.append(margin.mean().item())
    network.train()
    return (
        sum(accuracies) / len(accuracies) if accuracies else 0.0,
        sum(rewards) / len(rewards) if rewards else 0.0,
    )


def _layer_indices(num_layers: int, scale: float) -> List[float]:
    if num_layers == 1:
        return [scale]
    return [(i + 1) / num_layers * scale for i in range(num_layers)]


def _extract_delta_t(states: torch.Tensor) -> torch.Tensor:
    if states.numel() == 0:
        return torch.empty(0, device=states.device)
    L = states.size(2)
    time_idx = torch.arange(L, device=states.device)
    last_pre = torch.where(states[:, 0, :] > 0, time_idx, torch.full_like(time_idx, -1)).max(dim=1).values
    last_post = torch.where(states[:, 1, :] > 0, time_idx, torch.full_like(time_idx, -1)).max(dim=1).values
    return last_pre - last_post


def run_grad(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(args.batch_size_images, args.seed)

    lif_params = LIFParams(dt=args.dt)
    network = GradMimicryNetwork(hidden_params=lif_params, output_params=lif_params).to(device)
    teacher = GradMimicryNetwork(hidden_params=lif_params, output_params=lif_params).to(device)
    actor = GaussianPolicy(sigma=getattr(args, "sigma_sup", args.sigma_unsup1), extra_feature_dim=4).to(device)
    critic = ValueFunction(extra_feature_dim=4).to(device)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    layer_norms = _layer_indices(len(network.w_layers), args.layer_index_scale)
    param_names = [name for name, _ in teacher.named_parameters()]

    metrics_train = os.path.join(args.result_dir, "metrics_train.txt")
    metrics_val = os.path.join(args.result_dir, "metrics_val.txt")
    metrics_test = os.path.join(args.result_dir, "metrics_test.txt")
    _ensure_metrics_file(metrics_train, "epoch\tacc\treward\talign")
    _ensure_metrics_file(metrics_val, "epoch\tacc\treward\talign")
    _ensure_metrics_file(metrics_test, "epoch\tacc\treward\talign")

    agent_deltas_log = []
    teacher_deltas_log = []
    delta_t_values = []
    delta_d_values = []

    weights_before = [w.detach().cpu().clone() for w in network.w_layers]

    s_scen = 1.0
    for epoch in range(1, args.num_epochs + 1):
        epoch_acc, epoch_reward, epoch_align = [], [], []
        for batch_idx, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            input_spikes = poisson_encode(images, args.T_sup, max_rate=args.max_rate)

            hidden_spikes_list, output_spikes, firing_rates = network(input_spikes)

            preds = firing_rates.argmax(dim=1)
            batch_acc_tensor = (preds == labels).float().mean()
            epoch_acc.append(batch_acc_tensor.detach())

            event_buffer = EventBatchBuffer()

            padded_cache = {}

            def _get_padded(spikes: torch.Tensor) -> torch.Tensor:
                """Pad once per spike tensor to avoid redundant F.pad calls across layers."""
                key = id(spikes)
                if key not in padded_cache:
                    padded_cache[key] = F.pad(spikes, (args.spike_array_len - 1, 0))
                return padded_cache[key]

            prev_spikes = input_spikes
            for li, hidden_spikes in enumerate(hidden_spikes_list):
                events = gather_events(
                    prev_spikes,
                    hidden_spikes,
                    network.w_layers[li],
                    args.spike_array_len,
                    l_norm=layer_norms[li],
                    padded_pre=_get_padded(prev_spikes),
                    padded_post=_get_padded(hidden_spikes),
                )
                if events[0].numel() > 0:
                    batch_idx = events[4]
                    event_buffer.add(batch_idx, li, events[0], events[1], events[2], events[3], batch_idx)
                prev_spikes = hidden_spikes

            events_out = gather_events(
                prev_spikes,
                output_spikes,
                network.w_layers[-1],
                args.spike_array_len,
                l_norm=layer_norms[-1],
                padded_pre=_get_padded(prev_spikes),
                padded_post=_get_padded(output_spikes),
            )
            if events_out[0].numel() > 0:
                batch_idx = events_out[4]
                event_buffer.add(batch_idx, len(network.w_layers) - 1, events_out[0], events_out[1], events_out[2], events_out[3], batch_idx)

            rewards = torch.zeros(input_spikes.size(0), device=device)

            if len(event_buffer) > 0:
                states, extras, _, connection_ids, pre_idx, post_idx, batch_idx_events = event_buffer.flatten()
                actions, log_probs_old, _ = actor(states, extras)
                values_old = critic(states, extras)

                delta = args.local_lr * s_scen * actions.detach()

                num_layers = len(network.w_layers)
                batch_size = input_spikes.size(0)
                agent_deltas = [torch.zeros((batch_size, *w.shape), device=device) for w in network.w_layers]

                for li in range(num_layers):
                    layer_mask = connection_ids == li
                    if layer_mask.any():
                        batch_layer = batch_idx_events[layer_mask]
                        pre_layer = pre_idx[layer_mask]
                        post_layer = post_idx[layer_mask]
                        delta_layer = delta[layer_mask]
                        agent_deltas[li].index_put_((batch_layer, pre_layer, post_layer), delta_layer, accumulate=True)

                teacher.load_state_dict(network.state_dict())
                teacher_params = tuple(param.detach().requires_grad_(True) for param in teacher.parameters())

                def loss_fn(params, spikes, label):
                    params_dict = {name: p for name, p in zip(param_names, params)}
                    _, _, firing_teacher = functional_call(teacher, params_dict, (spikes.unsqueeze(0),))
                    logits = firing_teacher * 5.0
                    return F.cross_entropy(logits, label.unsqueeze(0))

                grad_fn = grad(loss_fn)
                per_sample_grads = vmap(grad_fn, in_dims=(None, 0, 0))(teacher_params, input_spikes, labels)
                teacher_deltas = [-args.alpha_align * g for g in per_sample_grads]

                squared_error_sum = torch.zeros(batch_size, device=device)
                active_count = torch.zeros(batch_size, device=device)

                for li in range(num_layers):
                    mask = agent_deltas[li] != 0
                    diff = agent_deltas[li] - teacher_deltas[li]
                    squared_error_sum = squared_error_sum + (diff.pow(2) * mask).sum(dim=(1, 2))
                    active_count = active_count + mask.sum(dim=(1, 2))

                align_loss = torch.where(
                    active_count > 0,
                    squared_error_sum / active_count.clamp_min(1).float(),
                    torch.zeros_like(squared_error_sum),
                )
                rewards = -align_loss

                returns = rewards.detach()[batch_idx_events]
                advantages = returns - values_old.detach()

                full_event_batch = states.size(0)
                ppo_mini_batch = min(args.ppo_batch_size, full_event_batch)
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
                    batch_size=ppo_mini_batch,  # Respect Theory/CLI mini-batch guidance
                    eps_clip=args.ppo_eps,
                    c_v=1.0,
                )

                with torch.no_grad():
                    for li in range(num_layers):
                        update_mat = agent_deltas[li].sum(dim=0)
                        network.w_layers[li].add_(update_mat)
                        network.w_layers[li].clamp_(args.exc_clip_min, args.exc_clip_max)

                for li in range(num_layers):
                    agent_deltas_log.append(agent_deltas[li].sum(dim=0).detach())
                    teacher_deltas_log.append(teacher_deltas[li].sum(dim=0).detach())

                delta_t_values.append(_extract_delta_t(states).detach())
                delta_d_values.append(actions.detach())

                epoch_reward.append(rewards.detach())
                epoch_align.append(rewards.detach())
            else:
                zero_reward = torch.zeros(input_spikes.size(0), device=device)
                epoch_reward.append(zero_reward.detach())
                epoch_align.append(zero_reward.detach())

        mean_acc = torch.stack(epoch_acc).mean().item() if epoch_acc else 0.0
        reward_tensor = torch.cat(epoch_reward) if epoch_reward else torch.empty(0, device=device)
        align_tensor = torch.cat(epoch_align) if epoch_align else torch.empty(0, device=device)
        mean_reward = reward_tensor.mean().item() if reward_tensor.numel() > 0 else 0.0
        mean_align = align_tensor.mean().item() if align_tensor.numel() > 0 else 0.0

        val_acc, val_reward = _evaluate(network, val_loader, device, args)
        test_acc, test_reward = _evaluate(network, test_loader, device, args)

        with open(metrics_train, "a") as f:
            f.write(f"{epoch}\t{mean_acc:.6f}\t{mean_reward:.6f}\t{mean_align:.6f}\n")
        with open(metrics_val, "a") as f:
            f.write(f"{epoch}\t{val_acc:.6f}\t{val_reward:.6f}\t0.000000\n")
        with open(metrics_test, "a") as f:
            f.write(f"{epoch}\t{test_acc:.6f}\t{test_reward:.6f}\t0.000000\n")

        if epoch % args.log_interval == 0:
            logger.info(
                "Epoch %d | Train acc %.4f reward %.4f align %.4f | Val acc %.4f | Test acc %.4f",
                epoch,
                mean_acc,
                mean_reward,
                mean_align,
                val_acc,
                test_acc,
            )
            if args.log_gradient_stats:
                logger.info("Gradient alignment mean (train): %.4f", mean_align)

    delta_t_concat = torch.cat(delta_t_values, dim=0).cpu() if delta_t_values else torch.empty(0)
    delta_d_concat = torch.cat(delta_d_values, dim=0).cpu() if delta_d_values else torch.empty(0)
    if delta_t_concat.numel() > 0 and delta_d_concat.numel() > 0:
        plot_delta_t_delta_d(delta_t_concat, delta_d_concat, os.path.join(args.result_dir, "delta_t_delta_d.png"))

    weights_after = [w.detach().cpu().clone() for w in network.w_layers]
    for i, (w_before, w_after) in enumerate(zip(weights_before, weights_after)):
        plot_weight_histograms(
            w_before, w_after, os.path.join(args.result_dir, f"hist_layer{i}.png")
        )

    if agent_deltas_log and teacher_deltas_log:
        agent_cat = torch.cat(agent_deltas_log).cpu()
        teacher_cat = torch.cat(teacher_deltas_log).cpu()
        plot_grad_alignment(agent_cat, teacher_cat, os.path.join(args.result_dir, "grad_alignment.png"))
