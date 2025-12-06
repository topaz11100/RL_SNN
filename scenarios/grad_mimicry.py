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
    flat_index = pre_idx * weights.size(1) + post_idx
    delta_matrix.view(-1).scatter_add_(0, flat_index, delta)
    return delta_matrix


def _evaluate(network: GradMimicryNetwork, loader, device, args) -> Tuple[float, float]:
    network.eval()
    correct = torch.zeros((), device=device)
    reward_sum = torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            spikes = poisson_encode(images, args.T_sup, max_rate=args.max_rate)
            _, _, firing_rates = network(spikes)
            preds = firing_rates.argmax(dim=1)
            correct = correct + (preds == labels).sum()
            true_rates = firing_rates.gather(1, labels.view(-1, 1)).squeeze(1)
            masked_rates = firing_rates.clone()
            masked_rates.scatter_(1, labels.view(-1, 1), -1e9)
            max_other, _ = masked_rates.max(dim=1)
            margin = true_rates - max_other
            reward_sum = reward_sum + margin.sum()
            total = total + labels.numel()
    network.train()
    total_count = total.item()
    if total_count > 0:
        acc = (correct / total).item()
        reward_mean = (reward_sum / total).item()
    else:
        acc = 0.0
        reward_mean = 0.0
    return acc, reward_mean


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


def _forward_in_event_batches(actor, critic, states, extras, batch_size):
    actions_cat, log_probs_cat, values_cat = None, None, None
    extras_available = extras.numel() > 0
    for start in range(0, states.size(0), batch_size):
        end = start + batch_size
        states_mb = states[start:end]
        extras_mb = extras[start:end] if extras_available else None
        action_mb, log_prob_mb, _ = actor(states_mb, extras_mb)
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


def run_grad(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(args.batch_size_images, args.seed)

    lif_params = LIFParams(dt=args.dt)
    network = GradMimicryNetwork(hidden_params=lif_params, output_params=lif_params).to(device)
    actor = GaussianPolicy(sigma=getattr(args, "sigma_sup", args.sigma_unsup1), extra_feature_dim=4).to(device)
    critic = ValueFunction(extra_feature_dim=4).to(device)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    layer_norms = _layer_indices(len(network.w_layers), args.layer_index_scale)
    param_names = [name for name, _ in network.named_parameters()]
    synapse_pre_sum = sum(shape[0] for shape in network.synapse_shapes)
    estimated_events = args.batch_size_images * args.spike_array_len * synapse_pre_sum
    event_buffer = EventBatchBuffer(initial_capacity=max(100_000, estimated_events))

    metrics_train = os.path.join(args.result_dir, "metrics_train.txt")
    metrics_val = os.path.join(args.result_dir, "metrics_val.txt")
    metrics_test = os.path.join(args.result_dir, "metrics_test.txt")
    _ensure_metrics_file(metrics_train, "epoch\tacc\treward\talign\tactive_ratio")
    _ensure_metrics_file(metrics_val, "epoch\tacc\treward\talign")
    _ensure_metrics_file(metrics_test, "epoch\tacc\treward\talign")

    agent_deltas_log = [
        torch.empty((0, *w.shape), device=device, dtype=w.dtype) for w in network.w_layers
    ]
    teacher_deltas_log = [
        torch.empty((0, *w.shape), device=device, dtype=w.dtype) for w in network.w_layers
    ]
    delta_t_values = torch.empty(0, device=device)
    delta_d_values = torch.empty(0, device=device)

    weights_before = [w.detach().cpu().clone() for w in network.w_layers]
    total_synapses = sum(w.numel() for w in network.w_layers)

    s_scen = 1.0
    for epoch in range(1, args.num_epochs + 1):
        total_correct = torch.zeros((), device=device)
        total_reward = torch.zeros((), device=device)
        total_align = torch.zeros((), device=device)
        total_active = torch.zeros((), device=device)
        total_samples = torch.zeros((), device=device)
        active_batches = torch.zeros((), device=device)
        for batch_idx, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            input_spikes = poisson_encode(images, args.T_sup, max_rate=args.max_rate)

            hidden_spikes_list, output_spikes, firing_rates = network(input_spikes)

            preds = firing_rates.argmax(dim=1)
            batch_size = labels.numel()
            sample_increment = torch.tensor(batch_size, device=device, dtype=torch.float32)
            total_samples = total_samples + sample_increment
            total_correct = total_correct + (preds == labels).sum()

            event_buffer.reset()

            padded_cache = {}

            def _get_padded(spikes: torch.Tensor) -> torch.Tensor:
                """Pad once per spike tensor to avoid redundant F.pad calls across layers."""
                key = id(spikes)
                if key not in padded_cache:
                    padded_cache[key] = F.pad(spikes, (args.spike_array_len - 1, 0))
                return padded_cache[key]

            prev_spikes = input_spikes
            for li, hidden_spikes in enumerate(hidden_spikes_list):
                gather_events(
                    prev_spikes,
                    hidden_spikes,
                    network.w_layers[li],
                    args.spike_array_len,
                    event_buffer,
                    li,
                    l_norm=layer_norms[li],
                    padded_pre=_get_padded(prev_spikes),
                    padded_post=_get_padded(hidden_spikes),
                )
                prev_spikes = hidden_spikes

            gather_events(
                prev_spikes,
                output_spikes,
                network.w_layers[-1],
                args.spike_array_len,
                event_buffer,
                len(network.w_layers) - 1,
                l_norm=layer_norms[-1],
                padded_pre=_get_padded(prev_spikes),
                padded_post=_get_padded(output_spikes),
            )

            rewards = torch.zeros(input_spikes.size(0), device=device)

            if len(event_buffer) > 0:
                states, extras, connection_ids, pre_idx, post_idx, batch_idx_events = event_buffer.flatten()
                actions, log_probs_old, values_old = _forward_in_event_batches(
                    actor, critic, states, extras, batch_size=args.event_batch_size
                )

                delta = args.local_lr * s_scen * actions.detach()

                num_layers = len(network.w_layers)
                agent_deltas = [
                    torch.zeros((batch_size, *w.shape), device=device, dtype=w.dtype)
                    for w in network.w_layers
                ]

                for li in range(num_layers):
                    layer_mask = connection_ids == li
                    if layer_mask.any():
                        batch_layer = batch_idx_events[layer_mask]
                        pre_layer = pre_idx[layer_mask]
                        post_layer = post_idx[layer_mask]
                        delta_layer = delta[layer_mask]
                        agent_deltas[li].index_put_((batch_layer, pre_layer, post_layer), delta_layer, accumulate=True)

                teacher_params = tuple(param.detach().requires_grad_(True) for param in network.parameters())

                def loss_fn(params, spikes, label):
                    params_dict = {name: p for name, p in zip(param_names, params)}
                    _, _, firing_teacher = functional_call(network, params_dict, (spikes.unsqueeze(0),))
                    logits = firing_teacher * 5.0
                    return F.cross_entropy(logits, label.unsqueeze(0))

                grad_fn = grad(loss_fn)
                batch_size = input_spikes.size(0)
                chunk_size = getattr(args, "grad_chunk_size", 0)
                if chunk_size <= 0:
                    chunk_size = min(64, batch_size)

                if chunk_size < batch_size:
                    grad_chunks = None
                    for start in range(0, input_spikes.size(0), chunk_size):
                        end = start + chunk_size
                        grads_chunk = vmap(grad_fn, in_dims=(None, 0, 0))(teacher_params, input_spikes[start:end], labels[start:end])
                        if grad_chunks is None:
                            grad_chunks = grads_chunk
                        else:
                            grad_chunks = tuple(
                                torch.cat((acc, gc), dim=0) for acc, gc in zip(grad_chunks, grads_chunk)
                            )
                    per_sample_grads = grad_chunks if grad_chunks is not None else tuple()
                else:
                    per_sample_grads = vmap(grad_fn, in_dims=(None, 0, 0))(teacher_params, input_spikes, labels)
                teacher_deltas = [-args.alpha_align * g for g in per_sample_grads]

                acc_dtype = agent_deltas[0].dtype
                squared_error_sum = torch.zeros(batch_size, device=device, dtype=acc_dtype)
                active_count = torch.zeros(batch_size, device=device, dtype=acc_dtype)
                active_ratio = torch.zeros((), device=device, dtype=torch.float32)

                for li in range(num_layers):
                    mask = agent_deltas[li].ne(0).to(acc_dtype)
                    diff = agent_deltas[li] - teacher_deltas[li]
                    squared_error_sum = squared_error_sum + (diff.pow(2) * mask).sum(dim=(1, 2))
                    active_count = active_count + mask.sum(dim=(1, 2))

                if total_synapses > 0:
                    active_ratio = active_count.sum().float() / (batch_size * float(total_synapses))

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
                    agent_deltas_log[li] = torch.cat(
                        (agent_deltas_log[li], agent_deltas[li].sum(dim=0, keepdim=True)), dim=0
                    )
                    teacher_deltas_log[li] = torch.cat(
                        (teacher_deltas_log[li], teacher_deltas[li].sum(dim=0, keepdim=True)), dim=0
                    )

                delta_t_values = torch.cat((delta_t_values, _extract_delta_t(states)), dim=0)
                delta_d_values = torch.cat((delta_d_values, actions.detach()), dim=0)

                total_reward = total_reward + rewards.sum()
                total_align = total_align + rewards.sum()
                total_active = total_active + active_ratio
                active_batches = active_batches + torch.tensor(1.0, device=device)
            else:
                zero_reward_sum = torch.zeros((), device=device)
                total_reward = total_reward + zero_reward_sum
                total_align = total_align + zero_reward_sum
                total_active = total_active + torch.zeros((), device=device)
                active_batches = active_batches + torch.tensor(1.0, device=device)

        mean_acc = (total_correct / total_samples).item() if total_samples.item() > 0 else 0.0
        mean_reward = (total_reward / total_samples).item() if total_samples.item() > 0 else 0.0
        mean_align = (total_align / total_samples).item() if total_samples.item() > 0 else 0.0
        mean_active = (total_active / active_batches).item() if active_batches.item() > 0 else 0.0

        val_acc, val_reward = _evaluate(network, val_loader, device, args)
        test_acc, test_reward = _evaluate(network, test_loader, device, args)

        with open(metrics_train, "a") as f:
            f.write(f"{epoch}\t{mean_acc:.6f}\t{mean_reward:.6f}\t{mean_align:.6f}\t{mean_active:.6f}\n")
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

    delta_t_concat = delta_t_values.cpu()
    delta_d_concat = delta_d_values.cpu()
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
