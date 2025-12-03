import os
from typing import List, Tuple

import torch
import torch.nn.functional as F

from analysis_utils import plot_grad_alignment
from data.mnist import get_mnist_dataloaders
from rl.buffers import EpisodeBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.lif import LIFParams
from snn.network_grad_mimicry import GradMimicryNetwork


def _ensure_metrics_file(path: str, header: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header + "\n")


def _gather_events(
    pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor, L: int, l_norm: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pre_spikes.device
    batch_size, n_pre, T = pre_spikes.shape
    n_post = post_spikes.shape[1]
    pad_pre = F.pad(pre_spikes, (L - 1, 0))
    pad_post = F.pad(post_spikes, (L - 1, 0))
    idx_range = torch.arange(L, device=device)

    histories, extras, pre_indices, post_indices = [], [], [], []

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
        pos = time_idx + (L - 1)
        time_indices = pos.unsqueeze(1) - idx_range.view(1, -1)
        pre_hist = pad_pre[batch_idx.unsqueeze(1), pre_idx.unsqueeze(1), time_indices]
        post_hist = pad_post[batch_idx.unsqueeze(1), post_idx.unsqueeze(1), time_indices]
        histories.append(torch.stack([pre_hist, post_hist], dim=1))
        w_vals = weights[pre_idx, post_idx].unsqueeze(1)
        l_norm_tensor = torch.full_like(w_vals, l_norm)
        extras.append(torch.cat([w_vals, l_norm_tensor, e_type.expand(w_vals.size(0), -1)], dim=1))
        pre_indices.append(pre_idx)
        post_indices.append(post_idx)

    _append_events(pre_spikes, True)
    _append_events(post_spikes, False)

    if not histories:
        return (
            torch.empty(0, 2, L, device=device),
            torch.empty(0, 4, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    return (
        torch.cat(histories, dim=0),
        torch.cat(extras, dim=0),
        torch.cat(pre_indices, dim=0),
        torch.cat(post_indices, dim=0),
    )


def _scatter_updates(delta: torch.Tensor, pre_idx: torch.Tensor, post_idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    delta_matrix = torch.zeros_like(weights)
    delta_matrix.index_put_((pre_idx, post_idx), delta, accumulate=True)
    weights.data.add_(delta_matrix)
    return delta_matrix


def _evaluate(network: GradMimicryNetwork, loader, device, args) -> Tuple[float, float]:
    network.eval()
    accuracies, rewards = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            spikes = poisson_encode(images, args.T_sup, max_rate=args.max_rate).to(device)
            _, _, firing_rates = network(spikes)
            preds = firing_rates.argmax(dim=1)
            accuracies.append((preds == labels).float().mean().item())
            true_rates = firing_rates.gather(1, labels.view(-1, 1)).squeeze(1)
            max_other, _ = (firing_rates + torch.eye(firing_rates.size(1), device=firing_rates.device) * -1e9).max(dim=1)
            margin = true_rates - max_other
            rewards.append(margin.mean().item())
    network.train()
    return (
        sum(accuracies) / len(accuracies) if accuracies else 0.0,
        sum(rewards) / len(rewards) if rewards else 0.0,
    )


def _layer_indices(num_layers: int, scale: float) -> List[float]:
    if num_layers == 1:
        return [0.0]
    return [i / (num_layers - 1) * scale for i in range(num_layers)]


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

    metrics_train = os.path.join(args.result_dir, "metrics_train.txt")
    metrics_val = os.path.join(args.result_dir, "metrics_val.txt")
    metrics_test = os.path.join(args.result_dir, "metrics_test.txt")
    _ensure_metrics_file(metrics_train, "epoch\tacc\treward\talign")
    _ensure_metrics_file(metrics_val, "epoch\tacc\treward\talign")
    _ensure_metrics_file(metrics_test, "epoch\tacc\treward\talign")

    agent_deltas_log = []
    teacher_deltas_log = []

    for epoch in range(1, args.num_epochs + 1):
        epoch_acc, epoch_reward, epoch_align = [], [], []
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            input_spikes = poisson_encode(images, args.T_sup, max_rate=args.max_rate).to(device)

            hidden_spikes_list, output_spikes, firing_rates = network(input_spikes)

            preds = firing_rates.argmax(dim=1)
            epoch_acc.append((preds == labels).float().mean().item())

            batch_buffer = EpisodeBuffer()

            for b in range(input_spikes.size(0)):
                states_all = []
                extras_all = []
                event_specs = []

                pre_post_weights = []
                prev_spikes = input_spikes[b : b + 1]
                for li, hidden_spikes in enumerate(hidden_spikes_list):
                    pre_post_weights.append((prev_spikes, hidden_spikes[b : b + 1], network.w_layers[li], layer_norms[li]))
                    prev_spikes = hidden_spikes[b : b + 1]
                pre_post_weights.append((prev_spikes, output_spikes[b : b + 1], network.w_layers[-1], layer_norms[-1]))

                for idx, (pre_s, post_s, weight, l_norm) in enumerate(pre_post_weights):
                    events = _gather_events(pre_s, post_s, weight, args.spike_array_len, l_norm)
                    if events[0].numel() > 0:
                        states_all.append(events[0])
                        extras_all.append(events[1])
                        event_specs.append((idx, events))

                agent_deltas = [torch.zeros_like(w) for w in network.w_layers]

                if states_all:
                    states = torch.cat(states_all, dim=0)
                    extras = torch.cat(extras_all, dim=0)

                    action, log_prob, _ = actor(states, extras)
                    value = critic(states, extras)

                    offset = 0
                    for idx, events in event_specs:
                        count = events[0].size(0)
                        idx_slice = slice(offset, offset + count)
                        delta_mat = _scatter_updates(
                            args.local_lr * action[idx_slice], events[2], events[3], network.w_layers[idx]
                        )
                        agent_deltas[idx] = delta_mat
                        torch.clamp_(network.w_layers[idx], args.exc_clip_min, args.exc_clip_max)
                        offset += count

                    episode_buffer = EpisodeBuffer()
                    for i in range(states.size(0)):
                        episode_buffer.append(states[i], extras[i], action[i], log_prob[i], value[i])
                else:
                    action = torch.empty(0, device=device)
                    log_prob = torch.empty(0, device=device)
                    value = torch.empty(0, device=device)
                    episode_buffer = EpisodeBuffer()

                teacher.load_state_dict(network.state_dict())
                teacher.zero_grad()
                hidden_teacher, output_teacher, firing_teacher = teacher(input_spikes[b : b + 1])
                logits = firing_teacher * 5.0
                loss_sup = F.cross_entropy(logits, labels[b : b + 1])
                loss_sup.backward()

                teacher_deltas = [
                    -args.alpha_align * teacher.w_layers[i].grad for i in range(len(teacher.w_layers))
                ]

                total_synapses = sum(d.numel() for d in teacher_deltas)
                align_loss = sum((agent_deltas[i] - teacher_deltas[i]).pow(2).sum() for i in range(len(agent_deltas)))
                align_loss = align_loss / total_synapses if total_synapses > 0 else torch.tensor(0.0, device=device)
                reward = -align_loss

                epoch_reward.append(reward.item())
                epoch_align.append(reward.item())

                if len(episode_buffer) > 0:
                    episode_buffer.finalize(reward)
                    batch_buffer.extend(episode_buffer)

                for a_d, t_d in zip(agent_deltas, teacher_deltas):
                    agent_deltas_log.append(a_d.detach().cpu().flatten())
                    teacher_deltas_log.append(t_d.detach().cpu().flatten())

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

        mean_acc = sum(epoch_acc) / len(epoch_acc) if epoch_acc else 0.0
        mean_reward = sum(epoch_reward) / len(epoch_reward) if epoch_reward else 0.0
        mean_align = sum(epoch_align) / len(epoch_align) if epoch_align else 0.0

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

    if agent_deltas_log and teacher_deltas_log:
        agent_cat = torch.cat(agent_deltas_log)
        teacher_cat = torch.cat(teacher_deltas_log)
        plot_grad_alignment(agent_cat, teacher_cat, os.path.join(args.result_dir, "grad_alignment.png"))
