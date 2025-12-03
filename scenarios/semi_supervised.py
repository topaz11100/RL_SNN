import os
from typing import Tuple

import torch
import torch.nn.functional as F

from analysis_utils import plot_delta_t_delta_d, plot_weight_histograms

from data.mnist import get_mnist_dataloaders
from rl.buffers import EpisodeBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.lif import LIFParams
from snn.network_semi_supervised import SemiSupervisedNetwork


def _ensure_metrics_file(path: str, header: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header + "\n")


def _gather_events(
    pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor, L: int
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


def _scatter_updates(delta: torch.Tensor, pre_idx: torch.Tensor, post_idx: torch.Tensor, weights: torch.Tensor) -> None:
    delta_matrix = torch.zeros_like(weights)
    delta_matrix.index_put_((pre_idx, post_idx), delta, accumulate=True)
    weights.data.add_(delta_matrix)


def _compute_reward_components(
    firing_rates: torch.Tensor, labels: torch.Tensor, beta_margin: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = firing_rates.argmax(dim=1)
    correct = preds == labels
    r_cls = torch.where(
        correct, torch.tensor(1.0, device=firing_rates.device), torch.tensor(-1.0, device=firing_rates.device)
    )
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
    accuracies, margins, rewards = [], [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            spikes = poisson_encode(images, args.T_semi, max_rate=args.max_rate).to(device)
            _, output_spikes, rates = network(spikes)
            r_cls, r_margin, r_total = _compute_reward_components(rates, labels, args.beta_margin)
            preds = rates.argmax(dim=1)
            accuracies.append((preds == labels).float().mean().item())
            margins.append(r_margin.mean().item())
            rewards.append(r_total.mean().item())
    network.train()
    return (
        sum(accuracies) / len(accuracies) if accuracies else 0.0,
        sum(margins) / len(margins) if margins else 0.0,
        sum(rewards) / len(rewards) if rewards else 0.0,
    )


def _extract_delta_t(states: torch.Tensor) -> torch.Tensor:
    if states.numel() == 0:
        return torch.empty(0, device=states.device)
    L = states.size(2)
    time_idx = torch.arange(L, device=states.device)
    last_pre = torch.where(states[:, 0, :] > 0, time_idx, torch.full_like(time_idx, -1)).max(dim=1).values
    last_post = torch.where(states[:, 1, :] > 0, time_idx, torch.full_like(time_idx, -1)).max(dim=1).values
    return last_pre - last_post


def run_semi(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(args.batch_size_images, args.seed)

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

    metrics_train = os.path.join(args.result_dir, "metrics_train.txt")
    metrics_val = os.path.join(args.result_dir, "metrics_val.txt")
    metrics_test = os.path.join(args.result_dir, "metrics_test.txt")
    _ensure_metrics_file(metrics_train, "epoch\tacc\tmargin\treward")
    _ensure_metrics_file(metrics_val, "epoch\tacc\tmargin\treward")
    _ensure_metrics_file(metrics_test, "epoch\tacc\tmargin\treward")

    delta_t_values = []
    delta_d_values = []

    s_scen = 1.0
    for epoch in range(1, args.num_epochs + 1):
        epoch_acc, epoch_margin, epoch_reward = [], [], []
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            input_spikes = poisson_encode(images, args.T_semi, max_rate=args.max_rate).to(device)
            hidden_spikes, output_spikes, firing_rates = network(input_spikes)

            r_cls, r_margin, r_total = _compute_reward_components(firing_rates, labels, args.beta_margin)

            batch_buffer = EpisodeBuffer()

            for b in range(input_spikes.size(0)):
                events_in = _gather_events(
                    input_spikes[b : b + 1], hidden_spikes[b : b + 1], network.w_input_hidden, args.spike_array_len
                )
                events_out = _gather_events(
                    hidden_spikes[b : b + 1], output_spikes[b : b + 1], network.w_hidden_output, args.spike_array_len
                )

                state_batches = []
                extra_batches = []
                slices = []
                if events_in[0].numel() > 0:
                    state_batches.append(events_in[0])
                    extra_batches.append(events_in[1])
                    slices.append(("in", events_in))
                if events_out[0].numel() > 0:
                    state_batches.append(events_out[0])
                    extra_batches.append(events_out[1])
                    slices.append(("out", events_out))

                if state_batches:
                    states = torch.cat(state_batches, dim=0)
                    extras = torch.cat(extra_batches, dim=0)

                    action, log_prob, _ = actor(states, extras)
                    value = critic(states, extras)

                    offset = 0
                    episode_buffer = EpisodeBuffer()
                    for name, events in slices:
                        count = events[0].size(0)
                        idx_slice = slice(offset, offset + count)
                        if name == "in":
                            _scatter_updates(
                                args.local_lr * s_scen * action[idx_slice], events[2], events[3], network.w_input_hidden
                            )
                            torch.clamp_(network.w_input_hidden, args.exc_clip_min, args.exc_clip_max)
                        else:
                            _scatter_updates(
                                args.local_lr * s_scen * action[idx_slice], events[2], events[3], network.w_hidden_output
                            )
                            torch.clamp_(network.w_hidden_output, args.exc_clip_min, args.exc_clip_max)
                        delta_t_values.append(_extract_delta_t(events[0]).detach().cpu())
                        delta_d_values.append(action[idx_slice].detach().cpu())
                        offset += count

                    for i in range(states.size(0)):
                        episode_buffer.append(states[i], extras[i], action[i], log_prob[i], value[i])
                    episode_buffer.finalize(r_total[b])
                    batch_buffer.extend(episode_buffer)

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

            preds = firing_rates.argmax(dim=1)
            epoch_acc.append((preds == labels).float().mean().item())
            epoch_margin.append(r_margin.mean().item())
            epoch_reward.append(r_total.mean().item())

        mean_acc = sum(epoch_acc) / len(epoch_acc) if epoch_acc else 0.0
        mean_margin = sum(epoch_margin) / len(epoch_margin) if epoch_margin else 0.0
        mean_reward = sum(epoch_reward) / len(epoch_reward) if epoch_reward else 0.0

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

    delta_t_concat = torch.cat(delta_t_values, dim=0) if delta_t_values else torch.empty(0)
    delta_d_concat = torch.cat(delta_d_values, dim=0) if delta_d_values else torch.empty(0)
    if delta_t_concat.numel() > 0 and delta_d_concat.numel() > 0:
        plot_delta_t_delta_d(delta_t_concat, delta_d_concat, os.path.join(args.result_dir, "delta_t_delta_d.png"))

    plot_weight_histograms(w_input_hidden_before, network.w_input_hidden.detach().cpu(), os.path.join(args.result_dir, "hist_input_hidden.png"))
    plot_weight_histograms(w_hidden_output_before, network.w_hidden_output.detach().cpu(), os.path.join(args.result_dir, "hist_hidden_output.png"))
