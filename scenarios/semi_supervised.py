import os
from typing import Tuple

import torch
import torch.nn.functional as F

from utils.metrics import plot_delta_t_delta_d, plot_weight_histograms

import os
from typing import Tuple

import torch
import torch.nn.functional as F

from data.mnist import get_mnist_dataloaders
from rl.buffers import EventBatchBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update_events
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.lif import LIFParams
from snn.network_semi_supervised import SemiSupervisedNetwork
from utils.metrics import plot_delta_t_delta_d, plot_weight_histograms


def _ensure_metrics_file(path: str, header: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header + "\n")


def _gather_events(
    pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor, L: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = pre_spikes.device
    batch_size, n_pre, T = pre_spikes.shape
    n_post = post_spikes.shape[1]

    pad_pre = F.pad(pre_spikes, (L - 1, 0))
    pad_post = F.pad(post_spikes, (L - 1, 0))
    pre_windows = pad_pre.unfold(2, L, 1)
    post_windows = pad_post.unfold(2, L, 1)
    post_windows_t = post_windows.permute(0, 2, 1, 3)

    batch_grid = (
        torch.arange(batch_size, device=device)
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, n_pre, T, n_post)
        .reshape(-1)
    )
    pre_grid = (
        torch.arange(n_pre, device=device)
        .view(1, n_pre, 1, 1)
        .expand(batch_size, n_pre, T, n_post)
        .reshape(-1)
    )
    post_grid = (
        torch.arange(n_post, device=device)
        .view(1, 1, 1, n_post)
        .expand(batch_size, n_pre, T, n_post)
        .reshape(-1)
    )

    pre_mask = pre_spikes.bool().unsqueeze(3).expand(-1, -1, -1, n_post)
    pre_mask_flat = pre_mask.reshape(-1)
    pre_indices = pre_mask_flat.nonzero(as_tuple=False).squeeze(1)

    pre_windows_exp = pre_windows.unsqueeze(3).expand(-1, -1, -1, n_post, -1).reshape(-1, L)
    post_windows_exp = post_windows_t.unsqueeze(1).expand(-1, n_pre, -1, -1, -1).reshape(-1, L)

    pre_histories = pre_windows_exp.index_select(0, pre_indices)
    post_histories = post_windows_exp.index_select(0, pre_indices)
    histories_pre = torch.stack([pre_histories, post_histories], dim=1)

    batch_pre = batch_grid.index_select(0, pre_indices)
    pre_idx = pre_grid.index_select(0, pre_indices)
    post_idx = post_grid.index_select(0, pre_indices)
    weights_pre = weights[pre_idx, post_idx].unsqueeze(1)
    event_type_pre = torch.tensor([1.0, 0.0], device=device, dtype=weights.dtype).expand(weights_pre.size(0), -1)
    extras_pre = torch.cat([weights_pre, event_type_pre], dim=1)

    post_mask = post_spikes.bool().permute(0, 2, 1).unsqueeze(1).expand(-1, n_pre, -1, -1)
    post_mask_flat = post_mask.reshape(-1)
    post_indices = post_mask_flat.nonzero(as_tuple=False).squeeze(1)

    pre_histories_post = pre_windows_exp.index_select(0, post_indices)
    post_histories_post = post_windows_exp.index_select(0, post_indices)
    histories_post = torch.stack([pre_histories_post, post_histories_post], dim=1)

    batch_post = batch_grid.index_select(0, post_indices)
    pre_idx_post = pre_grid.index_select(0, post_indices)
    post_idx_post = post_grid.index_select(0, post_indices)
    weights_post = weights[pre_idx_post, post_idx_post].unsqueeze(1)
    event_type_post = torch.tensor([0.0, 1.0], device=device, dtype=weights.dtype).expand(weights_post.size(0), -1)
    extras_post = torch.cat([weights_post, event_type_post], dim=1)

    return (
        torch.cat([histories_pre, histories_post], dim=0),
        torch.cat([extras_pre, extras_post], dim=0),
        torch.cat([pre_idx, pre_idx_post], dim=0),
        torch.cat([post_idx, post_idx_post], dim=0),
        torch.cat([batch_pre, batch_post], dim=0),
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
        for batch_idx, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            input_spikes = poisson_encode(images, args.T_semi, max_rate=args.max_rate).to(device)
            hidden_spikes, output_spikes, firing_rates = network(input_spikes)

            preds = firing_rates.argmax(dim=1)
            batch_acc = (preds == labels).float().mean().item()
            epoch_acc.append(batch_acc)

            r_cls, r_margin, r_total = _compute_reward_components(firing_rates, labels, args.beta_margin)

            state_in, extra_in, pre_in, post_in, batch_in = _gather_events(
                input_spikes, hidden_spikes, network.w_input_hidden, args.spike_array_len
            )
            state_out, extra_out, pre_out, post_out, batch_out = _gather_events(
                hidden_spikes, output_spikes, network.w_hidden_output, args.spike_array_len
            )

            event_buffer = EventBatchBuffer()
            if state_in.numel() > 0:
                event_buffer.add(batch_in, 0, state_in, extra_in, pre_in, post_in, batch_in)
            if state_out.numel() > 0:
                event_buffer.add(batch_out, 1, state_out, extra_out, pre_out, post_out, batch_out)

            if len(event_buffer) > 0:
                states, extras, _, connection_ids, pre_idx, post_idx, batch_indices = event_buffer.flatten()
                actions, log_probs_old, _ = actor(states, extras)
                values_old = critic(states, extras)

                returns = r_total.detach()[batch_indices]
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

                with torch.no_grad():
                    delta = args.local_lr * s_scen * actions.detach()
                    in_mask = connection_ids == 0
                    if in_mask.any():
                        _scatter_updates(delta[in_mask], pre_idx[in_mask], post_idx[in_mask], network.w_input_hidden)
                        network.w_input_hidden.clamp_(args.exc_clip_min, args.exc_clip_max)
                    out_mask = connection_ids == 1
                    if out_mask.any():
                        _scatter_updates(delta[out_mask], pre_idx[out_mask], post_idx[out_mask], network.w_hidden_output)
                        network.w_hidden_output.clamp_(args.exc_clip_min, args.exc_clip_max)

                delta_t_values.append(_extract_delta_t(states).detach().cpu())
                delta_d_values.append(actions.detach().cpu())

            epoch_margin.append(r_margin.mean().item())
            epoch_reward.append(r_total.mean().item())

            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                logger.info(
                    "Epoch %d/%d | Batch %d/%d | Train acc %.4f",
                    epoch,
                    args.num_epochs,
                    batch_idx,
                    len(train_loader),
                    batch_acc,
                )

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
