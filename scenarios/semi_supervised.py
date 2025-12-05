import os
from typing import Tuple

import torch

from data.mnist import get_mnist_dataloaders
from rl.buffers import EventBatchBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update_events
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.lif import LIFParams
from snn.network_semi_supervised import SemiSupervisedNetwork
from utils.event_utils import gather_events
from utils.metrics import plot_delta_t_delta_d, plot_weight_histograms


def _ensure_metrics_file(path: str, header: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header + "\n")


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
    event_buffer = EventBatchBuffer(initial_capacity=args.batch_size_images * args.spike_array_len)

    s_scen = 1.0
    for epoch in range(1, args.num_epochs + 1):
        epoch_acc, epoch_margin, epoch_reward = [], [], []
        for batch_idx, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            input_spikes = poisson_encode(images, args.T_semi, max_rate=args.max_rate)
            hidden_spikes, output_spikes, firing_rates = network(input_spikes)

            preds = firing_rates.argmax(dim=1)
            batch_acc_tensor = (preds == labels).float().mean()
            epoch_acc.append(batch_acc_tensor.detach())

            r_cls, r_margin, r_total = _compute_reward_components(firing_rates, labels, args.beta_margin)

            event_buffer.reset()
            gather_events(
                input_spikes,
                hidden_spikes,
                network.w_input_hidden,
                args.spike_array_len,
                event_buffer,
                0,
            )
            gather_events(
                hidden_spikes,
                output_spikes,
                network.w_hidden_output,
                args.spike_array_len,
                event_buffer,
                1,
            )

            if len(event_buffer) > 0:
                states, extras, connection_ids, pre_idx, post_idx, batch_indices = event_buffer.flatten()
                # Actor/critic consume weight snapshots stored in extras during gather,
                # keeping the post-simulation evaluation aligned with the rollout
                # parameters (Theory.md episodic update assumption).
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

                delta_t_values.append(_extract_delta_t(states).detach())
                delta_d_values.append(actions.detach())

            epoch_margin.append(r_margin.detach())
            epoch_reward.append(r_total.detach())

        mean_acc = torch.stack(epoch_acc).mean().item() if epoch_acc else 0.0
        margin_tensor = (
            torch.cat([m.reshape(-1) for m in epoch_margin]) if epoch_margin else torch.empty(0, device=device)
        )
        reward_tensor = (
            torch.cat([r.reshape(-1) for r in epoch_reward]) if epoch_reward else torch.empty(0, device=device)
        )
        mean_margin = margin_tensor.mean().item() if margin_tensor.numel() > 0 else 0.0
        mean_reward = reward_tensor.mean().item() if reward_tensor.numel() > 0 else 0.0

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

    delta_t_concat = torch.cat(delta_t_values, dim=0).cpu() if delta_t_values else torch.empty(0)
    delta_d_concat = torch.cat(delta_d_values, dim=0).cpu() if delta_d_values else torch.empty(0)
    if delta_t_concat.numel() > 0 and delta_d_concat.numel() > 0:
        plot_delta_t_delta_d(delta_t_concat, delta_d_concat, os.path.join(args.result_dir, "delta_t_delta_d.png"))

    plot_weight_histograms(w_input_hidden_before, network.w_input_hidden.detach().cpu(), os.path.join(args.result_dir, "hist_input_hidden.png"))
    plot_weight_histograms(w_hidden_output_before, network.w_hidden_output.detach().cpu(), os.path.join(args.result_dir, "hist_hidden_output.png"))
