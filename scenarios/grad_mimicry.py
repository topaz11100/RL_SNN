import os
from typing import Tuple

import torch
import torch.nn.functional as F

from data.mnist import get_mnist_dataloaders
from rl.buffers import EpisodeBuffer
from rl.policy import GaussianPolicy
from rl.ppo import ppo_update
from rl.value import ValueFunction
from snn.encoding import poisson_encode
from snn.network_grad_mimicry import GradMimicryNetwork


def _ensure_metrics_file(path: str, header: str) -> None:
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header + "\n")


def _prepare_state(pre_spikes: torch.Tensor, post_spikes: torch.Tensor, L: int) -> torch.Tensor:
    pre_hist = pre_spikes.mean(dim=1)
    post_hist = post_spikes.mean(dim=1)
    T = pre_hist.shape[1]
    if L > T:
        L = T
    pre_seg = pre_hist[:, -L:]
    post_seg = post_hist[:, -L:]
    return torch.stack([pre_seg, post_seg], dim=1)


def _compute_reward(delta_agent: torch.Tensor, delta_teacher_in: torch.Tensor, delta_teacher_out: torch.Tensor) -> torch.Tensor:
    diff_in = -(delta_agent - delta_teacher_in).pow(2)
    diff_out = -(delta_agent - delta_teacher_out).pow(2)
    return torch.cat([diff_in.flatten(), diff_out.flatten()]).mean()


def _evaluate(network: GradMimicryNetwork, loader, device, args) -> Tuple[float, float]:
    network.eval()
    accuracies, rewards = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            spikes = poisson_encode(images, args.T_sup, max_rate=args.max_rate).to(device)
            _, firing_rates = network(spikes)
            preds = firing_rates.argmax(dim=1)
            accuracies.append((preds == labels).float().mean().item())
            # Reward placeholder based on margin for evaluation
            true_rates = firing_rates.gather(1, labels.view(-1, 1)).squeeze(1)
            max_other, _ = (firing_rates + torch.eye(firing_rates.size(1), device=firing_rates.device) * -1e9).max(dim=1)
            margin = true_rates - max_other
            rewards.append(margin.mean().item())
    network.train()
    return (
        sum(accuracies) / len(accuracies) if accuracies else 0.0,
        sum(rewards) / len(rewards) if rewards else 0.0,
    )


def run_grad(args, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(args.batch_size_images, args.seed)

    network = GradMimicryNetwork().to(device)
    teacher = GradMimicryNetwork().to(device)
    actor = GaussianPolicy(sigma=getattr(args, "sigma_sup", args.sigma_unsup1)).to(device)
    critic = ValueFunction().to(device)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    metrics_train = os.path.join(args.result_dir, "metrics_train.txt")
    metrics_val = os.path.join(args.result_dir, "metrics_val.txt")
    metrics_test = os.path.join(args.result_dir, "metrics_test.txt")
    _ensure_metrics_file(metrics_train, "epoch\tacc\treward\talign")
    _ensure_metrics_file(metrics_val, "epoch\tacc\treward\talign")
    _ensure_metrics_file(metrics_test, "epoch\tacc\treward\talign")

    for epoch in range(1, args.num_epochs + 1):
        epoch_acc, epoch_reward, epoch_align = [], [], []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            input_spikes = poisson_encode(images, args.T_sup, max_rate=args.max_rate).to(device)

            # RL network forward
            output_spikes, firing_rates = network(input_spikes)
            state = _prepare_state(input_spikes, output_spikes, args.spike_array_len)
            action, log_prob, _ = actor(state)
            value = critic(state).squeeze(-1)

            # Apply agent update (scalar applied across weights)
            with torch.no_grad():
                delta_agent = 0.01 * action.mean()
                network.w_input_hidden.data = network.w_input_hidden.data + delta_agent
                network.w_hidden_output.data = network.w_hidden_output.data + delta_agent

            # Teacher gradients
            teacher.load_state_dict(network.state_dict())
            teacher.zero_grad()
            teacher_output, teacher_rates = teacher(input_spikes)
            logits = teacher_rates * 5.0
            loss_sup = F.cross_entropy(logits, labels)
            loss_sup.backward()
            delta_teacher_in = -args.alpha_align * teacher.w_input_hidden.grad
            delta_teacher_out = -args.alpha_align * teacher.w_hidden_output.grad

            reward = _compute_reward(delta_agent, delta_teacher_in, delta_teacher_out)

            preds = firing_rates.argmax(dim=1)
            epoch_acc.append((preds == labels).float().mean().item())
            epoch_reward.append(reward.item())
            epoch_align.append(reward.item())

            for i in range(state.size(0)):
                buffer = EpisodeBuffer()
                buffer.append(state[i], action[i], log_prob[i], value[i])
                buffer.finalize(reward)
                ppo_update(
                    actor,
                    critic,
                    buffer,
                    optimizer_actor,
                    optimizer_critic,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=min(args.ppo_batch_size, len(buffer)),
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

        logger.info(
            "Epoch %d | Train acc %.4f reward %.4f align %.4f | Val acc %.4f | Test acc %.4f",
            epoch,
            mean_acc,
            mean_reward,
            mean_align,
            val_acc,
            test_acc,
        )
