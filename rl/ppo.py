from typing import Optional
import torch
from torch import nn

from rl.buffers import EpisodeBuffer


def ppo_update(
    actor: nn.Module,
    critic: nn.Module,
    buffer: EpisodeBuffer,
    optimizer_actor: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    ppo_epochs: int,
    batch_size: int,
    eps_clip: float = 0.2,
    c_v: float = 1.0,
):
    states, extras, actions, log_probs_old, values_old, rewards = buffer.get_batch()
    advantages = (rewards - values_old).detach()
    num_samples = states.size(0)

    for _ in range(ppo_epochs):
        indices = torch.randperm(num_samples, device=states.device)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            states_mb = states[batch_idx]
            actions_mb = actions[batch_idx]
            log_probs_old_mb = log_probs_old[batch_idx]
            advantages_mb = advantages[batch_idx]
            advantages_mb = (advantages_mb - advantages_mb.mean()) / (
                advantages_mb.std() + 1e-8
            )
            rewards_mb = rewards[batch_idx]

            extras_mb = extras[batch_idx] if extras.numel() > 0 else None

            _, log_probs_new, _ = actor(states_mb, extras_mb, actions=actions_mb)
            ratio = torch.exp(log_probs_new - log_probs_old_mb)

            unclipped = ratio * advantages_mb
            clipped = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages_mb
            actor_loss = (-torch.min(unclipped, clipped)).mean()

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            values_new = critic(states_mb, extras_mb)
            value_loss = (rewards_mb - values_new).pow(2).mean() * c_v

            optimizer_critic.zero_grad()
            value_loss.backward()
            optimizer_critic.step()


def ppo_update_events(
    actor: nn.Module,
    critic: nn.Module,
    states: torch.Tensor,
    extras: torch.Tensor,
    actions_old: torch.Tensor,
    log_probs_old: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    optimizer_actor: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    ppo_epochs: int,
    batch_size: int,
    eps_clip: float = 0.2,
    c_v: float = 1.0,
):
    """PPO update using flattened event batches."""

    num_samples = states.size(0)
    if num_samples == 0:
        return
    extras_available = extras.numel() > 0

    for _ in range(ppo_epochs):
        indices = torch.randperm(num_samples, device=states.device)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            states_mb = states[batch_idx]
            actions_mb = actions_old[batch_idx]
            log_probs_old_mb = log_probs_old[batch_idx]
            advantages_mb = advantages[batch_idx]
            advantages_mb = (advantages_mb - advantages_mb.mean()) / (
                advantages_mb.std() + 1e-8
            )
            returns_mb = returns[batch_idx]

            extras_mb = extras[batch_idx] if extras_available else None

            _, log_probs_new, _ = actor(states_mb, extras_mb, actions=actions_mb)
            ratio = torch.exp(log_probs_new - log_probs_old_mb)

            surr1 = ratio * advantages_mb
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages_mb
            actor_loss = (-torch.min(surr1, surr2)).mean()

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            values_new = critic(states_mb, extras_mb)
            value_loss = (returns_mb - values_new).pow(2).mean() * c_v

            optimizer_critic.zero_grad()
            value_loss.backward()
            optimizer_critic.step()


# Example self-check (not executed automatically):
# if __name__ == "__main__":
#     import torch.nn as nn
#     from rl.policy import GaussianPolicy
#     from rl.value import ValueFunction
#
#     torch.manual_seed(0)
#     batch, L = 8, 20
#     dummy_states = torch.randn(batch, 2, L)
#     dummy_rewards = torch.ones(batch)
#
#     actor = GaussianPolicy(sigma=0.1)
#     critic = ValueFunction()
#     opt_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
#     opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
#
#     buffer = EpisodeBuffer()
#     with torch.no_grad():
#         action, logp, _ = actor(dummy_states)
#         value = critic(dummy_states)
#         for i in range(batch):
#             buffer.append(dummy_states[i], action[i], logp[i], value[i])
#     buffer.finalize(dummy_rewards)
#
#     ppo_update(
#         actor,
#         critic,
#         buffer,
#         opt_actor,
#         opt_critic,
#         ppo_epochs=2,
#         batch_size=4,
#         eps_clip=0.2,
#         c_v=1.0,
#     )
#     print("PPO self-check finished")
