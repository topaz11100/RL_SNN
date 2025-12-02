from typing import List, Tuple
import torch


class EpisodeBuffer:
    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.extra_features: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs_old: List[torch.Tensor] = []
        self.values_old: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []

    def append(
        self,
        state: torch.Tensor,
        extra_features: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.states.append(state.detach())
        self.extra_features.append(extra_features.detach())
        self.actions.append(action.detach())
        self.log_probs_old.append(log_prob.detach())
        self.values_old.append(value.detach())

    def finalize(self, R: torch.Tensor) -> None:
        reward_tensor = R.detach()
        self.rewards = [reward_tensor.clone() for _ in self.states]

    def get_batch(self) -> Tuple[torch.Tensor, ...]:
        if not self.rewards:
            raise ValueError("Rewards must be finalized before batching")
        states = torch.stack(self.states)
        extras = torch.stack(self.extra_features)
        actions = torch.stack(self.actions)
        log_probs_old = torch.stack(self.log_probs_old)
        values_old = torch.stack(self.values_old)
        rewards = torch.stack(self.rewards)
        return states, extras, actions, log_probs_old, values_old, rewards

    def extend(self, other: "EpisodeBuffer") -> None:
        """Append contents of another finalized buffer in order."""
        self.states.extend(other.states)
        self.extra_features.extend(other.extra_features)
        self.actions.extend(other.actions)
        self.log_probs_old.extend(other.log_probs_old)
        self.values_old.extend(other.values_old)
        self.rewards.extend(other.rewards)

    def __len__(self) -> int:
        return len(self.states)
