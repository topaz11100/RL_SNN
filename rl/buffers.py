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


class EventBatchBuffer:
    """Buffer to accumulate events across multiple episodes before PPO updates.

    List-backed storage keeps per-event tensors on device without repeated
    reallocation; a running length counter avoids recomputing sizes when the
    event volume is large. If event counts grow further, consider replacing the
    lists with pre-allocated chunks shaped from observed statistics.
    """

    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.extra_features: List[torch.Tensor] = []
        self.episode_ids: List[torch.Tensor] = []
        self.batch_indices: List[torch.Tensor] = []
        self.connection_ids: List[torch.Tensor] = []
        self.pre_indices: List[torch.Tensor] = []
        self.post_indices: List[torch.Tensor] = []
        self._length: int = 0

    def add(
        self,
        episode_id,
        connection_id: int,
        states: torch.Tensor,
        extras: torch.Tensor,
        pre_idx: torch.Tensor,
        post_idx: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> None:
        if states.numel() == 0:
            return
        count = states.size(0)
        device = states.device
        self.states.append(states.detach())
        self.extra_features.append(extras.detach())
        if torch.is_tensor(episode_id):
            episode_tensor = episode_id.to(device=device, dtype=torch.long)
        else:
            episode_tensor = torch.full((count,), episode_id, device=device, dtype=torch.long)
        self.episode_ids.append(episode_tensor)
        self.connection_ids.append(torch.full((count,), connection_id, device=device, dtype=torch.long))
        self.pre_indices.append(pre_idx)
        self.post_indices.append(post_idx)
        self.batch_indices.append(batch_idx.to(device=device, dtype=torch.long))
        self._length += count

    def flatten(self, allow_empty: bool = False) -> Tuple[torch.Tensor, ...]:
        if not self.states:
            if allow_empty:
                empty = torch.empty(0)
                return empty, empty, empty, empty, empty, empty, empty
            raise ValueError("No events were added to the buffer")
        states = torch.cat(self.states, dim=0)
        extras = torch.cat(self.extra_features, dim=0) if self.extra_features else torch.empty(0, device=states.device)
        episode_ids = torch.cat(self.episode_ids, dim=0)
        connection_ids = torch.cat(self.connection_ids, dim=0)
        pre_idx = torch.cat(self.pre_indices, dim=0)
        post_idx = torch.cat(self.post_indices, dim=0)
        batch_idx = torch.cat(self.batch_indices, dim=0)
        return states, extras, episode_ids, connection_ids, pre_idx, post_idx, batch_idx

    def __len__(self) -> int:
        return self._length
