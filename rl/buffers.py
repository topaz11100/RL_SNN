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
    """GPU 친화적 이벤트 버퍼.

    Theory 2.9.3의 이미지 단위 미니배치 구성을 유지하면서도, 리스트 append로
    인한 메모리 파편화를 줄이기 위해 고정 크기 블록을 미리 할당하고 필요
    시 두 배 확장한다. 동일한 디바이스/데이터타입을 일관되게 유지하여
    연속적인 GPU 메모리 배치를 보장한다.
    """

    def __init__(self, initial_capacity: int = 4096):
        # Generous preallocation minimizes cudaMalloc/cudaFree churn during training
        self.initial_capacity = initial_capacity
        self.capacity = 0
        self.length = 0

        self.states: torch.Tensor | None = None
        self.extras: torch.Tensor | None = None
        self.batch_indices: torch.Tensor | None = None
        self.connection_ids: torch.Tensor | None = None
        self.pre_indices: torch.Tensor | None = None
        self.post_indices: torch.Tensor | None = None

    def _allocate(self, count: int, state_shape: torch.Size, extras_dim: int, device, dtype, extras_dtype) -> None:
        self.capacity = max(self.initial_capacity, count)
        self.states = torch.empty((self.capacity, *state_shape[1:]), device=device, dtype=dtype)
        self.extras = torch.empty((self.capacity, extras_dim), device=device, dtype=extras_dtype)
        self.batch_indices = torch.empty((self.capacity,), device=device, dtype=torch.long)
        self.connection_ids = torch.empty((self.capacity,), device=device, dtype=torch.long)
        self.pre_indices = torch.empty((self.capacity,), device=device, dtype=torch.long)
        self.post_indices = torch.empty((self.capacity,), device=device, dtype=torch.long)

    def _ensure_capacity(self, additional: int) -> None:
        if self.states is None:
            raise RuntimeError("Buffer not allocated")
        needed = self.length + additional
        if needed <= self.capacity:
            return
        new_capacity = max(needed, self.capacity * 2)
        self.states = torch.cat(
            [self.states, torch.empty((new_capacity - self.capacity, *self.states.shape[1:]), device=self.states.device, dtype=self.states.dtype)],
            dim=0,
        )
        self.extras = torch.cat(
            [self.extras, torch.empty((new_capacity - self.capacity, self.extras.size(1)), device=self.extras.device, dtype=self.extras.dtype)],
            dim=0,
        )
        self.batch_indices = torch.cat(
            [self.batch_indices, torch.empty((new_capacity - self.capacity,), device=self.batch_indices.device, dtype=self.batch_indices.dtype)],
            dim=0,
        )
        self.connection_ids = torch.cat(
            [self.connection_ids, torch.empty((new_capacity - self.capacity,), device=self.connection_ids.device, dtype=self.connection_ids.dtype)],
            dim=0,
        )
        self.pre_indices = torch.cat(
            [self.pre_indices, torch.empty((new_capacity - self.capacity,), device=self.pre_indices.device, dtype=self.pre_indices.dtype)],
            dim=0,
        )
        self.post_indices = torch.cat(
            [self.post_indices, torch.empty((new_capacity - self.capacity,), device=self.post_indices.device, dtype=self.post_indices.dtype)],
            dim=0,
        )
        self.capacity = new_capacity

    def add(
        self,
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

        if self.states is None:
            extras_dim = extras.size(1) if extras.numel() > 0 else 0
            self._allocate(count, states.shape, extras_dim, device, states.dtype, extras.dtype if extras_dim > 0 else states.dtype)
        else:
            self._ensure_capacity(count)

        end = self.length + count
        self.states[self.length : end] = states.detach()
        if extras.numel() > 0:
            self.extras[self.length : end] = extras.detach()
        self.connection_ids[self.length : end] = connection_id
        self.pre_indices[self.length : end] = pre_idx
        self.post_indices[self.length : end] = post_idx
        # batch_idx must be local to the current mini-batch (0..batch_size-1) to keep
        # reward/advantage lookups aligned during PPO updates.
        self.batch_indices[self.length : end] = batch_idx.to(device=device, dtype=torch.long)
        self.length = end

    def flatten(self, allow_empty: bool = False) -> Tuple[torch.Tensor, ...]:
        if self.states is None or self.length == 0:
            if allow_empty:
                empty = torch.empty(0)
                return empty, empty, empty, empty, empty, empty
            raise ValueError("No events were added to the buffer")

        states = self.states[: self.length]
        extras = self.extras[: self.length]
        connection_ids = self.connection_ids[: self.length]
        pre_idx = self.pre_indices[: self.length]
        post_idx = self.post_indices[: self.length]
        batch_idx = self.batch_indices[: self.length]
        return states, extras, connection_ids, pre_idx, post_idx, batch_idx

    def reset(self) -> None:
        """Reuse the allocated storage without freeing GPU memory."""
        self.length = 0

    def __len__(self) -> int:
        return self.length
