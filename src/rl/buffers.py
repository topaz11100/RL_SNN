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

    def _allocate(
        self,
        count: int,
        state_shape: torch.Size,
        extras_dim: int,
        device,
        dtype,
        extras_dtype,
    ) -> None:
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
        new_capacity = max(needed, int(self.capacity * 1.2))

        def _grow(storage: torch.Tensor, shape_tail: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            new_tensor = torch.empty((new_capacity, *shape_tail), device=storage.device, dtype=dtype)
            new_tensor[: self.length].copy_(storage[: self.length])
            return new_tensor

        self.states = _grow(self.states, tuple(self.states.shape[1:]), self.states.dtype)
        self.extras = _grow(self.extras, (self.extras.size(1),), self.extras.dtype)
        self.batch_indices = _grow(self.batch_indices, (), self.batch_indices.dtype)
        self.connection_ids = _grow(self.connection_ids, (), self.connection_ids.dtype)
        self.pre_indices = _grow(self.pre_indices, (), self.pre_indices.dtype)
        self.post_indices = _grow(self.post_indices, (), self.post_indices.dtype)
        self.capacity = new_capacity

    def reserve(
        self,
        count: int,
        *,
        state_shape: torch.Size,
        extras_dim: int,
        device,
        state_dtype: torch.dtype,
        extras_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensure capacity for ``count`` events and return writable slices.

        The buffer length is advanced eagerly to avoid re-checking capacity for
        every sub-block within a single gather call.
        """

        if count == 0:
            empty = torch.empty((0,), device=device, dtype=torch.long)
            empty_states = torch.empty((0, *state_shape[1:]), device=device, dtype=state_dtype)
            empty_extras = torch.empty((0, extras_dim), device=device, dtype=extras_dtype)
            return empty_states, empty_extras, empty, empty, empty, empty

        if self.states is None:
            self._allocate(count, state_shape, extras_dim, device, state_dtype, extras_dtype)
        else:
            if self.states.shape[1:] != state_shape[1:]:
                raise ValueError("State shape mismatch for EventBatchBuffer")
            if self.extras.size(1) != extras_dim:
                raise ValueError("Extras dimension mismatch for EventBatchBuffer")
            self._ensure_capacity(count)

        start = self.length
        end = start + count
        self.length = end

        states_view = self.states[start:end]
        extras_view = self.extras[start:end]
        batch_view = self.batch_indices[start:end]
        conn_view = self.connection_ids[start:end]
        pre_view = self.pre_indices[start:end]
        post_view = self.post_indices[start:end]
        return states_view, extras_view, conn_view, pre_view, post_view, batch_view

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

        states_view, extras_view, conn_view, pre_view, post_view, batch_view = self.reserve(
            states.size(0),
            state_shape=states.shape,
            extras_dim=extras.size(1) if extras.numel() > 0 else 0,
            device=states.device,
            state_dtype=states.dtype,
            extras_dtype=extras.dtype if extras.numel() > 0 else states.dtype,
        )

        states_view.copy_(states.detach())
        if extras_view.numel() > 0:
            extras_view.copy_(extras.detach())
        conn_view.fill_(connection_id)
        pre_view.copy_(pre_idx)
        post_view.copy_(post_idx)
        batch_view.copy_(batch_idx.to(device=states.device, dtype=torch.long))

    def flatten(self, allow_empty: bool = False) -> Tuple[torch.Tensor, ...]:
        if self.states is None or self.length == 0:
            if allow_empty:
                empty_states = torch.empty(0, device="cpu")
                empty_long = torch.empty(0, dtype=torch.long)
                return empty_states, empty_states, empty_long, empty_long, empty_long, empty_long
            raise ValueError("No events were added to the buffer")

        states = self.states[: self.length]
        extras = self.extras[: self.length]
        connection_ids = self.connection_ids[: self.length]
        pre_idx = self.pre_indices[: self.length]
        post_idx = self.post_indices[: self.length]
        batch_idx = self.batch_indices[: self.length]
        return states, extras, connection_ids, pre_idx, post_idx, batch_idx

    def subsample_per_image(self, k: int) -> None:
        """Reservoir-style subsampling to keep at most ``k`` events per image.

        This enforces the per-image cap described in Theory 2.9.3 to prevent
        oversized PPO updates. The operation is in-place and keeps storage
        allocation intact for reuse across batches.
        """

        if self.states is None or self.length == 0:
            return

        if k <= 0:
            self.length = 0
            return

        batch_idx = self.batch_indices[: self.length]
        unique_batches = torch.unique(batch_idx)

        keep_mask = torch.zeros(self.length, device=batch_idx.device, dtype=torch.bool)
        for b in unique_batches:
            indices = (batch_idx == b).nonzero(as_tuple=False).squeeze(1)
            if indices.numel() > k:
                perm = torch.randperm(indices.numel(), device=batch_idx.device)[:k]
                indices = indices[perm]
            keep_mask[indices] = True

        keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(1)
        new_length = keep_indices.numel()
        if new_length == self.length:
            return

        self.states[:new_length].copy_(self.states[keep_indices])
        if self.extras is not None:
            self.extras[:new_length].copy_(self.extras[keep_indices])
        self.batch_indices[:new_length].copy_(self.batch_indices[keep_indices])
        self.connection_ids[:new_length].copy_(self.connection_ids[keep_indices])
        self.pre_indices[:new_length].copy_(self.pre_indices[keep_indices])
        self.post_indices[:new_length].copy_(self.post_indices[keep_indices])

        self.length = new_length

    def reset(self) -> None:
        """Reuse the allocated storage without freeing GPU memory."""
        self.length = 0

    def __len__(self) -> int:
        return self.length
