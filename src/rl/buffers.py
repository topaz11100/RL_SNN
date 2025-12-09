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


class StreamingEventBuffer:
    def __init__(self, max_batch_size: int, max_events_per_image: int, device: torch.device):
        self.max_batch_size = max_batch_size
        self.k = max_events_per_image
        self.device = device

        # Track total events seen per image in the batch
        self.seen_counts = torch.zeros(max_batch_size, device=device, dtype=torch.long)

        # Pre-allocated memory slots (Initialized lazily on first add to get dtype/shapes)
        self.states = None
        self.extras = None
        self.connection_ids = None
        self.pre_indices = None
        self.post_indices = None
        self.batch_indices = None

    def _lazy_init(self, state_ex: torch.Tensor, extras_ex: torch.Tensor):
        if self.states is not None:
            return
        # Strictly fixed allocation
        self.states = torch.zeros((self.max_batch_size, self.k, *state_ex.shape[1:]),
                                  device=self.device, dtype=state_ex.dtype)
        extras_dim = extras_ex.shape[1] if extras_ex.numel() > 0 else 0
        self.extras = torch.zeros((self.max_batch_size, self.k, extras_dim),
                                  device=self.device, dtype=extras_ex.dtype if extras_dim > 0 else torch.float32)
        self.connection_ids = torch.full((self.max_batch_size, self.k), -1, device=self.device, dtype=torch.long)
        self.pre_indices = torch.zeros((self.max_batch_size, self.k), device=self.device, dtype=torch.long)
        self.post_indices = torch.zeros((self.max_batch_size, self.k), device=self.device, dtype=torch.long)
        # Helper for flattening later
        self.batch_indices = torch.arange(self.max_batch_size, device=self.device, dtype=torch.long).unsqueeze(1).expand(-1, self.k)

    def add(self, connection_id, states, extras, pre_idx, post_idx, batch_idx):
        if states.numel() == 0:
            return
        self._lazy_init(states, extras)

        unique_batches = torch.unique(batch_idx)
        for b in unique_batches:
            mask = (batch_idx == b)
            curr_states = states[mask]
            if curr_states.numel() == 0:
                continue

            n_new = curr_states.shape[0]
            n_seen = self.seen_counts[b].item()

            # Global indices for incoming events: n_seen+1, n_seen+2, ...
            indices = torch.arange(n_new, device=self.device) + n_seen

            # 1. Fill empty slots
            direct_mask = (indices < self.k)
            if direct_mask.any():
                write_idx = indices[direct_mask].long()
                # Direct Write
                self.states[b, write_idx] = curr_states[direct_mask]
                if self.extras.shape[-1] > 0:
                    self.extras[b, write_idx] = extras[mask][direct_mask]
                self.connection_ids[b, write_idx] = connection_id
                self.pre_indices[b, write_idx] = pre_idx[mask][direct_mask]
                self.post_indices[b, write_idx] = post_idx[mask][direct_mask]

            # 2. Reservoir Replacement (Algorithm L)
            overflow_mask = ~direct_mask
            if overflow_mask.any():
                # Prob = k / (current_index + 1)
                current_global_indices = indices[overflow_mask].float() + 1.0
                probs = self.k / current_global_indices
                keep_decisions = torch.rand(probs.shape, device=self.device) < probs

                if keep_decisions.any():
                    num_replace = keep_decisions.sum()
                    replace_locs = torch.randint(0, self.k, (num_replace,), device=self.device)

                    # Map back to source indices
                    overflow_indices_local = torch.nonzero(overflow_mask, as_tuple=True)[0]
                    valid_src_indices = overflow_indices_local[torch.nonzero(keep_decisions, as_tuple=True)[0]]

                    self.states[b, replace_locs] = curr_states[valid_src_indices]
                    if self.extras.shape[-1] > 0:
                        self.extras[b, replace_locs] = extras[mask][valid_src_indices]
                    self.connection_ids[b, replace_locs] = connection_id
                    self.pre_indices[b, replace_locs] = pre_idx[mask][valid_src_indices]
                    self.post_indices[b, replace_locs] = post_idx[mask][valid_src_indices]

            self.seen_counts[b] += n_new

    def flatten(self):
        # Return only valid filled slots
        if self.states is None:
            raise ValueError("Buffer empty")
        valid_counts = torch.clamp(self.seen_counts, max=self.k)
        range_tensor = torch.arange(self.k, device=self.device).unsqueeze(0)
        mask = range_tensor < valid_counts.unsqueeze(1)
        return (self.states[mask], self.extras[mask], self.connection_ids[mask],
                self.pre_indices[mask], self.post_indices[mask], self.batch_indices[mask])

    def reset(self):
        self.seen_counts.zero_()

    def __len__(self):
        return int(torch.clamp(self.seen_counts, max=self.k).sum().item())
