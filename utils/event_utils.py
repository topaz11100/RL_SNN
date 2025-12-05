import torch
import torch.nn.functional as F

_EVENT_TYPE_PRE = torch.tensor([1.0, 0.0])
_EVENT_TYPE_POST = torch.tensor([0.0, 1.0])


def _expand_event_type(base: torch.Tensor, count: int, device, dtype) -> torch.Tensor:
    return base.to(device=device, dtype=dtype).expand(count, -1)


def _select_windows(
    padded_spikes: torch.Tensor, batch_idx: torch.Tensor, neuron_idx: torch.Tensor, time_idx: torch.Tensor, window: int
) -> torch.Tensor:
    if batch_idx.numel() == 0:
        return torch.empty((0, window), device=padded_spikes.device, dtype=padded_spikes.dtype)
    time_offsets = torch.arange(window, device=padded_spikes.device)
    gather_times = time_idx.unsqueeze(1) + time_offsets
    return padded_spikes[batch_idx.unsqueeze(1), neuron_idx.unsqueeze(1), gather_times]


def gather_events(
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    weights: torch.Tensor,
    window: int,
    *,
    l_norm: float | None = None,
    valid_mask: torch.Tensor | None = None,
    padded_pre: torch.Tensor | None = None,
    padded_post: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect sparse pre/post spike histories without building dense unfold buffers.

    Windows are sliced directly around non-zero spikes using advanced indexing to
    avoid the `(batch, neurons, T, window)` allocations that previously caused
    OOM on large layers.
    """

    device = pre_spikes.device
    n_pre = pre_spikes.shape[1]
    n_post = post_spikes.shape[1]

    # Allow callers to reuse already padded tensors to avoid redundant F.pad
    # allocations when the same spike trains serve as pre/post across layers.
    padded_pre = padded_pre if padded_pre is not None else F.pad(pre_spikes, (window - 1, 0))
    padded_post = padded_post if padded_post is not None else F.pad(post_spikes, (window - 1, 0))

    states_list: list[torch.Tensor] = []
    extras_list: list[torch.Tensor] = []
    pre_indices_list: list[torch.Tensor] = []
    post_indices_list: list[torch.Tensor] = []
    batch_indices_list: list[torch.Tensor] = []

    pre_events = pre_spikes.nonzero(as_tuple=False)
    if pre_events.numel() > 0:
        batch_pre = pre_events[:, 0].repeat_interleave(n_post)
        pre_idx = pre_events[:, 1].repeat_interleave(n_post)
        time_idx = pre_events[:, 2].repeat_interleave(n_post)
        post_idx = torch.arange(n_post, device=device).repeat(pre_events.size(0))

        if valid_mask is not None:
            keep = valid_mask[pre_idx, post_idx].nonzero(as_tuple=False).squeeze(1)
            batch_pre = batch_pre.index_select(0, keep)
            pre_idx = pre_idx.index_select(0, keep)
            time_idx = time_idx.index_select(0, keep)
            post_idx = post_idx.index_select(0, keep)

        if pre_idx.numel() > 0:
            pre_hist = _select_windows(padded_pre, batch_pre, pre_idx, time_idx, window)
            post_hist = _select_windows(padded_post, batch_pre, post_idx, time_idx, window)
            states_list.append(torch.stack([pre_hist, post_hist], dim=1))

            weights_pre = weights[pre_idx, post_idx].unsqueeze(1)
            extras_parts = [weights_pre]
            if l_norm is not None:
                extras_parts.append(torch.full_like(weights_pre, l_norm))
            extras_parts.append(_expand_event_type(_EVENT_TYPE_PRE, weights_pre.size(0), device, weights.dtype))
            extras_list.append(torch.cat(extras_parts, dim=1))

            pre_indices_list.append(pre_idx)
            post_indices_list.append(post_idx)
            batch_indices_list.append(batch_pre)

    post_events = post_spikes.nonzero(as_tuple=False)
    if post_events.numel() > 0:
        batch_post = post_events[:, 0].repeat_interleave(n_pre)
        post_idx = post_events[:, 1].repeat_interleave(n_pre)
        time_idx = post_events[:, 2].repeat_interleave(n_pre)
        pre_idx = torch.arange(n_pre, device=device).repeat(post_events.size(0))

        if valid_mask is not None:
            keep = valid_mask[pre_idx, post_idx].nonzero(as_tuple=False).squeeze(1)
            batch_post = batch_post.index_select(0, keep)
            pre_idx = pre_idx.index_select(0, keep)
            time_idx = time_idx.index_select(0, keep)
            post_idx = post_idx.index_select(0, keep)

        if pre_idx.numel() > 0:
            pre_hist = _select_windows(padded_pre, batch_post, pre_idx, time_idx, window)
            post_hist = _select_windows(padded_post, batch_post, post_idx, time_idx, window)
            states_list.append(torch.stack([pre_hist, post_hist], dim=1))

            weights_post = weights[pre_idx, post_idx].unsqueeze(1)
            extras_parts = [weights_post]
            if l_norm is not None:
                extras_parts.append(torch.full_like(weights_post, l_norm))
            extras_parts.append(_expand_event_type(_EVENT_TYPE_POST, weights_post.size(0), device, weights.dtype))
            extras_list.append(torch.cat(extras_parts, dim=1))

            pre_indices_list.append(pre_idx)
            post_indices_list.append(post_idx)
            batch_indices_list.append(batch_post)

    if not states_list:
        empty_state = torch.empty((0, 2, window), device=device, dtype=pre_spikes.dtype)
        empty_extras = torch.empty((0, (3 if l_norm is None else 4)), device=device, dtype=weights.dtype)
        empty_index = torch.empty((0,), device=device, dtype=torch.long)
        return empty_state, empty_extras, empty_index, empty_index, empty_index

    states_cat = torch.cat(states_list, dim=0) if len(states_list) > 1 else states_list[0]
    extras_cat = torch.cat(extras_list, dim=0) if len(extras_list) > 1 else extras_list[0]
    pre_cat = torch.cat(pre_indices_list, dim=0) if len(pre_indices_list) > 1 else pre_indices_list[0]
    post_cat = torch.cat(post_indices_list, dim=0) if len(post_indices_list) > 1 else post_indices_list[0]
    batch_cat = torch.cat(batch_indices_list, dim=0) if len(batch_indices_list) > 1 else batch_indices_list[0]

    return states_cat, extras_cat, pre_cat, post_cat, batch_cat
