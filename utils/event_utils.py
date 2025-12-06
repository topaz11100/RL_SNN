import torch
import torch.nn.functional as F

from rl.buffers import EventBatchBuffer

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


def _extract_events(spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = spikes.to(torch.bool)
    if not mask.any():
        empty = torch.empty(0, device=spikes.device, dtype=torch.long)
        return empty, empty, empty

    batch_range = torch.arange(spikes.size(0), device=spikes.device)[:, None, None]
    neuron_range = torch.arange(spikes.size(1), device=spikes.device)[None, :, None]
    time_range = torch.arange(spikes.size(2), device=spikes.device)[None, None, :]

    batch_idx = batch_range.expand_as(spikes)[mask]
    neuron_idx = neuron_range.expand_as(spikes)[mask]
    time_idx = time_range.expand_as(spikes)[mask]
    return batch_idx, neuron_idx, time_idx


def _build_states_and_extras(
    padded_pre: torch.Tensor,
    padded_post: torch.Tensor,
    window: int,
    weights: torch.Tensor,
    *,
    l_norm: float | None,
    event_type: torch.Tensor,
    batch_and_time: torch.Tensor,
    pre_idx: torch.Tensor,
    post_idx: torch.Tensor,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # batch_and_time는 shape (N, 2) = [batch, time]
    batches = batch_and_time[:, 0]
    times = batch_and_time[:, 1]

    pre_hist = _select_windows(padded_pre, batches, pre_idx, times, window)
    post_hist = _select_windows(padded_post, batches, post_idx, times, window)

    states = torch.stack([pre_hist, post_hist], dim=1).to(torch.bool)
    weights_sel = weights[pre_idx, post_idx].unsqueeze(1)
    extras_parts = [weights_sel]
    if l_norm is not None:
        extras_parts.append(torch.full_like(weights_sel, l_norm))
    extras_parts.append(_expand_event_type(event_type, weights_sel.size(0), device, weights.dtype))
    extras = torch.cat(extras_parts, dim=1)
    return states, extras


def gather_events(
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    weights: torch.Tensor,
    window: int,
    buffer: EventBatchBuffer,
    connection_id: int,
    *,
    l_norm: float | None = None,
    valid_mask: torch.Tensor | None = None,
    padded_pre: torch.Tensor | None = None,
    padded_post: torch.Tensor | None = None,
    max_pairs: int = 131072,
) -> None:
    """Sparse 이벤트를 직접 ``EventBatchBuffer``에 기록한다.

    Triple-copy가 발생하던 리스트 → cat → add 경로를 제거하고, JIT friendly한
    연속 버퍼에 즉시 기록하여 GPU 메모리 대역폭을 절약한다. `weights`는
    호출 시점의 값을 detach하여 Actor가 시뮬레이션 당시 가중치에 대한
    extras를 받도록 명시적으로 보장한다.
    """

    device = pre_spikes.device
    _ = max_pairs  # Legacy argument retained for API compatibility.
    n_pre = pre_spikes.shape[1]
    n_post = post_spikes.shape[1]

    padded_pre = padded_pre if padded_pre is not None else F.pad(pre_spikes, (window - 1, 0))
    padded_post = padded_post if padded_post is not None else F.pad(post_spikes, (window - 1, 0))
    weight_snapshot = weights.detach()

    def _write_events(states: torch.Tensor, extras: torch.Tensor, pre_idx: torch.Tensor, post_idx: torch.Tensor, batch_idx: torch.Tensor) -> None:
        states_view, extras_view, conn_view, pre_view, post_view, batch_view = buffer.reserve(
            states.size(0),
            state_shape=states.shape,
            extras_dim=extras.size(1) if extras.numel() > 0 else 0,
            device=device,
            state_dtype=states.dtype,
            extras_dtype=extras.dtype if extras.numel() > 0 else weight_snapshot.dtype,
        )
        if states_view.numel() == 0:
            return
        states_view.copy_(states)
        if extras_view.numel() > 0:
            extras_view.copy_(extras)
        conn_view.fill_(connection_id)
        pre_view.copy_(pre_idx)
        post_view.copy_(post_idx)
        batch_view.copy_(batch_idx)

    if valid_mask is not None:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    # Pre-synaptic spikes paired with all post-synaptic neurons (masked if provided).
    batch_pre, pre_indices, time_pre = _extract_events(pre_spikes)
    if batch_pre.numel() > 0:
        post_all = torch.arange(n_post, device=device)
        repeat_factor = n_post
        batch_rep = batch_pre.repeat_interleave(repeat_factor)
        time_rep = time_pre.repeat_interleave(repeat_factor)
        pre_rep = pre_indices.repeat_interleave(repeat_factor)
        post_rep = post_all.repeat(batch_pre.size(0))

        if valid_mask is not None:
            keep = valid_mask[pre_rep, post_rep]
            batch_rep = batch_rep[keep]
            time_rep = time_rep[keep]
            pre_rep = pre_rep[keep]
            post_rep = post_rep[keep]

        if batch_rep.numel() > 0:
            batch_time = torch.stack((batch_rep, time_rep), dim=1)
            states, extras = _build_states_and_extras(
                padded_pre,
                padded_post,
                window,
                weight_snapshot,
                l_norm=l_norm,
                event_type=_EVENT_TYPE_PRE,
                batch_and_time=batch_time,
                pre_idx=pre_rep,
                post_idx=post_rep,
                device=device,
            )
            _write_events(states, extras, pre_rep, post_rep, batch_rep)

    # Post-synaptic spikes paired with all pre-synaptic neurons (masked if provided).
    batch_post, post_indices, time_post = _extract_events(post_spikes)
    if batch_post.numel() > 0:
        pre_all = torch.arange(n_pre, device=device)
        repeat_factor = n_pre
        batch_rep = batch_post.repeat_interleave(repeat_factor)
        time_rep = time_post.repeat_interleave(repeat_factor)
        pre_rep = pre_all.repeat(batch_post.size(0))
        post_rep = post_indices.repeat_interleave(repeat_factor)

        if valid_mask is not None:
            keep = valid_mask[pre_rep, post_rep]
            batch_rep = batch_rep[keep]
            time_rep = time_rep[keep]
            pre_rep = pre_rep[keep]
            post_rep = post_rep[keep]

        if batch_rep.numel() > 0:
            batch_time = torch.stack((batch_rep, time_rep), dim=1)
            states, extras = _build_states_and_extras(
                padded_pre,
                padded_post,
                window,
                weight_snapshot,
                l_norm=l_norm,
                event_type=_EVENT_TYPE_POST,
                batch_and_time=batch_time,
                pre_idx=pre_rep,
                post_idx=post_rep,
                device=device,
            )
            _write_events(states, extras, pre_rep, post_rep, batch_rep)
