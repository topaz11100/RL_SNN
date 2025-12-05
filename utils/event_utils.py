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


def _pairwise_indices(
    primary_events: torch.Tensor,
    other_count: int,
    *,
    max_pairs: int,
    device,
    valid_mask: torch.Tensor | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Chunked pre/post 페어 인덱스 생성.

    repeat_interleave 로 전체 조합을 한 번에 만들 때 발생하는 거대한 임시
    텐서를 피하기 위해, `max_pairs` 에 맞춰 이벤트 블록을 잘라 처리한다.
    """

    batch_list: list[torch.Tensor] = []
    primary_list: list[torch.Tensor] = []
    other_list: list[torch.Tensor] = []

    if primary_events.numel() == 0:
        return batch_list, primary_list, other_list

    block_events = max(1, max_pairs // max(other_count, 1))
    other_indices_full = torch.arange(other_count, device=device)

    for start in range(0, primary_events.size(0), block_events):
        end = min(start + block_events, primary_events.size(0))
        chunk = primary_events[start:end]
        # (chunk, other_count)
        batch_grid = chunk[:, 0].unsqueeze(1).expand(-1, other_count)
        primary_grid = chunk[:, 1].unsqueeze(1).expand(-1, other_count)
        other_grid = other_indices_full.unsqueeze(0).expand(chunk.size(0), -1)
        time_grid = chunk[:, 2].unsqueeze(1).expand(-1, other_count)

        flat_batch = batch_grid.reshape(-1)
        flat_primary = primary_grid.reshape(-1)
        flat_other = other_grid.reshape(-1)
        flat_time = time_grid.reshape(-1)

        if valid_mask is not None:
            keep = valid_mask[flat_primary, flat_other].nonzero(as_tuple=False).squeeze(1)
            flat_batch = flat_batch.index_select(0, keep)
            flat_primary = flat_primary.index_select(0, keep)
            flat_other = flat_other.index_select(0, keep)
            flat_time = flat_time.index_select(0, keep)

        if flat_batch.numel() == 0:
            continue

        batch_list.append(torch.stack([flat_batch, flat_time], dim=1))
        primary_list.append(flat_primary)
        other_list.append(flat_other)

    return batch_list, primary_list, other_list


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
    n_pre = pre_spikes.shape[1]
    n_post = post_spikes.shape[1]

    padded_pre = padded_pre if padded_pre is not None else F.pad(pre_spikes, (window - 1, 0))
    padded_post = padded_post if padded_post is not None else F.pad(post_spikes, (window - 1, 0))
    weight_snapshot = weights.detach()

    pre_events = pre_spikes.nonzero(as_tuple=False)
    batch_time_pre, primary_pre, other_pre = _pairwise_indices(
        pre_events, n_post, max_pairs=max_pairs, device=device, valid_mask=valid_mask
    )
    for bt, pre_idx, post_idx in zip(batch_time_pre, primary_pre, other_pre):
        states, extras = _build_states_and_extras(
            padded_pre,
            padded_post,
            window,
            weight_snapshot,
            l_norm=l_norm,
            event_type=_EVENT_TYPE_PRE,
            batch_and_time=bt,
            pre_idx=pre_idx,
            post_idx=post_idx,
            device=device,
        )
        states_view, extras_view, conn_view, pre_view, post_view, batch_view = buffer.reserve(
            states.size(0),
            state_shape=states.shape,
            extras_dim=extras.size(1) if extras.numel() > 0 else 0,
            device=device,
            state_dtype=states.dtype,
            extras_dtype=extras.dtype if extras.numel() > 0 else weight_snapshot.dtype,
        )
        if states_view.numel() == 0:
            continue
        states_view.copy_(states)
        if extras_view.numel() > 0:
            extras_view.copy_(extras)
        conn_view.fill_(connection_id)
        pre_view.copy_(pre_idx)
        post_view.copy_(post_idx)
        batch_view.copy_(bt[:, 0])

    post_events = post_spikes.nonzero(as_tuple=False)
    batch_time_post, primary_post, other_post = _pairwise_indices(
        post_events, n_pre, max_pairs=max_pairs, device=device, valid_mask=valid_mask
    )
    for bt, pre_idx, post_idx in zip(batch_time_post, other_post, primary_post):
        states, extras = _build_states_and_extras(
            padded_pre,
            padded_post,
            window,
            weight_snapshot,
            l_norm=l_norm,
            event_type=_EVENT_TYPE_POST,
            batch_and_time=bt,
            pre_idx=pre_idx,
            post_idx=post_idx,
            device=device,
        )
        states_view, extras_view, conn_view, pre_view, post_view, batch_view = buffer.reserve(
            states.size(0),
            state_shape=states.shape,
            extras_dim=extras.size(1) if extras.numel() > 0 else 0,
            device=device,
            state_dtype=states.dtype,
            extras_dtype=extras.dtype if extras.numel() > 0 else weight_snapshot.dtype,
        )
        if states_view.numel() == 0:
            continue
        states_view.copy_(states)
        if extras_view.numel() > 0:
            extras_view.copy_(extras)
        conn_view.fill_(connection_id)
        pre_view.copy_(pre_idx)
        post_view.copy_(post_idx)
        batch_view.copy_(bt[:, 0])
