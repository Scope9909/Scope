import numpy as np
import torch
import math
from logging import getLogger

def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))


def time_weighted_huber_loss(preds, labels, delta=1.0, weights=None):
    """
    时间步加权的 Huber Loss
    
    参数：
        preds: 预测值，形状 (B, T, N, C)
        labels: 真实值，形状 (B, T, N, C)
        delta: Huber 损失阈值
        weights: 时间步权重，形状 (T,) 或 None（不使用权重）
    
    返回：
        加权后的损失（标量）
    """
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    loss = torch.where(condition, small_res, large_res)  # (B, T, N, C)
    
    # ===== 时间步加权 =====
    if weights is not None:
        # weights: (T,) -> (1, T, 1, 1) 用于广播
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=preds.device, dtype=preds.dtype)
        weights = weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        loss = loss * weights  # 对每个时间步加权
    
    return torch.mean(loss)


def _prepare_mask_torch(labels, null_val=np.nan, mask_val=np.nan):
    """Return cleaned labels and a boolean mask without mutating the caller's tensor."""
    labels = torch.where(
        torch.abs(labels) < 1e-4,
        torch.zeros_like(labels),
        labels,
    )
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    return labels, mask


def masked_valid_count_torch(labels, null_val=np.nan, mask_val=np.nan):
    """Count elements included by the same mask used by the masked metrics."""
    _, mask = _prepare_mask_torch(labels, null_val, mask_val)
    return mask.sum()


def _normalized_mask(mask, dtype):
    mask = mask.to(dtype=dtype)
    eps = torch.finfo(mask.dtype).eps
    return mask / torch.clamp(mask.mean(), min=eps)


def masked_huber_loss(preds, labels, delta=1.0, null_val=np.nan, mask_val=np.nan):
    labels, mask = _prepare_mask_torch(labels, null_val, mask_val)
    mask = _normalized_mask(mask, preds.dtype)
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    loss = torch.where(condition, small_res, large_res)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def mae(preds,labels):
    loss = torch.mean(torch.abs(preds - labels))
    return loss
def mse(preds,labels):
    loss = torch.mean(torch.square(preds - labels))
    return loss
def rmse(preds,labels):
    loss = torch.mean(torch.square(preds - labels))
    return loss**0.5



def masked_mae_torch(preds, labels, null_val=np.nan, mask_val=np.nan, reduce=True):
    labels, mask = _prepare_mask_torch(labels, null_val, mask_val)
    mask = _normalized_mask(mask, preds.dtype)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if reduce:
        return torch.mean(loss)
    else:
        return loss


def history_volatility_weights_torch(
    history,
    high_quantile=0.75,
    weight_strength=0.5,
    null_val=0.0,
):
    """Return per-sample/node weights from observable historical volatility.

    ``history`` is expected in the original traffic-flow scale with shape
    ``(B, T, N, C)``.  Volatility uses the first flow channel and an adjacent
    change is valid only when both historical endpoints are non-missing.
    Future labels are intentionally not accepted by this function.
    """
    if history.ndim != 4 or history.shape[1] < 2:
        raise ValueError(
            'history must have shape (B, T>=2, N, C)'
        )
    if not 0.0 < float(high_quantile) < 1.0:
        raise ValueError('high_quantile must be in the interval (0, 1)')
    if float(weight_strength) < 0.0:
        raise ValueError('weight_strength must be non-negative')

    # Weight construction is a non-learned routing rule. Detaching prevents
    # gradients from flowing through the quantile/rank operation into inputs.
    flow = history[..., 0].detach()
    flow = torch.where(
        torch.abs(flow) < 1e-4,
        torch.zeros_like(flow),
        flow,
    )
    left = flow[:, :-1]
    right = flow[:, 1:]
    if np.isnan(null_val):
        pair_valid = torch.isfinite(left) & torch.isfinite(right)
    else:
        pair_valid = left.ne(null_val) & right.ne(null_val)
        pair_valid &= torch.isfinite(left) & torch.isfinite(right)

    changes = torch.where(
        pair_valid,
        torch.abs(right - left),
        torch.zeros_like(right),
    )
    pair_count = pair_valid.sum(dim=1)
    eligible = pair_count > 0
    volatility = changes.sum(dim=1) / pair_count.clamp_min(1).to(changes.dtype)

    weights = torch.ones_like(volatility)
    if bool(eligible.any()) and float(weight_strength) > 0.0:
        # Select an exact upper fraction instead of using ``>= quantile``.
        # The latter can accidentally route far more than 25% of cases when
        # many sensors share the cutoff value.
        eligible_indices = torch.nonzero(
            eligible.reshape(-1), as_tuple=False
        ).squeeze(-1)
        eligible_values = volatility.reshape(-1)[eligible_indices]
        high_count = max(
            1,
            int(np.ceil(
                (1.0 - float(high_quantile)) * eligible_values.numel()
            )),
        )
        top_values, top_positions = torch.topk(
            eligible_values, k=high_count, largest=True, sorted=False
        )
        high_flat = torch.zeros_like(eligible.reshape(-1))
        high_flat[eligible_indices[top_positions]] = True
        high_volatility = high_flat.reshape_as(eligible)
        threshold = top_values.min()
        weights = weights + float(weight_strength) * high_volatility.to(
            weights.dtype
        )
        high_fraction = high_volatility.sum().to(weights.dtype) / (
            eligible.sum().to(weights.dtype).clamp_min(1.0)
        )
    else:
        threshold = torch.zeros((), device=history.device, dtype=history.dtype)
        high_fraction = torch.zeros(
            (), device=history.device, dtype=history.dtype
        )

    stats = {
        'threshold': threshold.detach(),
        'high_fraction': high_fraction.detach(),
        'eligible_fraction': eligible.to(weights.dtype).mean().detach(),
    }
    return weights, stats


def masked_weighted_mae_torch(
    preds,
    labels,
    sample_node_weights,
    null_val=np.nan,
    mask_val=np.nan,
):
    """Masked MAE with weights shaped ``(B, N)`` or broadcast-compatible."""
    labels, valid = _prepare_mask_torch(labels, null_val, mask_val)
    if sample_node_weights.ndim == 2:
        weights = sample_node_weights[:, None, :, None]
    else:
        weights = sample_node_weights
    try:
        weights = torch.broadcast_to(weights, preds.shape)
    except RuntimeError as exc:
        raise ValueError(
            f'weights with shape {sample_node_weights.shape} cannot broadcast '
            f'to predictions with shape {preds.shape}'
        ) from exc
    if bool((weights < 0).any()):
        raise ValueError('sample_node_weights must be non-negative')

    weighted_valid = valid.to(preds.dtype) * weights.to(preds.dtype)
    error = torch.abs(preds - labels)
    error = torch.where(torch.isnan(error), torch.zeros_like(error), error)
    denominator = weighted_valid.sum().clamp_min(
        torch.finfo(preds.dtype).eps
    )
    return (error * weighted_valid).sum() / denominator


def masked_temporal_environment_cvar_torch(
    preds,
    labels,
    teacher_preds,
    temporal_position,
    kd_alpha=0.3,
    num_environments=4,
    tail_fraction=0.5,
    null_val=0.0,
):
    """CVaR of Student risk over Train-only chronological environments."""
    if preds.shape != labels.shape or preds.shape != teacher_preds.shape:
        raise ValueError('preds, labels, and teacher_preds must have one shape')
    if preds.ndim != 4:
        raise ValueError('inputs must use (B, T, N, C) layout')
    if not 0.0 <= float(kd_alpha) <= 1.0:
        raise ValueError('kd_alpha must be in [0, 1]')
    if int(num_environments) < 2:
        raise ValueError('num_environments must be at least 2')
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError('tail_fraction must be in (0, 1]')

    position = torch.as_tensor(
        temporal_position, device=preds.device, dtype=preds.dtype
    ).reshape(-1)
    if position.numel() != preds.shape[0]:
        raise ValueError('temporal_position must contain one value per sample')
    environment = torch.floor(
        position.clamp(0.0, 1.0) * int(num_environments)
    ).to(torch.long).clamp_max(int(num_environments) - 1)

    labels, valid = _prepare_mask_torch(labels, null_val, np.nan)
    valid_float = valid.to(preds.dtype)
    supervised_error = torch.abs(preds - labels)
    teacher_error = torch.abs(preds - teacher_preds)
    supervised_error = torch.where(
        torch.isfinite(supervised_error), supervised_error,
        torch.zeros_like(supervised_error),
    )
    teacher_error = torch.where(
        torch.isfinite(teacher_error), teacher_error,
        torch.zeros_like(teacher_error),
    )
    combined_error = (
        (1.0 - float(kd_alpha)) * supervised_error
        + float(kd_alpha) * teacher_error
    )

    group_losses = []
    for group_id in range(int(num_environments)):
        group = environment == group_id
        if not bool(group.any()):
            continue
        group_valid = valid_float[group]
        denominator = group_valid.sum()
        if bool(denominator > 0):
            group_losses.append(
                (combined_error[group] * group_valid).sum()
                / denominator.clamp_min(torch.finfo(preds.dtype).eps)
            )
    if not group_losses:
        zero = preds.sum() * 0.0
        return zero, {
            'mean_environment_loss': zero.detach(),
            'tail_environment_loss': zero.detach(),
            'environment_count': torch.zeros(
                (), device=preds.device, dtype=torch.long
            ),
        }

    group_losses = torch.stack(group_losses)
    tail_count = max(
        1, int(np.ceil(float(tail_fraction) * group_losses.numel()))
    )
    tail_loss = torch.topk(
        group_losses, k=tail_count, largest=True, sorted=False
    ).values.mean()
    mean_loss = (
        (combined_error * valid_float).sum()
        / valid_float.sum().clamp_min(torch.finfo(preds.dtype).eps)
    )
    return tail_loss, {
        'mean_environment_loss': mean_loss.detach(),
        'tail_environment_loss': tail_loss.detach(),
        'environment_count': torch.as_tensor(
            group_losses.numel(), device=preds.device, dtype=torch.long
        ),
        'tail_count': torch.as_tensor(
            tail_count, device=preds.device, dtype=torch.long
        ),
    }


def masked_node_cvar_mae_torch(
    preds,
    labels,
    tail_fraction=0.1,
    node_scales=None,
    null_val=np.nan,
    mask_val=np.nan,
):
    """Return CVaR over per-node masked MAE for a training batch.

    Inputs must use ``(B, T, N, C)`` layout. A node's risk is its MAE over
    batch samples, horizons, and channels. The returned loss is the mean risk
    of the largest ``tail_fraction`` eligible nodes. Selection uses only the
    current supervised Train batch and is never used at inference.
    """
    if preds.shape != labels.shape or preds.ndim != 4:
        raise ValueError(
            'preds and labels must have matching (B, T, N, C) shapes'
        )
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError('tail_fraction must be in the interval (0, 1]')

    labels, valid = _prepare_mask_torch(labels, null_val, mask_val)
    valid_float = valid.to(preds.dtype)
    error = torch.abs(preds - labels)
    error = torch.where(torch.isfinite(error), error, torch.zeros_like(error))

    reduce_dims = (0, 1, 3)
    valid_count = valid_float.sum(dim=reduce_dims)
    eligible = valid_count > 0
    if not bool(eligible.any()):
        zero = preds.sum() * 0.0
        stats = {
            'tail_loss': zero.detach(),
            'mean_node_loss': zero.detach(),
            'tail_normalized_risk': zero.detach(),
            'tail_count': torch.zeros(
                (), device=preds.device, dtype=torch.long
            ),
            'eligible_nodes': torch.zeros(
                (), device=preds.device, dtype=torch.long
            ),
        }
        return zero, stats

    node_mae = (
        (error * valid_float).sum(dim=reduce_dims)
        / valid_count.clamp_min(torch.finfo(preds.dtype).eps)
    )
    if node_scales is None:
        risk = node_mae
        reference_scale = torch.ones(
            (), device=preds.device, dtype=preds.dtype
        )
    else:
        scales = torch.as_tensor(
            node_scales, device=preds.device, dtype=preds.dtype
        ).reshape(-1)
        if scales.numel() != node_mae.numel():
            raise ValueError(
                'node_scales length does not match the node axis: '
                f'{scales.numel()} != {node_mae.numel()}'
            )
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise ValueError('node_scales must be finite and strictly positive')
        risk = node_mae / scales
        reference_scale = scales[eligible].mean()

    eligible_risk = risk[eligible]
    tail_count = max(
        1,
        int(math.ceil(float(tail_fraction) * eligible_risk.numel())),
    )
    tail_values = torch.topk(
        eligible_risk,
        k=tail_count,
        largest=True,
        sorted=False,
    ).values
    tail_normalized_risk = tail_values.mean()
    tail_loss = tail_normalized_risk * reference_scale
    stats = {
        'tail_loss': tail_loss.detach(),
        'mean_node_loss': node_mae[eligible].mean().detach(),
        'tail_normalized_risk': tail_normalized_risk.detach(),
        'tail_count': torch.as_tensor(
            tail_count, device=preds.device, dtype=torch.long
        ),
        'eligible_nodes': eligible.sum().detach(),
    }
    return tail_loss, stats


def future_change_balance_weights_torch(
    history_raw,
    labels_raw,
    node_thresholds,
    weight_strength=0.0,
    null_val=0.0,
):
    """Build train-only sample/node weights from the supervised future path.

    The change score contains all 12 forecast transitions: the boundary from
    the last observed value to horizon 1 and the 11 adjacent future
    differences.  ``node_thresholds`` must be fitted once from the training
    split, so batch composition cannot change which examples are considered
    high-change.  This helper is for training only; validation and test losses
    remain the ordinary masked MAE.
    """
    if history_raw.ndim != 4 or labels_raw.ndim != 4:
        raise ValueError('history_raw and labels_raw must be four-dimensional')
    if history_raw.shape[0] != labels_raw.shape[0]:
        raise ValueError('history/label batch sizes do not match')
    if history_raw.shape[2] != labels_raw.shape[2]:
        raise ValueError('history/label node counts do not match')
    if labels_raw.shape[1] < 1:
        raise ValueError('labels_raw must contain at least one horizon')
    if float(weight_strength) < 0.0:
        raise ValueError('weight_strength must be non-negative')

    history_last = history_raw[:, -1, :, 0].detach()
    future = labels_raw[..., 0].detach()
    left = torch.cat([history_last[:, None], future[:, :-1]], dim=1)
    right = future
    if np.isnan(null_val):
        pair_valid = torch.isfinite(left) & torch.isfinite(right)
    else:
        pair_valid = left.ne(null_val) & right.ne(null_val)
        pair_valid &= torch.isfinite(left) & torch.isfinite(right)

    changes = torch.where(
        pair_valid,
        torch.abs(right - left),
        torch.zeros_like(right),
    )
    pair_count = pair_valid.sum(dim=1)
    eligible = pair_count > 0
    score = changes.sum(dim=1) / pair_count.clamp_min(1).to(changes.dtype)

    thresholds = torch.as_tensor(
        node_thresholds,
        device=score.device,
        dtype=score.dtype,
    ).reshape(1, -1)
    if thresholds.shape[1] != score.shape[1]:
        raise ValueError(
            'future-change threshold node count does not match labels: '
            f'{thresholds.shape[1]} != {score.shape[1]}'
        )
    high_change = eligible & (score >= thresholds)
    weights = torch.ones_like(score)
    weights = weights + float(weight_strength) * high_change.to(weights.dtype)

    eligible_count = eligible.sum().to(weights.dtype).clamp_min(1.0)
    stats = {
        'high_fraction': (
            high_change.sum().to(weights.dtype) / eligible_count
        ).detach(),
        'eligible_fraction': eligible.to(weights.dtype).mean().detach(),
        'mean_score': (
            torch.where(eligible, score, torch.zeros_like(score)).sum()
            / eligible_count
        ).detach(),
    }
    return weights, stats


def active_history_change_persistence_loss_torch(
    history_raw,
    predictions_raw,
    labels_raw,
    history_node_thresholds,
    null_val=0.0,
):
    """Match future change magnitude only where the observed history is active.

    The history-active decision uses fixed per-node thresholds fitted on the
    Train split.  The supervised target is the mean absolute change over all
    forecast transitions, including the input/output boundary.  Log1p keeps
    very large traffic jumps from dominating the auxiliary objective.  This
    helper is intended for training only and introduces no inference module.
    """
    if history_raw.ndim != 4:
        raise ValueError('history_raw must be four-dimensional')
    if predictions_raw.shape != labels_raw.shape or labels_raw.ndim != 4:
        raise ValueError(
            'predictions_raw and labels_raw must have matching 4-D shapes'
        )
    if history_raw.shape[0] != labels_raw.shape[0]:
        raise ValueError('history/label batch sizes do not match')
    if history_raw.shape[2] != labels_raw.shape[2]:
        raise ValueError('history/label node counts do not match')
    if history_raw.shape[1] < 2 or labels_raw.shape[1] < 1:
        raise ValueError(
            'persistence loss requires at least two history steps and one horizon'
        )

    history = history_raw[..., 0].detach()
    labels = labels_raw[..., 0].detach()
    predictions = predictions_raw[..., 0]

    history_left = history[:, :-1]
    history_right = history[:, 1:]
    if np.isnan(null_val):
        history_valid = (
            torch.isfinite(history_left) & torch.isfinite(history_right)
        )
    else:
        history_valid = (
            history_left.ne(null_val) & history_right.ne(null_val)
            & torch.isfinite(history_left) & torch.isfinite(history_right)
        )
    history_changes = torch.where(
        history_valid,
        torch.abs(history_right - history_left),
        torch.zeros_like(history_right),
    )
    history_count = history_valid.sum(dim=1)
    history_eligible = history_count > 0
    history_score = (
        history_changes.sum(dim=1)
        / history_count.clamp_min(1).to(history_changes.dtype)
    )

    thresholds = torch.as_tensor(
        history_node_thresholds,
        device=history_score.device,
        dtype=history_score.dtype,
    ).reshape(1, -1)
    if thresholds.shape[1] != history_score.shape[1]:
        raise ValueError(
            'history-change threshold node count does not match labels: '
            f'{thresholds.shape[1]} != {history_score.shape[1]}'
        )
    history_active = history_eligible & (history_score >= thresholds)

    last_observed = history[:, -1]
    true_left = torch.cat([last_observed[:, None], labels[:, :-1]], dim=1)
    pred_left = torch.cat([last_observed[:, None], predictions[:, :-1]], dim=1)
    if np.isnan(null_val):
        future_valid = torch.isfinite(true_left) & torch.isfinite(labels)
    else:
        future_valid = (
            true_left.ne(null_val) & labels.ne(null_val)
            & torch.isfinite(true_left) & torch.isfinite(labels)
        )
    future_count = future_valid.sum(dim=1)
    future_eligible = future_count > 0
    true_changes = torch.where(
        future_valid,
        torch.abs(labels - true_left),
        torch.zeros_like(labels),
    )
    pred_changes = torch.where(
        future_valid,
        torch.abs(predictions - pred_left),
        torch.zeros_like(predictions),
    )
    denominator = future_count.clamp_min(1).to(predictions.dtype)
    true_score = true_changes.sum(dim=1) / denominator
    predicted_score = pred_changes.sum(dim=1) / denominator

    active = history_active & future_eligible
    active_float = active.to(predictions.dtype)
    active_count = active_float.sum().clamp_min(1.0)
    score_error = torch.abs(
        torch.log1p(predicted_score.clamp_min(0.0))
        - torch.log1p(true_score.clamp_min(0.0))
    )
    loss = (score_error * active_float).sum() / active_count
    eligible_count = (
        (history_eligible & future_eligible).to(predictions.dtype)
        .sum().clamp_min(1.0)
    )
    stats = {
        'active_fraction': (active_float.sum() / eligible_count).detach(),
        'true_change': (
            (true_score * active_float).sum() / active_count
        ).detach(),
        'predicted_change': (
            (predicted_score * active_float).sum() / active_count
        ).detach(),
    }
    return loss, stats


def active_history_signed_event_loss_torch(
    history_raw,
    predictions_raw,
    labels_raw,
    history_node_thresholds,
    event_emphasis=2.0,
    null_val=0.0,
):
    """Supervise the signed forecast transition at its exact horizon.

    The loss is active only for sample/node pairs whose observed-history
    change exceeds a fixed Train-only threshold.  Each of the 12 forecast
    transitions is supervised separately, including the boundary from the
    last input step to horizon 1.  Errors are normalized by the node's Train
    threshold and true large-change transitions receive additional weight,
    so the objective preserves direction and timing instead of merely
    matching total variation.
    """
    if history_raw.ndim != 4:
        raise ValueError('history_raw must be four-dimensional')
    if predictions_raw.shape != labels_raw.shape or labels_raw.ndim != 4:
        raise ValueError(
            'predictions_raw and labels_raw must have matching 4-D shapes'
        )
    if history_raw.shape[0] != labels_raw.shape[0]:
        raise ValueError('history/label batch sizes do not match')
    if history_raw.shape[2] != labels_raw.shape[2]:
        raise ValueError('history/label node counts do not match')
    if history_raw.shape[1] < 2 or labels_raw.shape[1] < 1:
        raise ValueError(
            'signed-event loss requires at least two history steps and one horizon'
        )
    if float(event_emphasis) < 0.0:
        raise ValueError('event_emphasis must be non-negative')

    history = history_raw[..., 0].detach()
    labels = labels_raw[..., 0].detach()
    predictions = predictions_raw[..., 0]

    history_left = history[:, :-1]
    history_right = history[:, 1:]
    if np.isnan(null_val):
        history_valid = (
            torch.isfinite(history_left) & torch.isfinite(history_right)
        )
    else:
        history_valid = (
            history_left.ne(null_val) & history_right.ne(null_val)
            & torch.isfinite(history_left) & torch.isfinite(history_right)
        )
    history_changes = torch.where(
        history_valid,
        torch.abs(history_right - history_left),
        torch.zeros_like(history_right),
    )
    history_count = history_valid.sum(dim=1)
    history_eligible = history_count > 0
    history_score = (
        history_changes.sum(dim=1)
        / history_count.clamp_min(1).to(history_changes.dtype)
    )

    thresholds = torch.as_tensor(
        history_node_thresholds,
        device=history_score.device,
        dtype=history_score.dtype,
    ).reshape(1, -1)
    if thresholds.shape[1] != history_score.shape[1]:
        raise ValueError(
            'history-change threshold node count does not match labels: '
            f'{thresholds.shape[1]} != {history_score.shape[1]}'
        )
    scale = thresholds.clamp_min(1.0)
    history_active = history_eligible & (history_score >= thresholds)

    last_observed = history[:, -1]
    true_left = torch.cat([last_observed[:, None], labels[:, :-1]], dim=1)
    pred_left = torch.cat([last_observed[:, None], predictions[:, :-1]], dim=1)
    if np.isnan(null_val):
        transition_valid = torch.isfinite(true_left) & torch.isfinite(labels)
    else:
        transition_valid = (
            true_left.ne(null_val) & labels.ne(null_val)
            & torch.isfinite(true_left) & torch.isfinite(labels)
        )
    active_valid = transition_valid & history_active[:, None, :]

    true_delta = labels - true_left
    predicted_delta = predictions - pred_left
    scaled_error = (predicted_delta - true_delta) / scale[:, None, :]
    # Smooth L1 retains a stable gradient for rare very large traffic jumps.
    abs_error = torch.abs(scaled_error)
    transition_loss = torch.where(
        abs_error < 1.0,
        0.5 * abs_error.square(),
        abs_error - 0.5,
    )

    relative_true_change = (
        torch.abs(true_delta) / scale[:, None, :]
    ).clamp(max=3.0)
    transition_weight = 1.0 + float(event_emphasis) * relative_true_change
    weighted_valid = active_valid.to(predictions.dtype) * transition_weight
    denominator = weighted_valid.sum().clamp_min(1.0)
    loss = (transition_loss * weighted_valid).sum() / denominator

    valid_count = active_valid.to(predictions.dtype).sum().clamp_min(1.0)
    significant = active_valid & (
        torch.abs(true_delta) >= scale[:, None, :]
    )
    significant_count = significant.to(predictions.dtype).sum().clamp_min(1.0)
    same_direction = (
        torch.sign(predicted_delta) == torch.sign(true_delta)
    ) & significant
    eligible_pairs = (
        history_eligible.to(predictions.dtype).sum().clamp_min(1.0)
    )
    stats = {
        'active_fraction': (
            history_active.to(predictions.dtype).sum() / eligible_pairs
        ).detach(),
        'significant_transition_fraction': (
            significant.to(predictions.dtype).sum() / valid_count
        ).detach(),
        'significant_direction_accuracy': (
            same_direction.to(predictions.dtype).sum() / significant_count
        ).detach(),
        'true_abs_change': (
            (torch.abs(true_delta) * active_valid).sum() / valid_count
        ).detach(),
        'predicted_abs_change': (
            (torch.abs(predicted_delta) * active_valid).sum() / valid_count
        ).detach(),
    }
    return loss, stats


def masked_adjacent_difference_mae_torch(preds, labels, null_val=0.0):
    """MAE between adjacent horizon differences with endpoint masking.

    A future difference is valid only when both of its endpoint labels are
    valid. In particular, a true difference of zero is a legitimate target and
    must not itself be treated as missing.
    """
    if preds.shape != labels.shape:
        raise ValueError(
            f'preds/labels shape mismatch: {preds.shape} vs {labels.shape}'
        )
    if preds.ndim < 2 or preds.shape[1] < 2:
        raise ValueError(
            'adjacent difference loss requires at least two horizon steps'
        )

    if np.isnan(null_val):
        valid = (~torch.isnan(labels[:, 1:])) & (~torch.isnan(labels[:, :-1]))
    else:
        valid = (labels[:, 1:] != null_val) & (labels[:, :-1] != null_val)

    prediction_difference = preds[:, 1:] - preds[:, :-1]
    label_difference = labels[:, 1:] - labels[:, :-1]
    error = torch.abs(prediction_difference - label_difference)
    error = torch.where(valid, error, torch.zeros_like(error))
    valid_count = valid.sum().to(error.dtype).clamp_min(1.0)
    return error.sum() / valid_count


def masked_block_mean_mae_torch(
    preds,
    labels,
    block_size=3,
    null_val=0.0,
):
    """MAE between valid-element means of consecutive horizon blocks.

    Missing targets never contribute to either the predicted or target block
    mean.  A sample/node/channel block is supervised when at least one of its
    target horizons is valid.  Validation and Test metrics remain the standard
    per-horizon masked MAE; this function is training-only auxiliary
    supervision for low-frequency future structure.
    """
    if preds.shape != labels.shape:
        raise ValueError(
            f'preds/labels shape mismatch: {preds.shape} vs {labels.shape}'
        )
    if preds.ndim < 2:
        raise ValueError('block mean loss requires a horizon dimension')
    if block_size <= 0 or preds.shape[1] % block_size != 0:
        raise ValueError(
            'horizon length must be divisible by block_size'
        )

    labels, valid = _prepare_mask_torch(labels, null_val)
    valid &= torch.isfinite(labels)
    num_blocks = preds.shape[1] // block_size
    block_shape = (
        preds.shape[0], num_blocks, block_size, *preds.shape[2:]
    )
    valid = valid.reshape(block_shape)
    pred_blocks = preds.reshape(block_shape)
    label_blocks = labels.reshape(block_shape)
    valid_float = valid.to(preds.dtype)
    count = valid_float.sum(dim=2)
    block_valid = count > 0
    denominator = count.clamp_min(1.0)
    pred_mean = (
        torch.where(valid, pred_blocks, torch.zeros_like(pred_blocks))
        * valid_float
    ).sum(dim=2) / denominator
    label_mean = (
        torch.where(valid, label_blocks, torch.zeros_like(label_blocks))
        * valid_float
    ).sum(dim=2) / denominator
    error = torch.abs(pred_mean - label_mean)
    error = torch.where(block_valid, error, torch.zeros_like(error))
    valid_count = block_valid.sum().to(error.dtype).clamp_min(1.0)
    return error.sum() / valid_count


def masked_mape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels, valid_mask = _prepare_mask_torch(labels, null_val, mask_val)
    safe_labels = torch.where(valid_mask, labels, torch.ones_like(labels))
    mask = _normalized_mask(valid_mask, preds.dtype)
    loss = torch.abs((preds - labels) / safe_labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels, mask = _prepare_mask_torch(labels, null_val, mask_val)
    mask = _normalized_mask(mask, preds.dtype)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val, mask_val=mask_val))

def metric_torch(predict, real, mask_val=-1):
    if mask_val == -1:
        nmae = mae(predict, real).item()
        nrmse = rmse(predict, real).item()
        mape = masked_mape_torch(predict, real, 0.0).item()
    elif mask_val == 0:
        nmae = masked_mae_torch(predict, real, 0.0).item()
        nrmse = masked_rmse_torch(predict, real, 0.0).item()
        mape = masked_mape_torch(predict, real, 0.0).item()
    elif mask_val > 0:
        nmae = masked_mae_torch(predict, real, 0.0, mask_val).item()
        nrmse = masked_rmse_torch(predict, real, 0.0, mask_val).item()
        mape = masked_mape_torch(predict, real, 0.0, mask_val).item()
    return nmae, mape, nrmse

def mae_np(preds,labels):
    loss = np.mean(np.abs(preds - labels))
    return loss
def mse_np(preds,labels):
    loss = np.mean(np.square(preds - labels))
    return loss
def rmse_np(preds,labels):
    loss = np.mean(np.square(preds - labels))
    return loss**0.5

def masked_mape_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        if not np.isnan(mask_val):
            mask &= np.greater_equal(y_true, mask_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

def masked_mae_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        if not np.isnan(mask_val):
            mask &= np.greater_equal(y_true, mask_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(y_pred, y_true).astype('float32'))
        mae = np.nan_to_num(mask * mae)
        return np.mean(mae)

def masked_mse_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        if not np.isnan(mask_val):
            mask &= np.greater_equal(y_true, mask_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(y_pred, y_true).astype('float32'))
        mse = np.nan_to_num(mask * mse)
        return np.mean(mse)

def masked_rmse_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan):
    return np.sqrt(masked_mse_np(y_pred=y_pred, y_true=y_true,
                                       null_val=null_val, mask_val=mask_val))


def masked_metric_np(predict, real, mask_val=-1):
    print('in metric_np MASK:',mask_val)
    if mask_val == -1:
        nmae = mae_np(predict, real).item()
        nrmse = rmse_np(predict, real).item()
        mape = masked_mape_np(predict, real, 0.0).item()
    elif mask_val == 0:
        nmae = masked_mae_np(predict, real, 0.0).item()
        nrmse = masked_rmse_np(predict, real, 0.0).item()
        mape = masked_mape_np(predict, real, 0.0).item()
    elif mask_val > 0:
        nmae = masked_mae_np(predict, real, 0.0, mask_val).item()
        nrmse = masked_rmse_np(predict, real, 0.0, mask_val).item()
        mape = masked_mape_np(predict, real, 0.0, mask_val).item()
    return nmae, mape, nrmse



class Scheduler:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True):
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch):
        return None

    def get_update_values(self, num_updates):
        return None

    def step(self, epoch, metric=None):
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates, metric=None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs


class CosineLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        self._logger = getLogger()
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            self._logger.warning("Cosine annealing scheduler will have no effect on the learning "
                                 "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))
