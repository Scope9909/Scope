"""Leakage-safe observable history distances for analog retrieval.

The functions in this module operate only on query and candidate input
histories.  They deliberately do not inspect future targets, so changing the
distance metric cannot introduce validation/test leakage.
"""

import numpy as np


DISTANCE_METRICS = (
    "masked_mse",
    "masked_mse_trend",
    "masked_mae",
    "masked_mae_trend_volatility",
)


def masked_history_distance(
    query_history,
    candidate_history,
    min_valid_history,
    metric="masked_mse",
):
    """Return pairwise distances shaped ``(query, candidate, node)``.

    Inputs are ``(query,time,node)`` and ``(candidate,time,node)``.  Missing
    values are encoded as zero and ignored.  The composite metric keeps the
    robust MAE term while adding small, scale-consistent penalties for endpoint
    trend and first-difference volatility mismatch.
    """
    if metric not in DISTANCE_METRICS:
        raise ValueError(
            f"Unsupported history distance metric {metric!r}; "
            f"choose one of {DISTANCE_METRICS}"
        )
    query = np.asarray(query_history, dtype=np.float32)
    candidate = np.asarray(candidate_history, dtype=np.float32)
    if query.ndim != 3 or candidate.ndim != 3:
        raise ValueError("History arrays must be shaped (sample,time,node)")
    if query.shape[1] != candidate.shape[1] or query.shape[2] != candidate.shape[2]:
        raise ValueError("Query and candidate history shapes are incompatible")

    query_expanded = query[:, None, :, :]
    candidate_expanded = candidate[None, :, :, :]
    valid = (query_expanded != 0) & (candidate_expanded != 0)
    difference = np.where(valid, query_expanded - candidate_expanded, 0.0)
    valid_count = valid.sum(axis=2)
    distance = np.full(valid_count.shape, np.inf, dtype=np.float32)

    if metric in ("masked_mse", "masked_mse_trend"):
        numerator = np.square(difference).sum(axis=2)
    else:
        numerator = np.abs(difference).sum(axis=2)
    np.divide(
        numerator,
        valid_count,
        out=distance,
        where=valid_count >= int(min_valid_history),
    )
    if metric in ("masked_mse", "masked_mae"):
        return distance

    if metric == "masked_mse_trend":
        # Add exactly one endpoint-trend pseudo-observation in squared-flow
        # units. Dividing by the same valid-history count gives the trend term
        # the fixed weight of one ordinary history point and avoids a tuned
        # lambda. Missing endpoints contribute no trend term.
        endpoint_valid = (
            (query_expanded[:, :, 0, :] != 0)
            & (query_expanded[:, :, -1, :] != 0)
            & (candidate_expanded[:, :, 0, :] != 0)
            & (candidate_expanded[:, :, -1, :] != 0)
        )
        trend_difference = (
            (query_expanded[:, :, -1, :] - query_expanded[:, :, 0, :])
            - (
                candidate_expanded[:, :, -1, :]
                - candidate_expanded[:, :, 0, :]
            )
        )
        trend_squared = np.where(
            endpoint_valid, np.square(trend_difference), 0.0
        )
        trend_penalty = np.zeros_like(distance, dtype=np.float32)
        np.divide(
            trend_squared,
            valid_count,
            out=trend_penalty,
            where=valid_count >= int(min_valid_history),
        )
        return np.where(
            np.isfinite(distance), distance + trend_penalty, np.inf
        ).astype(np.float32)

    # The extra terms use only observable history.  If an endpoint or an
    # adjacent pair is missing, that term contributes zero rather than making
    # an otherwise valid candidate unusable.
    endpoint_valid = (
        (query_expanded[:, :, 0, :] != 0)
        & (query_expanded[:, :, -1, :] != 0)
        & (candidate_expanded[:, :, 0, :] != 0)
        & (candidate_expanded[:, :, -1, :] != 0)
    )
    slope_difference = np.abs(
        (query_expanded[:, :, -1, :] - query_expanded[:, :, 0, :])
        - (candidate_expanded[:, :, -1, :] - candidate_expanded[:, :, 0, :])
    )
    slope_difference = np.where(endpoint_valid, slope_difference, 0.0)

    consecutive_valid = valid[:, :, 1:, :] & valid[:, :, :-1, :]
    consecutive_count = consecutive_valid.sum(axis=2)
    query_delta = np.abs(query_expanded[:, :, 1:, :] - query_expanded[:, :, :-1, :])
    candidate_delta = np.abs(
        candidate_expanded[:, :, 1:, :] - candidate_expanded[:, :, :-1, :]
    )
    volatility_difference = np.abs(
        np.where(consecutive_valid, query_delta, 0.0).sum(axis=2)
        / np.maximum(consecutive_count, 1)
        - np.where(consecutive_valid, candidate_delta, 0.0).sum(axis=2)
        / np.maximum(consecutive_count, 1)
    )
    composite = distance + 0.25 * slope_difference + 0.25 * volatility_difference
    return np.where(np.isfinite(distance), composite, np.inf).astype(np.float32)
