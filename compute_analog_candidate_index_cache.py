"""Build leakage-safe fixed-K analog candidate endpoint indices.

Only observable input histories are used to choose candidates.  Every
candidate endpoint belongs to ``idx_train``; training queries use leave-one-out
selection.  Validation and test targets are never read.  Candidate futures are
materialized later by the dataloader from these train-only endpoint indices.
"""

import argparse
import os

import numpy as np

from compute_analog_retrieval_cache import _training_fallbacks
from analog_distance import DISTANCE_METRICS, masked_history_distance
from utils.datasets_utils import file_path_dict, load_raw_data


def _normalize(values, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    if mean.size == 1:
        return (
            (values - float(mean.reshape(-1)[0]))
            / float(std.reshape(-1)[0])
        ).astype(np.float32)
    return (
        (values - mean.reshape(1, 1, -1, 1))
        / std.reshape(1, 1, -1, 1)
    ).astype(np.float32)


def _greedy_mmr_pool(dq, pair, k, lam):
    """Greedy MMR over a batched candidate pool.

    ``dq`` is ``(batch, pool, node)`` and ``pair`` is ``(batch, pool, pool,
    node)``.  Returns the selected pool positions in selection order, shaped
    ``(batch, k, node)`` with ``-1`` where fewer than ``k`` candidates remain.
    """
    batch, pool_size, num_nodes = dq.shape
    selected = np.full((batch, k, num_nodes), -1, dtype=np.int64)
    chosen = np.zeros((batch, pool_size, num_nodes), dtype=bool)
    batch_axis = np.arange(batch, dtype=np.int64)[:, None, None, None]
    pool_axis = np.arange(pool_size, dtype=np.int64)[None, :, None, None]
    node_axis = np.arange(num_nodes, dtype=np.int64)[None, None, None, :]

    for step in range(k):
        # Relevance term: a smaller observable history distance is better.
        score = -dq.copy()
        if step > 0:
            prior = selected[:, :step, :]
            usable = prior >= 0
            picked = pair[
                batch_axis,
                pool_axis,
                np.clip(prior, 0, pool_size - 1)[:, None, :, :],
                node_axis,
            ]
            picked = np.where(usable[:, None, :, :], picked, np.inf)
            nearest = np.min(picked, axis=2)
            score = score + lam * np.where(np.isfinite(nearest), nearest, 0.0)
        score = np.where(chosen, -np.inf, score)
        score = np.where(np.isfinite(score), score, -np.inf)
        best = np.argmax(score, axis=1)
        best_score = np.take_along_axis(
            score, best[:, None, :], axis=1
        )[:, 0, :]
        usable_best = np.isfinite(best_score)
        best = np.where(usable_best, best, -1)
        selected[:, step, :] = best
        rows, nodes = np.nonzero(usable_best)
        chosen[rows, best[rows, nodes], nodes] = True
    return selected


def _mmr_select(
    data,
    x_offsets,
    candidate_group,
    distance,
    k,
    pool_size,
    lam,
    min_valid_history,
):
    """Select K candidates by greedy maximal marginal relevance.

    ``distance`` is ``(query, candidate, node)`` in observable history space and
    is already leave-one-out adjusted by the caller.  For every query the
    nearest ``pool_size`` candidates form a pool, then K are chosen greedily as

        ``score(j) = -d(query, j) + lambda * min_{i in S} d(j, i)``

    Returns positions into the candidate axis, relevance-ordered (ascending
    history distance), with ``-1`` where no candidate is available.
    """
    num_queries, num_candidates, num_nodes = distance.shape
    pool_size = int(min(pool_size, num_candidates))
    if pool_size < k:
        raise ValueError(
            f'MMR pool {pool_size} is smaller than K={k}'
        )
    pool = np.argpartition(
        distance, kth=pool_size - 1, axis=1
    )[:, :pool_size, :]
    pool_distance = np.take_along_axis(distance, pool, axis=1)
    # PEMS flow values are integer-valued, so many candidate histories are
    # exactly equidistant.  argpartition breaks those ties arbitrarily, which
    # would make the greedy MMR trajectory depend on the tie order.  Order the
    # pool by (distance, candidate endpoint) so the selection is a
    # deterministic function of the data alone.
    pool_order = np.lexsort(
        np.stack([candidate_group[pool], pool_distance], axis=0), axis=1
    )
    pool = np.take_along_axis(pool, pool_order, axis=1)
    pool_distance = np.take_along_axis(pool_distance, pool_order, axis=1)
    pool_available = np.isfinite(pool_distance)

    selected = np.full((num_queries, k, num_nodes), -1, dtype=np.int64)
    node_axis = np.arange(num_nodes, dtype=np.int64)
    horizon = len(x_offsets)
    # Bound the (batch, pool, pool, time, node) intermediate.
    per_query = max(pool_size * pool_size * horizon * num_nodes, 1)
    chunk = max(1, int(2.4e7 // per_query))

    for start in range(0, num_queries, chunk):
        stop = min(start + chunk, num_queries)
        pool_index = pool[start:stop]
        pool_valid = pool_available[start:stop]
        positions = (
            candidate_group[pool_index][:, :, None, :]
            + x_offsets[None, None, :, None]
        )
        history = data[
            positions, node_axis[None, None, None, :], 0
        ].astype(np.float32, copy=False)
        observed = history != 0
        pair_valid = (
            observed[:, :, None, :, :] & observed[:, None, :, :, :]
        )
        difference = np.where(
            pair_valid,
            history[:, :, None, :, :] - history[:, None, :, :, :],
            0.0,
        )
        count = pair_valid.sum(axis=3)
        numerator = np.square(difference).sum(axis=3)
        pair = np.full(numerator.shape, np.inf, dtype=np.float32)
        np.divide(
            numerator,
            count,
            out=pair,
            where=count >= int(min_valid_history),
        )
        pair = np.where(
            pool_valid[:, :, None, :] & pool_valid[:, None, :, :],
            pair,
            np.inf,
        )
        selected[start:stop] = _greedy_mmr_pool(
            pool_distance[start:stop], pair, k, lam
        )

    keep = selected >= 0
    clipped = np.clip(selected, 0, pool_size - 1)
    mapped = np.where(keep, np.take_along_axis(pool, clipped, axis=1), -1)
    gathered = np.take_along_axis(pool_distance, clipped, axis=1)
    gathered = np.where(keep, gathered, np.inf)
    relevance_order = np.argsort(gathered, axis=1, kind='stable')
    return np.take_along_axis(mapped, relevance_order, axis=1)


def _select_split(
    data,
    time_data,
    query_indices,
    train_indices,
    fallback,
    x_offsets,
    y_offsets,
    k,
    min_valid_history,
    leave_one_out,
    split_name,
    scaler_mean,
    scaler_std,
    reference_retrieval,
    history_distance_metric,
    selection_strategy,
    mmr_lambda=0.5,
    mmr_pool=32,
):
    num_queries = len(query_indices)
    num_nodes = data.shape[1]
    candidate_index = np.full(
        (num_queries, k, num_nodes), -1, dtype=np.int32
    )
    query_tod = time_data[query_indices, 0].astype(np.int64)
    train_tod = time_data[train_indices, 0].astype(np.int64)
    audit_reference = (
        selection_strategy == 'nearest'
        and reference_retrieval is not None
    )
    maximum_reference_error = 0.0

    for tod in np.unique(query_tod):
        query_positions = np.flatnonzero(query_tod == tod)
        query_group = query_indices[query_positions]
        candidate_group = train_indices[train_tod == tod]
        minimum_candidates = k + (1 if leave_one_out else 0)
        if len(candidate_group) < minimum_candidates:
            raise ValueError(
                f'TOD={int(tod)} has {len(candidate_group)} train '
                f'candidates; {minimum_candidates} are required'
            )

        query_history = data[
            query_group[:, None] + x_offsets[None, :], :, 0
        ].astype(np.float32, copy=False)
        candidate_history = data[
            candidate_group[:, None] + x_offsets[None, :], :, 0
        ].astype(np.float32, copy=False)
        distance = masked_history_distance(
            query_history,
            candidate_history,
            min_valid_history=min_valid_history,
            metric=history_distance_metric,
        )

        if leave_one_out:
            candidate_position = {
                int(index): position
                for position, index in enumerate(candidate_group)
            }
            own_position = np.asarray([
                candidate_position[int(index)] for index in query_group
            ], dtype=np.int64)
            distance[
                np.arange(len(query_group)), own_position, :
            ] = np.inf

        if selection_strategy == 'mmr':
            # Diversity-aware selection: keep the deployed K unchanged, but
            # choose it from a deeper nearest-neighbour pool so the eight
            # histories are not near-duplicates of one another.
            nearest = _mmr_select(
                data,
                x_offsets,
                candidate_group,
                distance,
                k=k,
                pool_size=mmr_pool,
                lam=mmr_lambda,
                min_valid_history=min_valid_history,
            )
            nearest_distance = np.take_along_axis(distance, nearest, axis=1)
            selected_index = candidate_group[nearest]
            finite = np.isfinite(nearest_distance)
            selected_index = np.where(finite, selected_index, -1)
            candidate_index[query_positions] = selected_index.astype(
                np.int32, copy=False
            )
            continue

        shortlist_size = k
        if selection_strategy == 'rank_spread':
            shortlist_size = min(distance.shape[1], max(4 * k, k))
        nearest = np.argpartition(
            distance, kth=shortlist_size - 1, axis=1
        )[:, :shortlist_size, :]
        nearest_distance = np.take_along_axis(distance, nearest, axis=1)
        # A stable ascending order makes the eighth causal feature ("rank")
        # meaningful.  Sorting does not change the K-set or its uniform mean.
        order = np.argsort(
            nearest_distance, axis=1, kind='stable'
        )
        nearest = np.take_along_axis(nearest, order, axis=1)
        nearest_distance = np.take_along_axis(
            nearest_distance, order, axis=1
        )
        if selection_strategy == 'rank_spread' and shortlist_size > k:
            # Select evenly spaced ranks from a larger nearest-neighbour
            # shortlist. This is deterministic, target-free, and avoids
            # returning eight almost-duplicate histories without changing K.
            spread_positions = np.rint(np.linspace(
                0, shortlist_size - 1, k, dtype=np.float32
            )).astype(np.int64)
            nearest = nearest[:, spread_positions, :]
            nearest_distance = nearest_distance[:, spread_positions, :]
        selected_index = candidate_group[nearest]
        finite = np.isfinite(nearest_distance)
        selected_index = np.where(finite, selected_index, -1)
        candidate_index[query_positions] = selected_index.astype(
            np.int32, copy=False
        )

        if audit_reference:
            # Verify that the endpoint cache and the matching retrieval cache
            # materialize exactly the same K-candidate mean. This audit is
            # metric-agnostic as long as both caches declare the same rule.
            candidate_future = data[
                candidate_group[:, None] + y_offsets[None, :], :, 0
            ].astype(np.float32, copy=False).transpose(0, 2, 1)
            node_index = np.arange(num_nodes)[None, None, :]
            selected_future = candidate_future[nearest, node_index, :]
            selected_valid = finite[..., None] & (selected_future != 0)
            selected_count = selected_valid.sum(axis=1)
            selected_sum = np.where(
                selected_valid, selected_future, 0.0
            ).sum(axis=1, dtype=np.float64)
            retrieved = np.zeros_like(selected_sum, dtype=np.float64)
            np.divide(
                selected_sum,
                selected_count,
                out=retrieved,
                where=selected_count > 0,
            )
            fallback_node_horizon = fallback[int(tod)].T
            retrieved = np.where(
                selected_count > 0,
                retrieved,
                fallback_node_horizon[None, :, :],
            )
            retrieved = retrieved.transpose(0, 2, 1)[..., None]
            normalized = _normalize(retrieved, scaler_mean, scaler_std)
            group_error = float(np.max(np.abs(
                normalized - reference_retrieval[query_positions]
            )))
            maximum_reference_error = max(maximum_reference_error, group_error)

    valid = candidate_index >= 0
    if np.any(candidate_index[valid] >= len(data)):
        raise ValueError(f'{split_name} contains an invalid endpoint')
    if split_name == 'train' and np.any(
        candidate_index == query_indices[:, None, None]
    ):
        raise ValueError('Training candidate selection is not leave-one-out')
    if audit_reference and maximum_reference_error > 1e-5:
        raise RuntimeError(
            f'{split_name} candidates do not reproduce the formal K={k} '
            f'mean cache; max normalized error={maximum_reference_error:.8g}'
        )
    return candidate_index, maximum_reference_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='PEMS04')
    parser.add_argument('--input_length', type=int, default=12)
    parser.add_argument('--predict_length', type=int, default=12)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--min_valid_history', type=int, default=6)
    parser.add_argument(
        '--history_distance_metric',
        choices=DISTANCE_METRICS,
        default='masked_mse',
        help=(
            'Observable history distance used to rank the fixed K candidates. '
            'The default reproduces the canonical masked-MSE cache.'
        ),
    )
    parser.add_argument(
        '--selection_strategy',
        choices=('nearest', 'rank_spread', 'mmr'),
        default='nearest',
        help=(
            'Candidate selection strategy. rank_spread selects K ranks from '
            'a 4K nearest-neighbour shortlist while keeping K fixed. mmr '
            'selects K by greedy maximal marginal relevance over a deeper '
            'pool, also keeping K fixed.'
        ),
    )
    parser.add_argument(
        '--mmr_lambda',
        type=float,
        default=0.5,
        help=(
            'Diversity weight for --selection_strategy mmr. Both MMR terms '
            'use the same observable history distance, so lambda is '
            'dimensionless.'
        ),
    )
    parser.add_argument(
        '--mmr_pool',
        type=int,
        default=32,
        help='Nearest-neighbour pool depth that MMR selects K candidates from.',
    )
    parser.add_argument(
        '--selection_strategy_tag',
        default=None,
        help=(
            'Optional explicit tag recorded as candidate_selection_strategy. '
            'Use it to make the lambda part of the cache identity so a '
            'mismatched configuration cannot silently reuse the cache.'
        ),
    )
    parser.add_argument(
        '--index_scaler_path',
        default='none',
        help=(
            'Optional explicit split/index/scaler cache. When omitted, use '
            'the canonical dataset cache.'
        ),
    )
    parser.add_argument(
        '--reference_cache',
        default='datasets/PEMS04/PEMS04_12to12_analog_k4_exact_mean.npz',
        help=(
            'Mean-retrieval cache used to audit nearest selections. Pass '
            '"none" to skip the audit when building a pool cache whose K '
            'does not match any existing mean cache (e.g. K=32 pools).'
        ),
    )
    parser.add_argument(
        '--output',
        default=(
            'datasets/PEMS04/'
            'PEMS04_12to12_analog_k4_candidate_indices.npz'
        ),
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError('k must be positive')
    if not 1 <= args.min_valid_history <= args.input_length:
        raise ValueError(
            'min_valid_history must be in [1, input_length]'
        )

    data, _ = load_raw_data(file_path_dict[args.dataset_name])
    time_path = (
        f'datasets/{args.dataset_name}/{args.dataset_name}_time.npz'
    )
    index_path = (
        args.index_scaler_path
        if args.index_scaler_path not in (None, '', 'none')
        else (
            f'datasets/{args.dataset_name}/{args.dataset_name}_'
            f'{args.input_length}to{args.predict_length}_index_scaler.npz'
        )
    )
    with np.load(time_path, allow_pickle=False) as time_cache:
        time_data = np.asarray(time_cache['time'])
    with np.load(index_path, allow_pickle=False) as index_cache:
        idx_train = np.asarray(index_cache['idx_train'], dtype=np.int64)
        idx_val = np.asarray(index_cache['idx_val'], dtype=np.int64)
        idx_test = np.asarray(index_cache['idx_test'], dtype=np.int64)
        scaler_mean = np.array(index_cache['scaler_mean'], copy=True)
        scaler_std = np.array(index_cache['scaler_std'], copy=True)
    if np.any(np.asarray(scaler_std) <= 0):
        raise ValueError('Scaler standard deviation must be positive')

    x_offsets = np.arange(
        -(args.input_length - 1), 1, dtype=np.int64
    )
    y_offsets = np.arange(
        1, args.predict_length + 1, dtype=np.int64
    )
    num_tod = int(time_data[:, 0].max()) + 1
    fallback = _training_fallbacks(
        data, time_data, idx_train, y_offsets, num_tod
    )

    reference = None
    if args.reference_cache.strip().lower() != 'none':
        reference_cache = np.load(
            args.reference_cache, allow_pickle=False
        )
        for metadata_key, expected in (
            ('k', args.k),
            ('input_length', args.input_length),
            ('predict_length', args.predict_length),
            ('train_leave_one_out', 1),
        ):
            actual = int(np.asarray(
                reference_cache[metadata_key]
            ).item())
            if actual != expected:
                raise ValueError(
                    f'Reference cache {metadata_key}={actual}, '
                    f'expected {expected}'
                )
        aggregation = (
            str(np.asarray(reference_cache['aggregation']).item())
            if 'aggregation' in reference_cache
            else 'mean'
        )
        if aggregation != 'mean':
            raise ValueError('Reference cache aggregation must be mean')
        reference_metric = (
            str(np.asarray(
                reference_cache['history_distance_metric']
            ).item())
            if 'history_distance_metric' in reference_cache
            else 'masked_mse'
        )
        if reference_metric != args.history_distance_metric:
            raise ValueError(
                'Reference cache history_distance_metric='
                f'{reference_metric!r}, expected '
                f'{args.history_distance_metric!r}'
            )
        if not (
            np.allclose(
                reference_cache['scaler_mean'],
                scaler_mean,
                rtol=0.0,
                atol=1e-7,
            )
            and np.allclose(
                reference_cache['scaler_std'],
                scaler_std,
                rtol=0.0,
                atol=1e-7,
            )
        ):
            raise ValueError('Reference cache scaler mismatch')
        reference = {
            split: np.asarray(
                reference_cache[f'retrieval_{split}'],
                dtype=np.float32,
            )
            for split in ('train', 'val', 'test')
        }
        reference_cache.close()

    selected = {}
    audit_error = {}
    for split_name, query_index, leave_one_out in (
        ('train', idx_train, True),
        ('val', idx_val, False),
        ('test', idx_test, False),
    ):
        print(
            f'Building {split_name}: queries={len(query_index)}, '
            f'K={args.k}, leave_one_out={leave_one_out}',
            flush=True,
        )
        selected[split_name], audit_error[split_name] = _select_split(
            data=data,
            time_data=time_data,
            query_indices=query_index,
            train_indices=idx_train,
            fallback=fallback,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            k=args.k,
            min_valid_history=args.min_valid_history,
            leave_one_out=leave_one_out,
            split_name=split_name,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            reference_retrieval=(
                reference[split_name] if reference is not None else None
            ),
            history_distance_metric=args.history_distance_metric,
            selection_strategy=args.selection_strategy,
            mmr_lambda=args.mmr_lambda,
            mmr_pool=args.mmr_pool,
        )
        if args.selection_strategy == 'nearest':
            print(
                f'{split_name} uniform-mean audit max_abs='
                f'{audit_error[split_name]:.8g}',
                flush=True,
            )
        else:
            print(
                f'{split_name} alternative-metric candidate set built '
                '(canonical-mean audit skipped)',
                flush=True,
            )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + '.tmp.npz'
    strategy_tag = (
        args.selection_strategy_tag
        or (
            f'mmr_lam{args.mmr_lambda:g}_pool{args.mmr_pool}'
            if args.selection_strategy == 'mmr'
            else args.selection_strategy
        )
    )
    np.savez_compressed(
        temporary_path,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        candidate_index_train=selected['train'],
        candidate_index_val=selected['val'],
        candidate_index_test=selected['test'],
        k=np.asarray(args.k, dtype=np.int64),
        min_valid_history=np.asarray(
            args.min_valid_history, dtype=np.int64
        ),
        input_length=np.asarray(args.input_length, dtype=np.int64),
        predict_length=np.asarray(args.predict_length, dtype=np.int64),
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        train_leave_one_out=np.asarray(1, dtype=np.int64),
        candidates_train_only=np.asarray(1, dtype=np.int64),
        candidate_cache_type=np.asarray('train_endpoint_indices'),
        calendar_mode=np.asarray('tod'),
        history_distance_metric=np.asarray(args.history_distance_metric),
        candidate_selection_strategy=np.asarray(strategy_tag),
        mmr_lambda=np.asarray(args.mmr_lambda, dtype=np.float64),
        mmr_pool=np.asarray(args.mmr_pool, dtype=np.int64),
        retrieval_rule=np.asarray(
            'masked_history_' + args.history_distance_metric.removeprefix('masked_')
        ),
        candidate_order=np.asarray(
            args.history_distance_metric + (
                '_rank_spread'
                if args.selection_strategy == 'rank_spread'
                else '_ascending'
            )
        ),
        feature_names=np.asarray([
            'log1p_masked_mse_z',
            'log1p_masked_mae_z',
            'tanh_last_difference_z',
            'tanh_mean_difference_z',
            'tanh_slope_difference_z',
            'tanh_volatility_difference_z',
            'valid_history_fraction',
            'candidate_rank',
        ]),
        uniform_mean_audit_train=np.asarray(
            audit_error['train'], dtype=np.float64
        ),
        uniform_mean_audit_val=np.asarray(
            audit_error['val'], dtype=np.float64
        ),
        uniform_mean_audit_test=np.asarray(
            audit_error['test'], dtype=np.float64
        ),
    )
    os.replace(temporary_path, output_path)
    print(
        'Saved leakage-safe candidate endpoint cache: '
        f'{output_path}',
        flush=True,
    )


if __name__ == '__main__':
    main()
