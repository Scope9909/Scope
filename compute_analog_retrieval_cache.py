"""Build leakage-safe same-TOD analog forecasts for PEMS experiments.

Candidate histories and candidate futures always come from ``idx_train``.
Training queries use leave-one-out retrieval.  Validation/test targets are
never indexed; only their observable input histories are used as queries.
"""

import argparse
import os

import numpy as np

from analog_distance import DISTANCE_METRICS, masked_history_distance
from utils.datasets_utils import file_path_dict, load_raw_data


def _nonzero_mean(values, axis):
    valid = values != 0
    total = np.where(valid, values, 0.0).sum(axis=axis, dtype=np.float64)
    count = valid.sum(axis=axis)
    mean = np.zeros_like(total, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return mean, count


def _training_fallbacks(data, time_data, idx_train, y_offsets, num_tod):
    train_future = data[idx_train[:, None] + y_offsets[None, :], :, 0]
    global_mean, global_count = _nonzero_mean(train_future, axis=0)
    if np.any(global_count == 0):
        raise ValueError(
            'At least one horizon/node has no nonzero training target for '
            'the analog fallback'
        )

    fallback = np.empty(
        (num_tod, train_future.shape[1], train_future.shape[2]),
        dtype=np.float32,
    )
    train_tod = time_data[idx_train, 0]
    for tod in range(num_tod):
        selected = train_future[train_tod == tod]
        if len(selected) == 0:
            fallback[tod] = global_mean
            continue
        tod_mean, tod_count = _nonzero_mean(selected, axis=0)
        fallback[tod] = np.where(
            tod_count > 0, tod_mean, global_mean
        ).astype(np.float32)
    return fallback


def _retrieve_split(
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
    return_quality=False,
    aggregation='mean',
    calendar_mode='tod',
    num_tod=None,
    history_distance_metric='masked_mse',
    quality_distance_metric=None,
    external_selection=None,
):
    if aggregation not in ('mean', 'median', 'inverse_distance'):
        raise ValueError(
            f'Unsupported analog aggregation: {aggregation!r}'
        )
    if calendar_mode not in ('tod', 'tod_dow'):
        raise ValueError(
            f'Unsupported calendar_mode: {calendar_mode!r}'
        )
    if calendar_mode == 'tod_dow' and (
        num_tod is None or int(num_tod) <= 0
    ):
        raise ValueError('num_tod is required for calendar_mode=tod_dow')
    if quality_distance_metric is None:
        quality_distance_metric = history_distance_metric
    num_queries = len(query_indices)
    horizon = len(y_offsets)
    num_nodes = data.shape[1]
    result = np.empty(
        (num_queries, horizon, num_nodes, 1), dtype=np.float32
    )
    if return_quality:
        history_distance = np.empty(
            (num_queries, num_nodes), dtype=np.float32
        )
        future_dispersion = np.empty(
            (num_queries, horizon, num_nodes, 1), dtype=np.float32
        )
    query_tod = time_data[query_indices, 0].astype(np.int64)
    train_tod = time_data[train_indices, 0].astype(np.int64)
    if calendar_mode == 'tod':
        query_calendar = query_tod
        train_calendar = train_tod
    else:
        query_dow = time_data[query_indices, 1].astype(np.int64)
        train_dow = time_data[train_indices, 1].astype(np.int64)
        query_calendar = query_tod + int(num_tod) * query_dow
        train_calendar = train_tod + int(num_tod) * train_dow

    for calendar_key in np.unique(query_calendar):
        query_positions = np.flatnonzero(query_calendar == calendar_key)
        query_group = query_indices[query_positions]
        candidate_group = train_indices[train_calendar == calendar_key]
        tod = (
            int(calendar_key)
            if calendar_mode == 'tod'
            else int(calendar_key) % int(num_tod)
        )
        minimum_candidates = k + (1 if leave_one_out else 0)
        if len(candidate_group) < minimum_candidates:
            raise ValueError(
                f'calendar_key={int(calendar_key)} '
                f'(mode={calendar_mode}) has {len(candidate_group)} '
                f'training candidates; '
                f'{minimum_candidates} are required for {split_name}'
            )

        # These are the only query values read for validation/test.
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
        quality_distance = distance
        if return_quality and quality_distance_metric != history_distance_metric:
            quality_distance = masked_history_distance(
                query_history,
                candidate_history,
                min_valid_history=min_valid_history,
                metric=quality_distance_metric,
            )

        if leave_one_out:
            candidate_position = {
                int(index): position
                for position, index in enumerate(candidate_group)
            }
            own_position = np.asarray(
                [candidate_position[int(index)] for index in query_group],
                dtype=np.int64,
            )
            distance[
                np.arange(len(query_group)), own_position, :
            ] = np.inf
            if quality_distance is not distance:
                quality_distance[
                    np.arange(len(query_group)), own_position, :
                ] = np.inf

        if external_selection is not None:
            # The candidate set was chosen elsewhere (for example by the MMR
            # diversity selector).  Map its train endpoints back to positions
            # inside this calendar group so the mean, the quality distance and
            # the dispersion are all built from exactly the same candidates.
            chosen = external_selection[query_positions]
            position_of_index = np.full(
                data.shape[0], -1, dtype=np.int64
            )
            position_of_index[candidate_group] = np.arange(
                len(candidate_group), dtype=np.int64
            )
            nearest = np.where(
                chosen >= 0,
                position_of_index[np.clip(chosen, 0, data.shape[0] - 1)],
                -1,
            )
            available = nearest >= 0
            nearest_distance = np.take_along_axis(
                distance, np.clip(nearest, 0, len(candidate_group) - 1), axis=1
            )
            nearest_distance = np.where(
                available, nearest_distance, np.inf
            )
            nearest = np.where(available, nearest, 0)
        else:
            nearest = np.argpartition(
                distance, kth=k - 1, axis=1
            )[:, :k, :]
            nearest_distance = np.take_along_axis(
                distance, nearest, axis=1
            )
        if return_quality:
            nearest_quality_distance = np.take_along_axis(
                quality_distance, nearest, axis=1
            )
            finite_distance = (
                np.isfinite(nearest_distance)
                & np.isfinite(nearest_quality_distance)
            )
            distance_count = finite_distance.sum(axis=1)
            distance_mean = np.zeros(
                (len(query_group), num_nodes), dtype=np.float64
            )
            np.divide(
                np.where(
                    finite_distance, nearest_quality_distance, 0.0
                ).sum(axis=1, dtype=np.float64),
                distance_count,
                out=distance_mean,
                where=distance_count > 0,
            )
            distance_fallback = np.where(
                finite_distance, nearest_quality_distance, 0.0
            ).max(axis=(0, 1))
            distance_mean = np.where(
                distance_count > 0,
                distance_mean,
                distance_fallback[None, :],
            )
            history_distance[query_positions] = distance_mean.astype(
                np.float32
            )

        # Only training candidate futures are indexed here.  Shape after
        # advanced indexing: (query, K, node, horizon).
        candidate_future = data[
            candidate_group[:, None] + y_offsets[None, :], :, 0
        ].astype(np.float32, copy=False)
        candidate_future = candidate_future.transpose(0, 2, 1)
        node_index = np.arange(num_nodes)[None, None, :]
        selected_future = candidate_future[nearest, node_index, :]
        selected_valid = (
            np.isfinite(nearest_distance)[..., None]
            & (selected_future != 0)
        )
        selected_count = selected_valid.sum(axis=1)
        if aggregation == 'mean':
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
        elif aggregation == 'median':
            # Exact nonzero median for K candidates without NaN warnings.
            # Invalid/missing futures sort to the end and are excluded using
            # the valid count.  For even counts this averages the two middle
            # values, matching the standard sample median.
            sorted_future = np.sort(
                np.where(selected_valid, selected_future, np.inf),
                axis=1,
            )
            lower_position = np.maximum(
                (selected_count - 1) // 2, 0
            )
            upper_position = np.maximum(selected_count // 2, 0)
            lower = np.take_along_axis(
                sorted_future, lower_position[:, None, :, :], axis=1
            ).squeeze(1)
            upper = np.take_along_axis(
                sorted_future, upper_position[:, None, :, :], axis=1
            ).squeeze(1)
            retrieved = 0.5 * (
                lower.astype(np.float64) + upper.astype(np.float64)
            )
        else:
            # Scale-free within each query/node: only relative distances among
            # the fixed K candidates matter after normalization.  A tiny
            # epsilon handles exact history matches deterministically.
            inverse_weight = np.where(
                np.isfinite(nearest_distance),
                1.0 / (nearest_distance.astype(np.float64) + 1e-6),
                0.0,
            )
            future_weight = (
                inverse_weight[..., None]
                * selected_valid.astype(np.float64)
            )
            weight_sum = future_weight.sum(axis=1, dtype=np.float64)
            retrieved = np.zeros_like(weight_sum, dtype=np.float64)
            np.divide(
                (
                    future_weight
                    * selected_future.astype(np.float64)
                ).sum(axis=1, dtype=np.float64),
                weight_sum,
                out=retrieved,
                where=weight_sum > 0,
            )
        fallback_node_horizon = fallback[int(tod)].T
        retrieved = np.where(
            selected_count > 0,
            retrieved,
            fallback_node_horizon[None, :, :],
        )
        result[query_positions, :, :, 0] = retrieved.transpose(0, 2, 1)
        if return_quality:
            centered = selected_future - retrieved[:, None, :, :]
            dispersion_variance = np.zeros_like(
                retrieved, dtype=np.float64
            )
            np.divide(
                np.where(
                    selected_valid, centered * centered, 0.0
                ).sum(axis=1, dtype=np.float64),
                selected_count,
                out=dispersion_variance,
                where=selected_count > 1,
            )
            dispersion = np.sqrt(dispersion_variance)
            dispersion_fallback = dispersion.max(axis=0)
            dispersion = np.where(
                selected_count > 1,
                dispersion,
                dispersion_fallback[None, :, :],
            )
            future_dispersion[query_positions, :, :, 0] = (
                dispersion.transpose(0, 2, 1).astype(np.float32)
            )

    if not np.isfinite(result).all():
        raise ValueError(f'Non-finite values produced for {split_name}')
    if return_quality:
        if not (
            np.isfinite(history_distance).all()
            and np.isfinite(future_dispersion).all()
        ):
            raise ValueError(
                f'Non-finite quality features produced for {split_name}'
            )
        return result, history_distance, future_dispersion
    return result


def _normalize(values, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    if np.any(std <= 0):
        raise ValueError('Scaler standard deviation must be positive')
    if mean.size == 1:
        return ((values - float(mean.reshape(-1)[0])) /
                float(std.reshape(-1)[0])).astype(np.float32)
    node_mean = mean.reshape(1, 1, -1, 1)
    node_std = std.reshape(1, 1, -1, 1)
    return ((values - node_mean) / node_std).astype(np.float32)


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
            'Observable history distance used to select train-only analogs. '
            'The default reproduces the canonical masked-MSE cache.'
        ),
    )
    parser.add_argument(
        '--quality_distance_metric',
        choices=('same',) + DISTANCE_METRICS,
        default='same',
        help=(
            'Distance semantics stored for the frozen confidence gate. '
            'Use masked_mse when an alternative metric selects candidates '
            'but the gate must retain its training-time feature semantics.'
        ),
    )
    parser.add_argument(
        '--candidate_index_cache',
        default='none',
        help=(
            'Optional candidate endpoint index cache produced by '
            'compute_analog_candidate_index_cache.py. When given, the '
            'candidate set comes from that cache instead of the internal '
            'nearest-K rule, so the mean, quality distance and dispersion all '
            'describe the same candidates.'
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
        '--aggregation',
        choices=('mean', 'median', 'inverse_distance'),
        default='mean',
        help=(
            'Aggregate the fixed K train-only candidate continuations by an '
            'unweighted mean, sample median, or normalized inverse history '
            'distance.'
        ),
    )
    parser.add_argument(
        '--nodewise_scaler',
        action='store_true',
        help=(
            'Read the train-only per-node scaler index cache instead of the '
            'global-scaler cache. The raw analog retrieval rule is unchanged; '
            'only the stored retrieval scale follows the active model scale.'
        ),
    )
    parser.add_argument(
        '--calendar_mode',
        choices=('tod', 'tod_dow'),
        default='tod',
        help=(
            'Candidate calendar constraint: same time-of-day only, or same '
            'time-of-day and day-of-week.'
        ),
    )
    parser.add_argument(
        '--output',
        default='datasets/PEMS04/PEMS04_12to12_analog_k4.npz',
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Store retrieval arrays as float16 to halve cache size. '
             'Predictions are in z-score space (range ~[-3, 3]), so fp16 '
             'precision is sufficient; adapters cast back to fp32 internally.',
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError('k must be positive')
    if not 1 <= args.min_valid_history <= args.input_length:
        raise ValueError(
            'min_valid_history must be between 1 and input_length'
        )

    raw_path = file_path_dict[args.dataset_name]
    data, _ = load_raw_data(raw_path)
    time_path = (
        f'datasets/{args.dataset_name}/{args.dataset_name}_time.npz'
    )
    scaler_tag = (
        'index_scaler_nodewise' if args.nodewise_scaler else 'index_scaler'
    )
    index_path = (
        args.index_scaler_path
        if args.index_scaler_path not in (None, '', 'none')
        else (
            f'datasets/{args.dataset_name}/{args.dataset_name}_'
            f'{args.input_length}to{args.predict_length}_{scaler_tag}.npz'
        )
    )
    with np.load(time_path, allow_pickle=False) as time_cache:
        time_data = time_cache['time']
    with np.load(index_path, allow_pickle=False) as index_cache:
        idx_train = index_cache['idx_train'].astype(np.int64)
        idx_val = index_cache['idx_val'].astype(np.int64)
        idx_test = index_cache['idx_test'].astype(np.int64)
        scaler_mean = np.array(index_cache['scaler_mean'], copy=True)
        scaler_std = np.array(index_cache['scaler_std'], copy=True)

    x_offsets = np.arange(-(args.input_length - 1), 1, dtype=np.int64)
    y_offsets = np.arange(1, args.predict_length + 1, dtype=np.int64)
    num_tod = int(time_data[:, 0].max()) + 1
    fallback = _training_fallbacks(
        data, time_data, idx_train, y_offsets, num_tod
    )

    split_specs = (
        ('train', idx_train, True),
        ('val', idx_val, False),
        ('test', idx_test, False),
    )
    retrieval = {}
    retrieval_quality = {}
    external = None
    external_metadata = {}
    if args.candidate_index_cache != 'none':
        with np.load(
            args.candidate_index_cache, allow_pickle=False
        ) as external_cache:
            for metadata_key, expected in (
                ('k', args.k),
                ('input_length', args.input_length),
                ('predict_length', args.predict_length),
                ('min_valid_history', args.min_valid_history),
            ):
                actual = int(np.asarray(
                    external_cache[metadata_key]
                ).item())
                if actual != expected:
                    raise ValueError(
                        f'Candidate index cache {metadata_key}={actual}, '
                        f'expected {expected}'
                    )
            for text_key, expected_text in (
                ('history_distance_metric', args.history_distance_metric),
                ('candidates_train_only', 1),
            ):
                actual_text = np.asarray(
                    external_cache[text_key]
                ).item()
                if str(actual_text) != str(expected_text):
                    raise ValueError(
                        f'Candidate index cache {text_key}={actual_text!r}, '
                        f'expected {expected_text!r}'
                    )
            for name, indices in (
                ('train', idx_train),
                ('val', idx_val),
                ('test', idx_test),
            ):
                if not np.array_equal(
                    np.asarray(external_cache[f'idx_{name}']), indices
                ):
                    raise ValueError(
                        f'Candidate index cache idx_{name} does not match '
                        'the active split'
                    )
            external = {
                name: np.asarray(
                    external_cache[f'candidate_index_{name}'], dtype=np.int64
                )
                for name in ('train', 'val', 'test')
            }
            external_metadata['candidate_selection_strategy'] = str(
                np.asarray(
                    external_cache['candidate_selection_strategy']
                ).item()
            )
            external_metadata['candidate_order'] = str(
                np.asarray(external_cache['candidate_order']).item()
            )
            print(
                'Using external candidate set '
                f"{args.candidate_index_cache} "
                f"({external_metadata['candidate_selection_strategy']})",
                flush=True,
            )
    for split_name, query_indices, leave_one_out in split_specs:
        print(
            f'Building {split_name}: queries={len(query_indices)}, '
            f'K={args.k}, leave_one_out={leave_one_out}',
            flush=True,
        )
        raw_retrieval, history_distance, future_dispersion = _retrieve_split(
            data=data,
            time_data=time_data,
            query_indices=query_indices,
            train_indices=idx_train,
            fallback=fallback,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            k=args.k,
            min_valid_history=args.min_valid_history,
            leave_one_out=leave_one_out,
            split_name=split_name,
            return_quality=True,
            aggregation=args.aggregation,
            calendar_mode=args.calendar_mode,
            num_tod=num_tod,
            history_distance_metric=args.history_distance_metric,
            quality_distance_metric=(
                args.history_distance_metric
                if args.quality_distance_metric == 'same'
                else args.quality_distance_metric
            ),
            external_selection=(
                None if external is None else external[split_name]
            ),
        )
        retrieval[split_name] = _normalize(
            raw_retrieval, scaler_mean, scaler_std
        )
        retrieval_quality[split_name] = (
            history_distance.astype(np.float32, copy=False),
            future_dispersion.astype(np.float32, copy=False),
        )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + '.tmp.npz'
    target_dtype = np.float16 if args.fp16 else np.float32
    np.savez_compressed(
        temporary_path,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        retrieval_train=retrieval['train'].astype(target_dtype, copy=False),
        retrieval_val=retrieval['val'].astype(target_dtype, copy=False),
        retrieval_test=retrieval['test'].astype(target_dtype, copy=False),
        history_distance_train=retrieval_quality['train'][0].astype(
            np.float32, copy=False
        ),
        history_distance_val=retrieval_quality['val'][0].astype(
            np.float32, copy=False
        ),
        history_distance_test=retrieval_quality['test'][0].astype(
            np.float32, copy=False
        ),
        future_dispersion_train=retrieval_quality['train'][1].astype(
            np.float32, copy=False
        ),
        future_dispersion_val=retrieval_quality['val'][1].astype(
            np.float32, copy=False
        ),
        future_dispersion_test=retrieval_quality['test'][1].astype(
            np.float32, copy=False
        ),
        k=np.asarray(args.k, dtype=np.int64),
        min_valid_history=np.asarray(
            args.min_valid_history, dtype=np.int64
        ),
        input_length=np.asarray(args.input_length, dtype=np.int64),
        predict_length=np.asarray(args.predict_length, dtype=np.int64),
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        scaler_mode=np.asarray(
            'nodewise_train_only' if args.nodewise_scaler else 'global_train_only'
        ),
        train_leave_one_out=np.asarray(1, dtype=np.int64),
        calendar_mode=np.asarray(args.calendar_mode),
        aggregation=np.asarray(args.aggregation),
        history_distance_metric=np.asarray(args.history_distance_metric),
        candidate_selection_strategy=np.asarray(
            external_metadata.get('candidate_selection_strategy', 'nearest')
        ),
        candidate_order=np.asarray(
            external_metadata.get(
                'candidate_order',
                args.history_distance_metric + '_ascending',
            )
        ),
        candidate_index_cache_source=np.asarray(
            str(args.candidate_index_cache)
        ),
        quality_distance_metric=np.asarray(
            args.history_distance_metric
            if args.quality_distance_metric == 'same'
            else args.quality_distance_metric
        ),
    )
    os.replace(temporary_path, output_path)
    print(f'Saved leakage-safe analog cache: {output_path}', flush=True)


if __name__ == '__main__':
    main()
