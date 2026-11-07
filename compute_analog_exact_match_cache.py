"""Add leakage-safe exact-history continuation means to an analog cache.

For each query/node, candidates are restricted to ``idx_train`` at the same
time of day.  A candidate is exact when at least ``min_valid_history`` paired
nonzero history values exist and their masked squared distance is zero.  The
forecast is the nonzero mean of every exact candidate continuation.  Training
queries exclude their own window; Validation/Test targets are never indexed.
"""

import argparse
import os

import numpy as np

from compute_analog_retrieval_cache import _normalize
from utils.datasets_utils import file_path_dict, load_raw_data


def _retrieve_exact_split(
    data,
    time_data,
    query_indices,
    train_indices,
    x_offsets,
    y_offsets,
    min_valid_history,
    leave_one_out,
    split_name,
):
    num_queries = len(query_indices)
    num_horizons = len(y_offsets)
    num_nodes = data.shape[1]
    forecast = np.zeros(
        (num_queries, num_horizons, num_nodes, 1), dtype=np.float32
    )
    available = np.zeros_like(forecast, dtype=np.uint8)
    query_tod = time_data[query_indices, 0]
    train_tod = time_data[train_indices, 0]

    for group_number, tod in enumerate(np.unique(query_tod), start=1):
        positions = np.flatnonzero(query_tod == tod)
        query_group = query_indices[positions]
        candidate_group = train_indices[train_tod == tod]
        query_history = data[
            query_group[:, None] + x_offsets[None, :], :, 0
        ].astype(np.float32, copy=False)
        candidate_history = data[
            candidate_group[:, None] + x_offsets[None, :], :, 0
        ].astype(np.float32, copy=False)
        query_expanded = query_history[:, None, :, :]
        candidate_expanded = candidate_history[None, :, :, :]
        valid = (query_expanded != 0) & (candidate_expanded != 0)
        difference = np.where(
            valid, query_expanded - candidate_expanded, 0.0
        )
        squared_distance = np.einsum(
            'qctn,qctn->qcn', difference, difference, optimize=True
        )
        exact = (
            (valid.sum(axis=2) >= min_valid_history)
            & (squared_distance == 0)
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
            exact[np.arange(len(query_group)), own_position, :] = False

        # Only fixed training candidate futures are read here.
        candidate_future = data[
            candidate_group[:, None] + y_offsets[None, :], :, 0
        ].astype(np.float32, copy=False).transpose(0, 2, 1)
        selected_valid = (
            exact[..., None] & (candidate_future[None, :, :, :] != 0)
        )
        count = selected_valid.sum(axis=1)
        total = np.where(
            selected_valid, candidate_future[None, :, :, :], 0.0
        ).sum(axis=1, dtype=np.float64)
        group_forecast = np.zeros_like(total, dtype=np.float64)
        np.divide(total, count, out=group_forecast, where=count > 0)
        forecast[positions, ..., 0] = group_forecast.transpose(0, 2, 1)
        available[positions, ..., 0] = (count > 0).transpose(
            0, 2, 1
        ).astype(np.uint8)

        if group_number % 48 == 0:
            print(
                f'{split_name}: processed TOD groups '
                f'{group_number}/288',
                flush=True,
            )

    if not np.isfinite(forecast).all():
        raise ValueError(f'Non-finite exact forecast for {split_name}')
    return forecast, available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='PEMS04')
    parser.add_argument('--input_length', type=int, default=12)
    parser.add_argument('--predict_length', type=int, default=12)
    parser.add_argument('--min_valid_history', type=int, default=6)
    parser.add_argument(
        '--index_scaler_path',
        default='none',
        help=(
            'Optional explicit split/index/scaler cache. When omitted, use '
            'the canonical dataset cache.'
        ),
    )
    parser.add_argument(
        '--base_cache',
        default='datasets/PEMS04/PEMS04_12to12_analog_k4.npz',
    )
    parser.add_argument(
        '--output',
        default=(
            'datasets/PEMS04/'
            'PEMS04_12to12_analog_k4_exact_mean.npz'
        ),
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Store exact_retrieval arrays as float16 to halve cache size. '
             'exact_available stays bool (only 1 byte per slot).',
    )
    args = parser.parse_args()
    if not 1 <= args.min_valid_history <= args.input_length:
        raise ValueError(
            'min_valid_history must be between 1 and input_length'
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
    with np.load(time_path, allow_pickle=False) as archive:
        time_data = archive['time']
    with np.load(index_path, allow_pickle=False) as archive:
        idx_train = archive['idx_train'].astype(np.int64)
        idx_val = archive['idx_val'].astype(np.int64)
        idx_test = archive['idx_test'].astype(np.int64)
        scaler_mean = np.array(archive['scaler_mean'], copy=True)
        scaler_std = np.array(archive['scaler_std'], copy=True)

    with np.load(args.base_cache, allow_pickle=False) as archive:
        base = {key: np.array(archive[key], copy=True) for key in archive.files}
    for name, expected in (
        ('idx_train', idx_train),
        ('idx_val', idx_val),
        ('idx_test', idx_test),
    ):
        if name not in base or not np.array_equal(base[name], expected):
            raise ValueError(f'Base cache {name} does not match active split')
    if int(np.asarray(base['train_leave_one_out']).item()) != 1:
        raise ValueError('Base cache training retrieval is not leave-one-out')

    x_offsets = np.arange(-(args.input_length - 1), 1, dtype=np.int64)
    y_offsets = np.arange(1, args.predict_length + 1, dtype=np.int64)
    exact = {}
    for split_name, indices, leave_one_out in (
        ('train', idx_train, True),
        ('val', idx_val, False),
        ('test', idx_test, False),
    ):
        print(
            f'Building exact {split_name}: queries={len(indices)}, '
            f'leave_one_out={leave_one_out}',
            flush=True,
        )
        raw, available = _retrieve_exact_split(
            data=data,
            time_data=time_data,
            query_indices=indices,
            train_indices=idx_train,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            min_valid_history=args.min_valid_history,
            leave_one_out=leave_one_out,
            split_name=split_name,
        )
        exact[f'exact_retrieval_{split_name}'] = _normalize(
            raw, scaler_mean, scaler_std
        )
        exact[f'exact_available_{split_name}'] = available
        print(
            f'exact {split_name} availability='
            f'{available.mean():.6%}',
            flush=True,
        )

    base.update(exact)
    base['exact_match_rule'] = np.asarray('masked_mse_eq_0')
    base['exact_min_valid_history'] = np.asarray(
        args.min_valid_history, dtype=np.int64
    )
    base['exact_train_leave_one_out'] = np.asarray(1, dtype=np.int64)
    if args.fp16:
        for key in list(base.keys()):
            if key.startswith('exact_retrieval_'):
                base[key] = base[key].astype(np.float16, copy=False)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + '.tmp.npz'
    np.savez_compressed(temporary_path, **base)
    os.replace(temporary_path, output_path)
    print(f'Saved exact-match analog cache: {output_path}', flush=True)


if __name__ == '__main__':
    main()
