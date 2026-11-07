"""Create an isolated 70/10/20 index/scaler cache for PEMS-BAY."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.datasets_utils import file_path_dict, load_raw_data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_length', type=int, default=12)
    parser.add_argument('--predict_length', type=int, default=12)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument(
        '--output',
        default='datasets/PEMSBAY/PEMSBAY_12to12_index_scaler_712.npz',
    )
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError('train_ratio must be in (0, 1)')
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError('test_ratio must be in (0, 1)')
    if args.train_ratio + args.test_ratio >= 1.0:
        raise ValueError('train_ratio + test_ratio must be less than 1')

    data, _ = load_raw_data(file_path_dict['PEMSBAY'])
    total_length = data.shape[0]
    min_t = args.input_length - 1
    max_t = total_length - args.predict_length
    indices = np.arange(min_t, max_t, dtype=np.int64)

    split_train = round(len(indices) * args.train_ratio)
    split_test = round(len(indices) * (1.0 - args.test_ratio))
    idx_train = indices[:split_train]
    idx_val = indices[split_train:split_test]
    idx_test = indices[split_test:]

    # Match create_dataset_files(): statistics use only observations before
    # the validation boundary and use one global scalar mean/std.
    train_values = data[: idx_val[0] - args.input_length, :, 0]
    scaler_mean = np.asarray(train_values.mean())
    scaler_std = np.asarray(train_values.std())
    if float(scaler_std) <= 0.0:
        raise ValueError('scaler standard deviation must be positive')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        train_ratio=np.asarray(args.train_ratio),
        test_ratio=np.asarray(args.test_ratio),
    )
    print(
        f'Saved {output}: train={len(idx_train)}, val={len(idx_val)}, '
        f'test={len(idx_test)}, ratios={args.train_ratio:.1f}/'
        f'{1.0 - args.train_ratio - args.test_ratio:.1f}/'
        f'{args.test_ratio:.1f}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
