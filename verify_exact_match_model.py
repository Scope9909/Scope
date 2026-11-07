"""Independent pre-Test checks for the exact-history model refinement.

This script never reads Test predictions, Test targets, or Test cache arrays.
It verifies checkpoint compatibility, epoch-0 identity, unavailable-element
identity, cache recomputation, train leave-one-out, and Validation metrics.
"""

import json
from types import SimpleNamespace

import numpy as np
import torch

from compute_analog_exact_match_cache import _retrieve_exact_split
from compute_analog_retrieval_cache import _normalize
from models.TLAST_improved import Model
from utils.datasets_utils import file_path_dict, load_raw_data


def _horizon_mae(prediction, truth):
    """Historical diagnostic metric; preserve the original nonzero mask."""
    values = []
    for horizon in range(truth.shape[1]):
        target = truth[:, horizon]
        forecast = prediction[:, horizon]
        mask = target != 0
        values.append(np.abs(forecast - target)[mask].mean(dtype=np.float64))
    return np.asarray(values, dtype=np.float64)


ROOT_CONFIG = (
    'EXP_experiments/TLAST_improved/PEMS04/'
    'TLAST_improved_PEMS04_12to12_exp07181246_s1/config.json'
)
PARENT_CHECKPOINT = (
    'EXP_experiments/TLAST_improved/PEMS04/'
    'TLAST_improved_PEMS04_12to12_exp07181246_s1/best.pth'
)
PARENT_OUTPUT = (
    'EXP_experiments/TLAST_improved/PEMS04/'
    'TLAST_improved_PEMS04_12to12_exp07181246_s1/'
    'TLAST_improved_PEMS04_exp07181246_s1_output.npz'
)
CACHE = 'datasets/PEMS04/PEMS04_12to12_analog_k4_exact_mean.npz'


def _load_args(use_exact):
    with open(ROOT_CONFIG, encoding='utf-8') as handle:
        values = json.load(handle)
    values.update({
        'num_nodes': 307,
        'use_analog_exact_match_residual': int(use_exact),
        'analog_exact_match_max': 1.0,
        'analog_exact_min_valid_history': 6,
        'analog_retrieval_cache': CACHE,
        'freeze_parent_train_analog_lowrank_retrieval': 0,
        'freeze_parent_train_analog_exact_match': int(use_exact),
        'frozen_parent_ckpt': PARENT_CHECKPOINT,
    })
    return SimpleNamespace(**values)


def _make_first_validation_batch(batch_size=2):
    data, _ = load_raw_data(file_path_dict['PEMS04'])
    with np.load('datasets/PEMS04/PEMS04_time.npz', allow_pickle=False) as ar:
        time_data = ar['time']
    with np.load(
        'datasets/PEMS04/PEMS04_12to12_index_scaler.npz',
        allow_pickle=False,
    ) as ar:
        idx_train = ar['idx_train'].astype(np.int64)
        idx_val = ar['idx_val'].astype(np.int64)
        mean = np.array(ar['scaler_mean'], copy=True)
        std = np.array(ar['scaler_std'], copy=True)
    x_offsets = np.arange(-11, 1, dtype=np.int64)
    y_offsets = np.arange(1, 13, dtype=np.int64)
    query = idx_val[:batch_size]
    x = data[query[:, None] + x_offsets[None, :], :, :1]
    x = _normalize(x, mean, std)
    x_time = time_data[query[:, None] + x_offsets[None, :], :]
    y_time = time_data[query[:, None] + y_offsets[None, :], :]
    with np.load(CACHE, allow_pickle=False) as ar:
        analog = np.array(ar['retrieval_val'][:batch_size], copy=True)
        exact = np.array(ar['exact_retrieval_val'][:batch_size], copy=True)
        available = np.array(
            ar['exact_available_val'][:batch_size], copy=True
        ).astype(np.float32)
    return (
        data, time_data, idx_train, idx_val, mean, std,
        x_offsets, y_offsets,
        torch.from_numpy(x), torch.from_numpy(x_time).long(),
        torch.from_numpy(y_time).long(), torch.from_numpy(analog),
        torch.from_numpy(exact), torch.from_numpy(available),
    )


def main():
    torch.manual_seed(1)
    parent = Model(_load_args(False)).eval()
    parent_state = torch.load(PARENT_CHECKPOINT, map_location='cpu')
    parent.load_state_dict(parent_state, strict=True)

    torch.manual_seed(1)
    candidate = Model(_load_args(True)).eval()
    missing, unexpected = candidate.load_state_dict(
        parent_state, strict=False
    )
    expected_missing = ['analog_exact_match_residual.horizon_logits']
    if list(missing) != expected_missing or list(unexpected):
        raise RuntimeError(
            f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}'
        )

    (
        data, time_data, idx_train, idx_val, mean, std,
        x_offsets, y_offsets,
        x, x_time, y_time, analog, exact, available,
    ) = _make_first_validation_batch()
    with torch.no_grad():
        parent_prediction, _ = parent([x, x_time, y_time, analog])
        epoch0_prediction, _ = candidate([
            x, x_time, y_time, analog, exact, available
        ])
    epoch0_difference = float(
        (parent_prediction - epoch0_prediction).abs().max()
    )

    with torch.no_grad():
        candidate.analog_exact_match_residual.horizon_logits.fill_(
            float(np.arctanh(0.999))
        )
        full_prediction, _ = candidate([
            x, x_time, y_time, analog, exact, available
        ])
    unavailable = available == 0
    unavailable_difference = float(
        (parent_prediction[unavailable] - full_prediction[unavailable])
        .abs().max()
    ) if unavailable.any() else 0.0
    expected_full = parent_prediction + 0.999 * available * (
        exact - parent_prediction
    )
    formula_difference = float(
        (expected_full - full_prediction).abs().max()
    )

    # Independently recompute small Validation and train subsets.  Validation
    # uses only train candidates; train explicitly excludes its own window.
    recomputed_val_raw, recomputed_val_available = _retrieve_exact_split(
        data, time_data, idx_val[:8], idx_train, x_offsets, y_offsets,
        6, False, 'verify-val',
    )
    recomputed_train_raw, recomputed_train_available = _retrieve_exact_split(
        data, time_data, idx_train[:8], idx_train, x_offsets, y_offsets,
        6, True, 'verify-train-loo',
    )
    recomputed_val = _normalize(recomputed_val_raw, mean, std)
    recomputed_train = _normalize(recomputed_train_raw, mean, std)
    with np.load(CACHE, allow_pickle=False) as ar:
        cached_val = np.array(ar['exact_retrieval_val'][:8], copy=True)
        cached_val_available = np.array(
            ar['exact_available_val'][:8], copy=True
        )
        cached_train = np.array(ar['exact_retrieval_train'][:8], copy=True)
        cached_train_available = np.array(
            ar['exact_available_train'][:8], copy=True
        )
        # Deliberately do not access exact_retrieval_test/retrieval_test.
        exact_val = np.array(ar['exact_retrieval_val'], copy=True)
        available_val = np.array(
            ar['exact_available_val'], copy=True
        ).astype(bool)

    cache_val_difference = float(np.max(np.abs(
        recomputed_val - cached_val
    )))
    cache_train_difference = float(np.max(np.abs(
        recomputed_train - cached_train
    )))
    availability_val_equal = bool(np.array_equal(
        recomputed_val_available, cached_val_available
    ))
    availability_train_equal = bool(np.array_equal(
        recomputed_train_available, cached_train_available
    ))

    with np.load(PARENT_OUTPUT, allow_pickle=False) as ar:
        current = np.array(ar['validation_prediction'], copy=True)[..., 0]
        truth = np.array(ar['validation_truth'], copy=True)[..., 0]
        scaler_mean = float(np.asarray(ar['scaler_mean']).item())
        scaler_std = float(np.asarray(ar['scaler_std']).item())
        # Deliberately do not access prediction/truth Test keys.
    exact_val_raw = exact_val[..., 0] * scaler_std + scaler_mean
    hard_prediction = np.where(
        available_val[..., 0], exact_val_raw, current
    )
    current_h = _horizon_mae(current, truth)
    hard_h = _horizon_mae(hard_prediction, truth)

    passed = (
        epoch0_difference == 0.0
        and unavailable_difference == 0.0
        and formula_difference < 1e-6
        and cache_val_difference == 0.0
        and cache_train_difference == 0.0
        and availability_val_equal
        and availability_train_equal
        and float(current_h.mean() - hard_h.mean()) >= 0.02
    )
    print(f'checkpoint_missing={list(missing)}')
    print(f'checkpoint_unexpected={list(unexpected)}')
    print(f'epoch0_max_abs_difference={epoch0_difference}')
    print(f'unavailable_max_abs_difference={unavailable_difference}')
    print(f'full_blend_formula_max_abs_difference={formula_difference}')
    print(f'cache_val_recompute_max_abs_difference={cache_val_difference}')
    print(f'cache_train_loo_recompute_max_abs_difference={cache_train_difference}')
    print(f'availability_val_equal={availability_val_equal}')
    print(f'availability_train_loo_equal={availability_train_equal}')
    print(f'validation_current_h={current_h.tolist()}')
    print(f'validation_hard_exact_h={hard_h.tolist()}')
    print(
        f'validation gain overall={current_h.mean() - hard_h.mean():.6f}, '
        f'late={current_h[7:].mean() - hard_h[7:].mean():.6f}'
    )
    print(f'PASS={passed}')


if __name__ == '__main__':
    main()
