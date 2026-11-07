"""Independently verify a trained exact-history refinement experiment."""

import argparse
import json

import numpy as np
import torch

from compute_analog_exact_match_cache import _retrieve_exact_split
from utils.datasets_utils import file_path_dict, load_raw_data


def _horizon_mae(prediction, truth):
    valid = (truth != 0) & np.isfinite(truth) & np.isfinite(prediction)
    total = np.where(valid, np.abs(prediction - truth), 0.0).sum(
        axis=(0, 2), dtype=np.float64
    )
    count = valid.sum(axis=(0, 2))
    return total / np.maximum(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_dir',
        default=(
            'EXP_experiments/TLAST_improved/PEMS04/'
            'TLAST_improved_PEMS04_12to12_exp07181357_s1'
        ),
    )
    parser.add_argument(
        '--parent_dir',
        default=(
            'EXP_experiments/TLAST_improved/PEMS04/'
            'TLAST_improved_PEMS04_12to12_exp07181246_s1'
        ),
    )
    parser.add_argument(
        '--cache',
        default=(
            'datasets/PEMS04/'
            'PEMS04_12to12_analog_k4_exact_mean.npz'
        ),
    )
    args = parser.parse_args()

    with open(
        f'{args.experiment_dir}/config.json', encoding='utf-8'
    ) as handle:
        config = json.load(handle)
    required_config = {
        'dataset_name': 'PEMS04',
        'input_length': 12,
        'predict_length': 12,
        'seed': 1,
        'use_ftle': 0,
        'use_time_lag_embedding': 1,
        'use_analog_lowrank_retrieval_residual': 1,
        'use_analog_exact_match_residual': 1,
        'freeze_parent_train_analog_exact_match': 1,
    }
    config_ok = all(
        config.get(key) == value for key, value in required_config.items()
    )

    checkpoint = torch.load(
        f'{args.experiment_dir}/best.pth', map_location='cpu'
    )
    parent_checkpoint = torch.load(
        f'{args.parent_dir}/best.pth', map_location='cpu'
    )
    exact_key = 'analog_exact_match_residual.horizon_logits'
    candidate_extra_keys = set(checkpoint) - set(parent_checkpoint)
    candidate_missing_keys = set(parent_checkpoint) - set(checkpoint)
    parent_max_difference = 0.0
    for key, parent_value in parent_checkpoint.items():
        candidate_value = checkpoint[key]
        if parent_value.shape != candidate_value.shape:
            raise RuntimeError(f'Parent tensor shape changed for {key}')
        if parent_value.numel():
            parent_max_difference = max(
                parent_max_difference,
                float((parent_value - candidate_value).abs().max()),
            )
    horizon_blend = torch.tanh(checkpoint[exact_key]).numpy()

    candidate_output_path = (
        f'{args.experiment_dir}/'
        'TLAST_improved_PEMS04_exp07181357_s1_output.npz'
    )
    parent_output_path = (
        f'{args.parent_dir}/'
        'TLAST_improved_PEMS04_exp07181246_s1_output.npz'
    )
    with np.load(candidate_output_path, allow_pickle=False) as archive:
        prediction = np.array(archive['prediction'], copy=True)[..., 0]
        truth = np.array(archive['truth'], copy=True)[..., 0]
        validation_prediction = np.array(
            archive['validation_prediction'], copy=True
        )[..., 0]
        validation_truth = np.array(
            archive['validation_truth'], copy=True
        )[..., 0]
        scaler_mean = float(np.asarray(archive['scaler_mean']).item())
        scaler_std = float(np.asarray(archive['scaler_std']).item())
        prediction_scale = str(np.asarray(
            archive['prediction_scale']
        ).item())
    with np.load(parent_output_path, allow_pickle=False) as archive:
        parent_prediction = np.array(
            archive['prediction'], copy=True
        )[..., 0]
        parent_truth = np.array(archive['truth'], copy=True)[..., 0]
        parent_validation_prediction = np.array(
            archive['validation_prediction'], copy=True
        )[..., 0]
        parent_validation_truth = np.array(
            archive['validation_truth'], copy=True
        )[..., 0]

    index_path = 'datasets/PEMS04/PEMS04_12to12_index_scaler.npz'
    with np.load(index_path, allow_pickle=False) as archive:
        expected_indices = {
            split: np.array(archive[f'idx_{split}'], copy=True)
            for split in ('train', 'val', 'test')
        }
    with np.load(args.cache, allow_pickle=False) as archive:
        cache_indices_ok = all(np.array_equal(
            archive[f'idx_{split}'], expected_indices[split]
        ) for split in ('train', 'val', 'test'))
        metadata_ok = (
            int(np.asarray(archive['train_leave_one_out']).item()) == 1
            and int(np.asarray(
                archive['exact_train_leave_one_out']
            ).item()) == 1
            and str(np.asarray(
                archive['exact_match_rule']
            ).item()) == 'masked_mse_eq_0'
            and int(np.asarray(
                archive['exact_min_valid_history']
            ).item()) == 6
        )
        exact_test = np.array(
            archive['exact_retrieval_test'], copy=True
        )[..., 0]
        available_test = np.array(
            archive['exact_available_test'], copy=True
        )[..., 0].astype(bool)
        exact_val = np.array(
            archive['exact_retrieval_val'], copy=True
        )[..., 0]
        available_val = np.array(
            archive['exact_available_val'], copy=True
        )[..., 0].astype(bool)

    # Counterfactual leakage audit on selected Test queries.  Changing their
    # complete future target windows must not affect retrieval because only
    # observable query histories and fixed training candidates are allowed.
    raw_data, _ = load_raw_data(file_path_dict['PEMS04'])
    with np.load(
        'datasets/PEMS04/PEMS04_time.npz', allow_pickle=False
    ) as archive:
        time_data = archive['time']
    x_offsets = np.arange(-11, 1, dtype=np.int64)
    y_offsets = np.arange(1, 13, dtype=np.int64)
    # Keep audited queries far enough apart that perturbing one query's future
    # cannot legitimately enter another audited query's observed history.
    selected_test_indices = expected_indices['test'][::400][:8]
    original_retrieval, original_available = _retrieve_exact_split(
        raw_data, time_data, selected_test_indices,
        expected_indices['train'], x_offsets, y_offsets,
        6, False, 'counterfactual-original',
    )
    perturbed_data = np.array(raw_data, copy=True)
    for query_index in selected_test_indices:
        perturbed_data[query_index + y_offsets, :, 0] += 12345.0
    perturbed_retrieval, perturbed_available = _retrieve_exact_split(
        perturbed_data, time_data, selected_test_indices,
        expected_indices['train'], x_offsets, y_offsets,
        6, False, 'counterfactual-perturbed-target',
    )
    counterfactual_retrieval_difference = float(np.max(np.abs(
        original_retrieval - perturbed_retrieval
    )))
    counterfactual_available_equal = bool(np.array_equal(
        original_available, perturbed_available
    ))

    if not (
        np.array_equal(truth, parent_truth)
        and np.array_equal(validation_truth, parent_validation_truth)
    ):
        raise RuntimeError('Candidate and parent truths are not aligned')
    exact_test_raw = exact_test * scaler_std + scaler_mean
    exact_val_raw = exact_val * scaler_std + scaler_mean
    reconstructed_test = parent_prediction + (
        horizon_blend[None, :, None]
        * available_test
        * (exact_test_raw - parent_prediction)
    )
    reconstructed_val = parent_validation_prediction + (
        horizon_blend[None, :, None]
        * available_val
        * (exact_val_raw - parent_validation_prediction)
    )
    reconstruction_test_difference = float(np.max(np.abs(
        prediction - reconstructed_test
    )))
    reconstruction_val_difference = float(np.max(np.abs(
        validation_prediction - reconstructed_val
    )))
    unavailable_test_difference = float(np.max(np.abs(
        prediction[~available_test] - parent_prediction[~available_test]
    )))
    unavailable_val_difference = float(np.max(np.abs(
        validation_prediction[~available_val]
        - parent_validation_prediction[~available_val]
    )))

    test_h = _horizon_mae(prediction, truth)
    val_h = _horizon_mae(validation_prediction, validation_truth)
    parent_test_h = _horizon_mae(parent_prediction, truth)
    parent_val_h = _horizon_mae(
        parent_validation_prediction, validation_truth
    )
    gain_h = parent_test_h - test_h
    test_mae = float(test_h.mean())
    parent_test_mae = float(parent_test_h.mean())
    late = float(test_h[7:].mean())
    parent_late = float(parent_test_h[7:].mean())
    drift = float(test_h[-1] - test_h[0])
    parent_drift = float(parent_test_h[-1] - parent_test_h[0])

    passed = (
        config_ok
        and candidate_extra_keys == {exact_key}
        and not candidate_missing_keys
        and parent_max_difference == 0.0
        and prediction_scale == 'raw'
        and prediction.shape == truth.shape == (3394, 12, 307)
        and validation_prediction.shape == validation_truth.shape
        == (3394, 12, 307)
        and cache_indices_ok
        and metadata_ok
        and counterfactual_retrieval_difference == 0.0
        and counterfactual_available_equal
        and reconstruction_test_difference < 1e-4
        and reconstruction_val_difference < 1e-4
        and unavailable_test_difference == 0.0
        and unavailable_val_difference == 0.0
        and test_mae < parent_test_mae
    )
    print(f'config_ok={config_ok}')
    print(f'candidate_extra_keys={sorted(candidate_extra_keys)}')
    print(f'candidate_missing_keys={sorted(candidate_missing_keys)}')
    print(f'parent_max_abs_difference={parent_max_difference}')
    print(f'horizon_blend={horizon_blend.tolist()}')
    print(f'prediction_scale={prediction_scale}')
    print(f'test_shape={prediction.shape}')
    print(f'validation_shape={validation_prediction.shape}')
    print(f'cache_indices_ok={cache_indices_ok}, metadata_ok={metadata_ok}')
    print(
        'counterfactual_test_future_target_change: '
        f'retrieval_max_abs_difference='
        f'{counterfactual_retrieval_difference}, '
        f'available_equal={counterfactual_available_equal}'
    )
    print(
        f'reconstruction_max_abs_difference test='
        f'{reconstruction_test_difference}, val={reconstruction_val_difference}'
    )
    print(
        f'unavailable_max_abs_difference test='
        f'{unavailable_test_difference}, val={unavailable_val_difference}'
    )
    print(f'test_h={test_h.tolist()}')
    print(f'validation_h={val_h.tolist()}')
    print(f'gain_vs_parent_h={gain_h.tolist()}')
    print(
        f'test_mae={test_mae:.12f}, parent={parent_test_mae:.12f}, '
        f'gain={parent_test_mae - test_mae:.12f}'
    )
    print(
        f'h8_h12={late:.12f}, parent={parent_late:.12f}, '
        f'gain={parent_late - late:.12f}'
    )
    print(
        f'drift={drift:.12f}, parent={parent_drift:.12f}, '
        f'reduction={parent_drift - drift:.12f}'
    )
    print(
        f'validation_mae={val_h.mean():.12f}, '
        f'parent={parent_val_h.mean():.12f}, '
        f'gain={parent_val_h.mean() - val_h.mean():.12f}'
    )
    print(f'PASS={passed}')


if __name__ == '__main__':
    main()
