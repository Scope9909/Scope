"""Independent audit of the formal low-rank analog result.

This script reconstructs the saved checkpoint, reproduces Validation/Test
predictions, checks that every parent tensor is bit-identical, recomputes raw
masked MAE, and performs a counterfactual future-target leakage test on the
retrieval implementation.
"""

import argparse
import logging
import pickle

import numpy as np
import torch

from compute_analog_retrieval_cache import (
    _normalize,
    _retrieve_split,
    _training_fallbacks,
)
from models import Scope
from utils import datasets_utils


def _collect(model, loader, scaler, device, num_samples=None):
    predictions = []
    truths = []
    model.eval()
    with torch.no_grad():
        for loader_input, loader_target in loader.get_iterator():
            if len(loader_input) != 3:
                raise RuntimeError(
                    'Expected traffic, time, and analog loader inputs'
                )
            x, x_time, analog = loader_input
            y, y_time = loader_target
            model_input = [
                torch.as_tensor(x, dtype=torch.float32, device=device),
                torch.as_tensor(x_time, dtype=torch.long, device=device),
                torch.as_tensor(y_time, dtype=torch.long, device=device),
                torch.as_tensor(analog, dtype=torch.float32, device=device),
            ]
            prediction, _ = model(model_input)
            prediction = scaler.inverse_transform(prediction)
            predictions.append(prediction.cpu().numpy())
            truths.append(np.asarray(y, dtype=np.float32).copy())
    prediction = np.concatenate(predictions, axis=0)
    truth = np.concatenate(truths, axis=0)
    if num_samples is not None:
        prediction = prediction[:num_samples]
        truth = truth[:num_samples]
    return prediction, truth


def _horizon_mae(prediction, truth):
    result = []
    for horizon in range(truth.shape[1]):
        mask = truth[:, horizon] != 0
        result.append(np.abs(
            prediction[:, horizon] - truth[:, horizon]
        )[mask].mean(dtype=np.float64))
    return np.asarray(result, dtype=np.float64)


def _load_index_and_time(model_args):
    time_path = (
        f'datasets/{model_args.dataset_name}/'
        f'{model_args.dataset_name}_time.npz'
    )
    index_path = (
        f'datasets/{model_args.dataset_name}/'
        f'{model_args.dataset_name}_{model_args.input_length}to'
        f'{model_args.predict_length}_index_scaler.npz'
    )
    with np.load(time_path, allow_pickle=False) as time_cache:
        time_data = np.array(time_cache['time'], copy=True)
    with np.load(index_path, allow_pickle=False) as index_cache:
        idx_train = np.array(index_cache['idx_train'], copy=True)
        idx_val = np.array(index_cache['idx_val'], copy=True)
        idx_test = np.array(index_cache['idx_test'], copy=True)
        scaler_mean = np.array(index_cache['scaler_mean'], copy=True)
        scaler_std = np.array(index_cache['scaler_std'], copy=True)
    return (
        time_data, idx_train, idx_val, idx_test,
        scaler_mean, scaler_std,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_dir',
        default=(
            'EXP_experiments/TLAST_improved/PEMS04/'
            'TLAST_improved_PEMS04_12to12_exp07181246_s1'
        ),
    )
    parser.add_argument(
        '--parent_checkpoint',
        default=(
            'EXP_experiments/TLAST_improved/PEMS04/'
            'TLAST_improved_PEMS04_12to12_exp07170128_s1/best.pth'
        ),
    )
    parser.add_argument(
        '--retrieval_cache',
        default='datasets/PEMS04/PEMS04_12to12_analog_k4.npz',
    )
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    config_path = f'{args.experiment_dir}/config.pkl'
    checkpoint_path = f'{args.experiment_dir}/best.pth'
    output_path = (
        f'{args.experiment_dir}/'
        'TLAST_improved_PEMS04_exp07181246_s1_output.npz'
    )
    with open(config_path, 'rb') as config_file:
        model_args = pickle.load(config_file)
    model_args.device = args.device

    logger = logging.getLogger('verify_lowrank_analog_result')
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.WARNING)
    dataloader, num_nodes, scaler, num_test_samples = (
        datasets_utils.load_dataset(model_args, logger)
    )
    if num_test_samples != 3394 or num_nodes != 307:
        raise RuntimeError(
            f'Unexpected dataset sizes: test={num_test_samples}, '
            f'nodes={num_nodes}'
        )
    model_args.num_nodes = num_nodes

    device = torch.device(args.device)
    model = Scope.Model(model_args).to(device)
    new_state = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'Formal checkpoint mismatch: missing={missing}, '
            f'unexpected={unexpected}'
        )

    # Stronger than requires_grad/log inspection: every tensor belonging to
    # the parent checkpoint must remain bit-identical after adapter fitting.
    parent_state = torch.load(args.parent_checkpoint, map_location='cpu')
    parent_max_abs_difference = 0.0
    changed_parent_tensors = []
    for key, parent_value in parent_state.items():
        if key not in new_state:
            raise RuntimeError(f'New checkpoint is missing parent key {key}')
        new_value = new_state[key].detach().cpu()
        difference = float(torch.max(torch.abs(
            new_value - parent_value.detach().cpu()
        )).item())
        parent_max_abs_difference = max(
            parent_max_abs_difference, difference
        )
        if difference != 0.0:
            changed_parent_tensors.append((key, difference))
    if changed_parent_tensors:
        raise RuntimeError(
            f'Frozen parent tensors changed: {changed_parent_tensors[:5]}'
        )

    validation_prediction, validation_truth = _collect(
        model, dataloader['val_loader'], scaler, device
    )
    test_prediction, test_truth = _collect(
        model,
        dataloader['test_loader'],
        scaler,
        device,
        num_samples=num_test_samples,
    )
    with np.load(output_path, allow_pickle=False) as saved:
        if str(np.asarray(saved['prediction_scale']).item()) != 'raw':
            raise RuntimeError('Saved prediction_scale is not raw')
        saved_test_prediction = np.array(saved['prediction'], copy=True)
        saved_test_truth = np.array(saved['truth'], copy=True)
        saved_validation_prediction = np.array(
            saved['validation_prediction'], copy=True
        )
        saved_validation_truth = np.array(
            saved['validation_truth'], copy=True
        )
    test_prediction_max_abs_difference = float(np.max(np.abs(
        test_prediction - saved_test_prediction
    )))
    validation_prediction_max_abs_difference = float(np.max(np.abs(
        validation_prediction - saved_validation_prediction
    )))
    if not np.array_equal(test_truth, saved_test_truth):
        raise RuntimeError('Reproduced Test truth does not match saved truth')
    if not np.array_equal(validation_truth, saved_validation_truth):
        raise RuntimeError(
            'Reproduced Validation truth does not match saved truth'
        )
    if test_prediction_max_abs_difference > 1e-5:
        raise RuntimeError(
            'Reproduced Test predictions differ from saved output: '
            f'{test_prediction_max_abs_difference}'
        )
    if validation_prediction_max_abs_difference > 1e-5:
        raise RuntimeError(
            'Reproduced Validation predictions differ from saved output: '
            f'{validation_prediction_max_abs_difference}'
        )

    test_h = _horizon_mae(test_prediction, test_truth)
    validation_h = _horizon_mae(
        validation_prediction, validation_truth
    )
    if abs(float(test_h.mean()) - 17.745213853981294) > 1e-6:
        raise RuntimeError('Independent Test MAE does not match 17.74521385')

    # Counterfactual leakage audit: change each selected query's complete
    # future target window by a huge value.  Retrieval for that query must be
    # invariant and still reproduce the formal cache entry.
    raw_data, _ = datasets_utils.load_raw_data(
        datasets_utils.file_path_dict[model_args.dataset_name]
    )
    (
        time_data, idx_train, _, idx_test,
        scaler_mean, scaler_std,
    ) = _load_index_and_time(model_args)
    x_offsets = np.arange(
        -(model_args.input_length - 1), 1, dtype=np.int64
    )
    y_offsets = np.arange(
        1, model_args.predict_length + 1, dtype=np.int64
    )
    num_tod = int(time_data[:, 0].max()) + 1
    fallback = _training_fallbacks(
        raw_data, time_data, idx_train, y_offsets, num_tod
    )
    with np.load(args.retrieval_cache, allow_pickle=False) as cache:
        cached_test_retrieval = np.array(
            cache['retrieval_test'], copy=True
        )
        if int(np.asarray(cache['k']).item()) != 4:
            raise RuntimeError('Formal retrieval cache K is not 4')
        if int(np.asarray(cache['train_leave_one_out']).item()) != 1:
            raise RuntimeError('Training retrieval cache is not leave-one-out')

    selected_positions = [0, len(idx_test) // 2, len(idx_test) - 1]
    future_counterfactual_max_difference = 0.0
    formal_cache_max_difference = 0.0
    for position in selected_positions:
        query_index = idx_test[position:position + 1]
        original = _retrieve_split(
            data=raw_data,
            time_data=time_data,
            query_indices=query_index,
            train_indices=idx_train,
            fallback=fallback,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            k=4,
            min_valid_history=6,
            leave_one_out=False,
            split_name='test-leakage-audit-original',
        )
        counterfactual_data = raw_data.copy()
        counterfactual_data[
            query_index[0] + y_offsets, :, 0
        ] += 12345.0
        counterfactual = _retrieve_split(
            data=counterfactual_data,
            time_data=time_data,
            query_indices=query_index,
            train_indices=idx_train,
            fallback=fallback,
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            k=4,
            min_valid_history=6,
            leave_one_out=False,
            split_name='test-leakage-audit-counterfactual',
        )
        future_counterfactual_max_difference = max(
            future_counterfactual_max_difference,
            float(np.max(np.abs(original - counterfactual))),
        )
        normalized = _normalize(original, scaler_mean, scaler_std)
        formal_cache_max_difference = max(
            formal_cache_max_difference,
            float(np.max(np.abs(
                normalized - cached_test_retrieval[position:position + 1]
            ))),
        )
    if future_counterfactual_max_difference != 0.0:
        raise RuntimeError(
            'Retrieval changed after counterfactual future-target mutation'
        )
    if formal_cache_max_difference > 1e-5:
        raise RuntimeError(
            'Recomputed retrieval does not match the formal cache'
        )

    adapter = model.analog_lowrank_retrieval_residual
    with torch.no_grad():
        blend = torch.tanh(
            adapter.horizon_logits[:, None]
            + torch.einsum(
                'nr,hr->hn', adapter.node_factor, adapter.horizon_factor
            )
        ).cpu().numpy()
    saturated_nodes = blend.mean(axis=0) > 0.9
    # A train-only TOD lookup explains why these sensors can be retrieved
    # almost exactly: they repeat a deterministic daily profile rather than
    # being constant or zero/missing sensors.
    training_raw_end = int(idx_train[-1] + model_args.predict_length)
    training_times = np.arange(training_raw_end + 1)
    tod_lookup = np.zeros((num_tod, num_nodes), dtype=np.float64)
    for tod in range(num_tod):
        selected = training_times[time_data[training_times, 0] == tod]
        tod_lookup[tod] = raw_data[selected, :, 0].mean(axis=0)
    test_target_indices = idx_test[:, None] + y_offsets[None, :]
    tod_prediction = tod_lookup[
        time_data[test_target_indices, 0]
    ]
    raw_test_truth = raw_data[test_target_indices, :, 0]
    saturated_tod_mae = float(np.abs(
        tod_prediction[:, :, saturated_nodes]
        - raw_test_truth[:, :, saturated_nodes]
    ).mean(dtype=np.float64))
    saturated_zero_rate = float(
        (raw_test_truth[:, :, saturated_nodes] == 0).mean()
    )

    print(f'parent_tensor_count={len(parent_state)}')
    print(f'parent_max_abs_difference={parent_max_abs_difference:.10f}')
    print(
        'reproduced_prediction_max_abs_difference '
        f'validation={validation_prediction_max_abs_difference:.10f}, '
        f'test={test_prediction_max_abs_difference:.10f}'
    )
    print(f'test_shape={test_prediction.shape}')
    print(f'validation_h={validation_h.tolist()}')
    print(f'test_h={test_h.tolist()}')
    print(
        f'validation_mae={validation_h.mean():.9f}, '
        f'test_mae={test_h.mean():.9f}, '
        f'test_h8_h12={test_h[7:].mean():.9f}, '
        f'test_drift={test_h[-1] - test_h[0]:.9f}'
    )
    print(
        'future_counterfactual_retrieval_max_difference='
        f'{future_counterfactual_max_difference:.10f}'
    )
    print(
        f'formal_cache_max_difference={formal_cache_max_difference:.10f}'
    )
    print(
        f'saturated_node_count={int(saturated_nodes.sum())}, '
        f'saturated_zero_rate={saturated_zero_rate:.10f}, '
        f'saturated_train_TOD_lookup_test_mae={saturated_tod_mae:.10f}'
    )
    print('PASS=True')


if __name__ == '__main__':
    main()
