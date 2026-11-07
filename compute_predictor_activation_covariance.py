"""Compute train-only second moments for activation-aware predictor SVD.

Only inputs to the frozen richer predictor are accumulated. Labels, validation
and Test are never read by the statistic. The output can be referenced by
``predictor_activation_covariance`` in a Lite configuration.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument(
        '--max-batches', type=int, default=0,
        help='0 uses the complete train loader; positive values are for smoke tests.',
    )
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


CLI_ARGS = parse_args()
# run.py owns a legacy global parser; hide this script's arguments from it.
sys.argv = [sys.argv[0]]

from run import trainer, unpack_dataloader_input  # noqa: E402
from utils import datasets_utils  # noqa: E402


def make_logger():
    logger = logging.getLogger('predictor_activation_covariance')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def append_optional_inputs(model_input, values, device):
    for value in values:
        if value is None:
            continue
        dtype = (
            torch.float16
            if getattr(value, 'dtype', None) == np.float16
            else torch.float32
        )
        model_input.append(torch.as_tensor(value, dtype=dtype, device=device))


def main():
    logger = make_logger()
    config_path = CLI_ARGS.exp_dir / 'config.pkl'
    checkpoint_path = CLI_ARGS.exp_dir / 'best.pth'
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Expected config.pkl and best.pth under {CLI_ARGS.exp_dir}'
        )
    if CLI_ARGS.max_batches < 0:
        raise ValueError('--max-batches must be non-negative')
    if CLI_ARGS.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    with config_path.open('rb') as stream:
        run_args = pickle.load(stream)
    run_args.device = CLI_ARGS.device
    run_args.cache_data_path = 'datasets'
    run_args.prob_factor = getattr(run_args, 'prob_factor', 5)
    if isinstance(getattr(run_args, 'k_hops', '1,2'), str):
        run_args.k_hops = [int(value) for value in run_args.k_hops.split(',')]
    needs_graph = bool(getattr(run_args, 'use_graph_conv', 0)) or (
        getattr(run_args, 'spatial_type', None) == 'agcrn'
    )
    pearson_path = getattr(run_args, 'pearson_adj_path', None)
    default_pearson_path = Path(
        f'datasets/{run_args.dataset_name}/'
        f'{run_args.dataset_name}_pearson_adj.npy'
    )
    if needs_graph and pearson_path and Path(pearson_path).is_file():
        run_args.pearson_adj = torch.as_tensor(
            np.load(pearson_path), dtype=torch.float32
        )
    elif needs_graph and default_pearson_path.is_file():
        run_args.pearson_adj = torch.as_tensor(
            np.load(default_pearson_path), dtype=torch.float32
        )
    else:
        run_args.pearson_adj = None
    # The parent experiment predates these optional Lite arguments.
    run_args.predictor_factor_rank = getattr(
        run_args, 'predictor_factor_rank', 256
    )
    run_args.predictor_activation_covariance = 'none'
    run_args.predictor_activation_ridge = 1e-5

    device = torch.device(CLI_ARGS.device)
    dataloader, num_nodes, scaler, _ = datasets_utils.load_dataset(
        run_args, logger
    )
    run_args.num_nodes = num_nodes
    engine = trainer(
        scaler=scaler, device=device, args=run_args, inverse=False
    )
    state = torch.load(checkpoint_path, map_location=device)
    engine.model.load_state_dict(state, strict=True)
    engine.model.eval()
    predictor = engine.model.time_wise_predictor
    if not hasattr(predictor, 'predict_unit'):
        raise TypeError('Parent checkpoint does not use a richer predictor')

    feature_dim = run_args.input_length * run_args.hid_dim
    gram = torch.zeros(
        feature_dim, feature_dim, device=device, dtype=torch.float32
    )
    sample_count = 0

    def accumulate_predictor_input(_module, inputs):
        nonlocal sample_count
        encoded = inputs[0].detach()
        if encoded.shape[1] != run_args.input_length:
            raise ValueError(f'Unexpected encoded shape: {tuple(encoded.shape)}')
        flat = encoded.permute(0, 2, 1, 3).reshape(-1, feature_dim).float()
        gram.add_(flat.T.matmul(flat))
        sample_count += flat.shape[0]

    handle = predictor.register_forward_pre_hook(accumulate_predictor_input)
    loader = dataloader['train_loader']
    processed_batches = 0
    try:
        with torch.inference_mode():
            for loader_input, loader_target in loader.get_iterator():
                (
                    x, x_time, analog_retrieval,
                    exact_retrieval, exact_available,
                    candidate_future, candidate_available, candidate_features,
                ) = unpack_dataloader_input(loader_input)
                _, y_time = loader_target
                model_input = [
                    torch.as_tensor(x, dtype=torch.float32, device=device),
                    torch.as_tensor(x_time, dtype=torch.long, device=device),
                    torch.as_tensor(y_time, dtype=torch.long, device=device),
                ]
                append_optional_inputs(
                    model_input,
                    (
                        analog_retrieval,
                        exact_retrieval,
                        exact_available,
                        candidate_future,
                        candidate_available,
                        candidate_features,
                    ),
                    device,
                )
                engine.model(model_input)
                processed_batches += 1
                if processed_batches % 50 == 0:
                    logger.info(
                        'processed_batches=%d, activation_rows=%d',
                        processed_batches, sample_count,
                    )
                if (
                    CLI_ARGS.max_batches > 0
                    and processed_batches >= CLI_ARGS.max_batches
                ):
                    break
    finally:
        handle.remove()

    if sample_count <= 0:
        raise RuntimeError('No predictor activations were collected')
    second_moment = (gram / float(sample_count)).cpu().numpy()
    if not np.isfinite(second_moment).all():
        raise ValueError('Computed second moment contains NaN/Inf')
    symmetry_error = float(np.max(np.abs(second_moment - second_moment.T)))
    if symmetry_error > 1e-5:
        raise AssertionError(
            f'Second moment is unexpectedly asymmetric: {symmetry_error}'
        )

    CLI_ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = CLI_ARGS.output.with_suffix(CLI_ARGS.output.suffix + '.tmp.npz')
    np.savez_compressed(
        temporary,
        second_moment=second_moment.astype(np.float32),
        sample_count=np.asarray(sample_count, dtype=np.int64),
        processed_batches=np.asarray(processed_batches, dtype=np.int64),
        feature_dim=np.asarray(feature_dim, dtype=np.int64),
        source_checkpoint=np.asarray(str(checkpoint_path.resolve())),
        split=np.asarray('train'),
    )
    temporary.replace(CLI_ARGS.output)
    logger.info(
        'saved=%s shape=%s batches=%d rows=%d symmetry_error=%.3g',
        CLI_ARGS.output, second_moment.shape,
        processed_batches, sample_count, symmetry_error,
    )


if __name__ == '__main__':
    main()
