"""Strict protocol and gradient audit for the candidate reranker experiment."""

import copy
import pickle
import sys

import numpy as np

for dependency_path in (
    'D:/论文/2025年12月/PIR-main/.venv/Lib/site-packages',
    'D:/论文/2026年1月/SSL-STMFomer-main/.venv/Lib/site-packages',
):
    if dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)

import torch

from models.TLAST_improved import Model
from utils.datasets_utils import load_dataset


FINAL_EXP = (
    'EXP_experiments/TLAST_improved/PEMS04/'
    'TLAST_improved_PEMS04_12to12_exp07181357_s1'
)
FINAL_CHECKPOINT = FINAL_EXP + '/best.pth'
CANDIDATE_CACHE = (
    'datasets/PEMS04/'
    'PEMS04_12to12_analog_k4_candidate_indices.npz'
)


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs


def _enable_reranker(args):
    args.batch_size = 2
    args.num_nodes = 307
    args.pearson_adj = None
    if isinstance(args.k_hops, str):
        args.k_hops = [
            int(value) for value in args.k_hops.split(',')
        ]
    args.use_analog_candidate_reranker = 1
    args.analog_candidate_feature_dim = 8
    args.analog_candidate_hidden = 16
    args.analog_candidate_min_valid_history = 6
    args.analog_candidate_cache = CANDIDATE_CACHE
    args.freeze_parent_train_analog_candidate_reranker = 1
    args.freeze_parent_train_analog_lowrank_retrieval = 0
    args.freeze_parent_train_analog_exact_match = 0
    return args


def _to_model_input(loader_input, y_time, device):
    if len(loader_input) != 8:
        raise RuntimeError(
            f'Expected 8 loader inputs, got {len(loader_input)}'
        )
    (
        x,
        x_time,
        analog_retrieval,
        exact_retrieval,
        exact_available,
        candidate_future,
        candidate_available,
        candidate_features,
    ) = loader_input
    return [
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(x_time, dtype=torch.long, device=device),
        torch.as_tensor(y_time, dtype=torch.long, device=device),
        torch.as_tensor(
            analog_retrieval, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            exact_retrieval, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            exact_available, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            candidate_future, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            candidate_available, dtype=torch.float32, device=device
        ),
        torch.as_tensor(
            candidate_features, dtype=torch.float32, device=device
        ),
    ]


def main():
    with open(FINAL_EXP + '/config.pkl', 'rb') as handle:
        target_args = _enable_reranker(pickle.load(handle))
    dataloader, num_nodes, scaler, _ = load_dataset(
        target_args, _Logger()
    )
    if num_nodes != 307:
        raise RuntimeError(f'Expected 307 nodes, got {num_nodes}')

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    parent_state = torch.load(FINAL_CHECKPOINT, map_location=device)
    target = Model(target_args).to(device)
    missing, unexpected = target.load_state_dict(
        parent_state, strict=False
    )
    expected_missing = {
        'analog_candidate_reranker.' + key
        for key in target.analog_candidate_reranker.state_dict()
    }
    if set(missing) != expected_missing or list(unexpected):
        raise RuntimeError(
            f'Parent load mismatch: missing={missing}, '
            f'unexpected={unexpected}'
        )

    baseline_args = copy.deepcopy(target_args)
    baseline_args.use_analog_candidate_reranker = 0
    baseline_args.freeze_parent_train_analog_candidate_reranker = 0
    baseline = Model(baseline_args).to(device)
    baseline.load_state_dict(parent_state, strict=True)

    loader_input, target_pair = next(
        dataloader['val_loader'].get_iterator()
    )
    y_raw, y_time = target_pair
    model_input = _to_model_input(loader_input, y_time, device)
    baseline_input = model_input[:6]
    target.eval()
    baseline.eval()
    with torch.no_grad():
        target_output, _ = target(model_input)
        baseline_output, _ = baseline(baseline_input)
    maximum_epoch0_difference = float(
        (target_output - baseline_output).abs().max().cpu()
    )
    maximum_initial_correction = float(
        target.analog_candidate_reranker.last_correction.abs().max().cpu()
    )
    if maximum_epoch0_difference != 0.0:
        raise RuntimeError(
            'Candidate reranker is not an exact epoch-0 identity: '
            f'{maximum_epoch0_difference}'
        )
    if maximum_initial_correction != 0.0:
        raise RuntimeError(
            'Initial candidate correction is not exactly zero: '
            f'{maximum_initial_correction}'
        )

    for parameter in target.parameters():
        parameter.requires_grad_(False)
    for parameter in target.analog_candidate_reranker.parameters():
        parameter.requires_grad_(True)
    trainable = sum(
        parameter.numel()
        for parameter in target.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in target.parameters())

    target.eval()
    target.analog_candidate_reranker.train()
    optimizer = torch.optim.AdamW(
        target.analog_candidate_reranker.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )
    truth = torch.as_tensor(y_raw, dtype=torch.float32, device=device)
    truth_mask = truth != 0

    first_step_nonzero = set()
    for step in range(2):
        optimizer.zero_grad()
        normalized_prediction, _ = target(model_input)
        raw_prediction = scaler.inverse_transform(normalized_prediction)
        loss = torch.abs(
            raw_prediction[truth_mask] - truth[truth_mask]
        ).mean()
        loss.backward()
        current_nonzero = {
            name
            for name, parameter
            in target.analog_candidate_reranker.named_parameters()
            if (
                parameter.grad is not None
                and bool(torch.any(parameter.grad != 0))
            )
        }
        if step == 0:
            first_step_nonzero = current_nonzero
        optimizer.step()

    upstream_names = {
        name
        for name, _ in target.analog_candidate_reranker.named_parameters()
        if not name.startswith('output_projection.')
    }
    second_step_nonzero = {
        name
        for name, parameter
        in target.analog_candidate_reranker.named_parameters()
        if (
            parameter.grad is not None
            and bool(torch.any(parameter.grad != 0))
        )
    }
    if not {
        'output_projection.weight',
        'output_projection.bias',
    }.issubset(first_step_nonzero):
        raise RuntimeError(
            'Zero-initialized output projection did not receive first-step '
            f'gradients: {sorted(first_step_nonzero)}'
        )
    if not upstream_names.intersection(second_step_nonzero):
        raise RuntimeError(
            'Reranker upstream features did not receive second-step gradients'
        )
    frozen_gradient_names = [
        name
        for name, parameter in target.named_parameters()
        if (
            not name.startswith('analog_candidate_reranker.')
            and parameter.grad is not None
        )
    ]
    if frozen_gradient_names:
        raise RuntimeError(
            f'Frozen parent received gradients: {frozen_gradient_names[:5]}'
        )

    with np.load(CANDIDATE_CACHE, allow_pickle=False) as cache:
        cache_report = {
            split: {
                'queries': int(len(cache[f'idx_{split}'])),
                'uniform_mean_max_abs': float(np.asarray(
                    cache[f'uniform_mean_audit_{split}']
                ).item()),
            }
            for split in ('train', 'val', 'test')
        }
        if int(np.asarray(
            cache['candidates_train_only']
        ).item()) != 1:
            raise RuntimeError('Candidate cache is not marked train-only')

    print(f'device={device}')
    print(f'parent_missing_keys={len(missing)}, unexpected_keys=0')
    print(f'total_parameters={total}, trainable_parameters={trainable}')
    print(f'epoch0_max_abs={maximum_epoch0_difference:.9g}')
    print(
        'initial_candidate_correction_max_abs='
        f'{maximum_initial_correction:.9g}'
    )
    print(f'first_step_nonzero={sorted(first_step_nonzero)}')
    print(f'second_step_nonzero={sorted(second_step_nonzero)}')
    print(f'frozen_parent_gradients={len(frozen_gradient_names)}')
    print(f'cache_report={cache_report}')
    print('AUDIT PASSED')


if __name__ == '__main__':
    main()
