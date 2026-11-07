"""Frozen-Student controls for the PEMS04 neighbor-innovation hypothesis.

The complete source Student stays frozen and in evaluation mode.  Each arm
trains the same 256-parameter temporal path; only its observable input differs:
own-history change, random-neighbor innovation, or correlated-neighbor
innovation.  Validation prefix selects the epoch and a disjoint suffix (with a
12-origin embargo) assesses the selected state.  Test is never evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from models import TLAST_improved
from utils import datasets_utils
from contrib3_horizon_conflict_20261016.audit_horizon_gradients import tensorize_model_input
from contrib3_horizon_structure_20261017.audit_predictable_structure import (
    EXPERIMENT, ROOT, metrics, subset_loader,
)
from contrib3_neighbor_path_20261017.screen_neighbor_signal import graph_from_train


class IncrementalPath(nn.Module):
    """Equal-capacity temporal path driven by one of three observed signals."""

    def __init__(self, mode: str, neighbor_indices: np.ndarray, channels: int):
        super().__init__()
        if mode not in {'own', 'random_neighbor', 'correlated_neighbor'}:
            raise ValueError(mode)
        self.mode = mode
        self.register_buffer(
            'neighbors', torch.as_tensor(neighbor_indices, dtype=torch.long)
        )
        self.temporal = nn.Conv1d(1, channels, kernel_size=3, padding=1, bias=False)
        self.gate = nn.Parameter(torch.zeros(channels))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        own = history[..., 0]
        if self.mode == 'own':
            signal = own
        else:
            neighbor_mean = own[:, :, self.neighbors].mean(dim=-1)
            signal = neighbor_mean - own
        innovation = torch.cat((
            torch.zeros_like(signal[:, :1]),
            signal[:, 1:] - signal[:, :-1],
        ), dim=1)
        batch, steps, nodes = innovation.shape
        encoded = self.temporal(
            innovation.permute(0, 2, 1).reshape(batch * nodes, 1, steps)
        )
        encoded = encoded.reshape(batch, nodes, -1, steps).permute(0, 2, 1, 3)
        return torch.tanh(self.gate)[None, :, None, None] * encoded


class FrozenStudentWithPath(nn.Module):
    def __init__(self, base: nn.Module, mode: str, graph: np.ndarray):
        super().__init__()
        self.base = base
        self.path = IncrementalPath(mode, graph, base.hid_dim)
        self.current_history = None
        self.base.time_conv.register_forward_pre_hook(self._inject)

    def _inject(self, module, inputs):
        if self.current_history is None:
            raise RuntimeError('History was not set before time_conv')
        return (inputs[0] + self.path(self.current_history),)

    def forward(self, model_input):
        self.current_history = model_input[0]
        try:
            return self.base(model_input)
        finally:
            self.current_history = None


def forward_raw(model, inputs, targets, scaler, device):
    y, y_time = targets
    model_input = tensorize_model_input(inputs, y_time, device)
    output, _ = model(model_input)
    prediction = scaler.inverse_transform(output)
    truth = torch.as_tensor(y, dtype=torch.float32, device=device)
    return prediction, truth


def assess(model, loader, scaler, device):
    model.eval()
    prediction, truth = [], []
    with torch.inference_mode():
        for inputs, targets in loader.get_iterator():
            p, y = forward_raw(model, inputs, targets, scaler, device)
            prediction.append(p[..., 0].transpose(1, 2).cpu().numpy())
            truth.append(y[..., 0].transpose(1, 2).cpu().numpy())
    prediction = np.concatenate(prediction, axis=0)
    truth = np.concatenate(truth, axis=0)
    assert len(prediction) == len(loader.idx)
    return prediction, truth


def clone_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def train_arm(name, args, source_state, graph, dataloader, scaler, device,
              epochs, lr, source_selection_mae, output_dir, logger):
    torch.manual_seed(31)
    np.random.seed(31)
    base = TLAST_improved.Model(args).to(device)
    base.load_state_dict(source_state, strict=True)
    model = FrozenStudentWithPath(base, name, graph).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.path.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in trainable) != 256:
        raise RuntimeError('Expected exactly 256 trainable path parameters')
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    train_loader = subset_loader(
        dataloader['train_loader'], np.arange(dataloader['train_loader'].size)
    )
    val_loader = dataloader['val_loader']
    split = len(val_loader.idx) // 2
    selection, assessment = slice(0, split - 12), slice(split, len(val_loader.idx))

    # The zero gate makes this an exact source prediction.  Save epoch 0 so a
    # non-improving arm cannot accidentally return its final, degraded state.
    initial_state = clone_state(model.path)
    anchor_p, anchor_y = assess(model, val_loader, scaler, device)
    anchor = metrics(anchor_p[selection], anchor_y[selection])['mae']
    if abs(anchor - source_selection_mae) >= 3e-5:
        raise RuntimeError(f'Non-zero epoch-0 drift: {anchor} vs {source_selection_mae}')
    best, selected_epoch, selected_state = anchor, 0, initial_state
    logs = []

    for epoch in range(1, epochs + 1):
        np.random.seed(1000 + epoch)
        train_loader.shuffle()
        model.eval()
        total, batches = 0.0, 0
        for inputs, targets in train_loader.get_iterator():
            optimizer.zero_grad(set_to_none=True)
            prediction, truth = forward_raw(model, inputs, targets, scaler, device)
            valid = torch.isfinite(truth) & truth.ne(0)
            loss = (prediction - truth).abs()[valid].mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        p, y = assess(model, val_loader, scaler, device)
        row = {
            'epoch': epoch,
            'train_batch_mae': total / batches,
            'selection_mae': metrics(p[selection], y[selection])['mae'],
            'full_validation_mae': metrics(p, y)['mae'],
        }
        logs.append(row)
        logger.info(
            '%s epoch %d Train %.6f Select %.6f FullVal %.6f',
            name, epoch, row['train_batch_mae'], row['selection_mae'],
            row['full_validation_mae'],
        )
        if row['selection_mae'] < best:
            best = row['selection_mae']
            selected_epoch = epoch
            selected_state = clone_state(model.path)

    model.path.load_state_dict(selected_state, strict=True)
    checkpoint_path = output_dir / f'{name}_selected.pth'
    torch.save(selected_state, checkpoint_path)
    p, y = assess(model, val_loader, scaler, device)
    return {
        'selected_epoch': selected_epoch,
        'selection_mae': best,
        'assessment': metrics(p[assessment], y[assessment]),
        'full_validation': metrics(p, y),
        'epochs': logs,
        'trainable_parameters': sum(parameter.numel() for parameter in trainable),
        'checkpoint': str(checkpoint_path),
        'gate_mean_abs': float(model.path.gate.abs().mean().detach()),
        'gate_max_abs': float(model.path.gate.abs().max().detach()),
    }, p, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument(
        '--output', type=Path,
        default=Path(__file__).with_name('pems04_s1_frozen_incremental_controls.json'),
    )
    cli = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('frozen-incremental-controls')
    torch.set_num_threads(4)

    args = SimpleNamespace(**json.loads((EXPERIMENT / 'config.json').read_text(encoding='utf-8')))
    args.device, args.batch_size, args.cache_data_path = cli.device, 32, 'datasets'
    args.kd_alpha, args.teacher_predictions_path = 0.0, 'none'
    args.teacher_predictions_strict_alignment = 0
    dataloader, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.num_nodes, args.pearson_adj = nodes, None
    args.prob_factor = getattr(args, 'prob_factor', 5)
    if isinstance(args.k_hops, str):
        args.k_hops = [int(value) for value in args.k_hops.split(',')]

    with np.load(ROOT / 'datasets/PEMS04/data.npz', allow_pickle=False) as archive:
        flow = np.asarray(archive['data'][..., 0], dtype=np.float32)
    correlated_graph, random_graph, correlated_corr, random_corr = graph_from_train(
        flow, int(dataloader['train_loader'].idx.max()) + 13, 4, 41
    )
    graphs = {
        'own': correlated_graph,  # Unused, retained for an identical module layout.
        'random_neighbor': random_graph,
        'correlated_neighbor': correlated_graph,
    }

    device = torch.device(cli.device)
    source_path = EXPERIMENT / 'best.pth'
    source_state = torch.load(source_path, map_location='cpu', weights_only=True)
    source = TLAST_improved.Model(args).to(device).eval()
    source.load_state_dict(source_state, strict=True)
    source_p, source_y = assess(source, dataloader['val_loader'], scaler, device)
    source_full = metrics(source_p, source_y)
    if abs(source_full['mae'] - 17.8334115) >= 2e-4:
        raise RuntimeError(f'Unexpected source MAE: {source_full["mae"]}')
    split = len(source_p) // 2
    selection, assessment = slice(0, split - 12), slice(split, len(source_p))
    source_selection = metrics(source_p[selection], source_y[selection])['mae']
    source_assessment = metrics(source_p[assessment], source_y[assessment])['mae']

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    reports, predictions = {}, {}
    for name in ('own', 'random_neighbor', 'correlated_neighbor'):
        report, prediction, truth = train_arm(
            name, args, source_state, graphs[name], dataloader, scaler, device,
            cli.epochs, cli.lr, source_selection, cli.output.parent, logger,
        )
        if not np.array_equal(source_y, truth):
            raise RuntimeError('Validation targets changed between arms')
        reports[name], predictions[name] = report, prediction

    assessment_indices = np.arange(len(source_p))[assessment]
    for name, report in reports.items():
        candidate = predictions[name][assessment]
        report['assessment_improvement_vs_source'] = (
            source_assessment - report['assessment']['mae']
        )
        report['eight_chronological_block_improvements'] = [
            metrics(source_p[block], source_y[block])['mae']
            - metrics(predictions[name][block], source_y[block])['mae']
            for block in np.array_split(assessment_indices, 8)
        ]

    correlated = reports['correlated_neighbor']
    controls = (reports['own'], reports['random_neighbor'])
    passed = (
        correlated['selected_epoch'] > 0
        and correlated['assessment_improvement_vs_source'] >= 0.01
        and correlated['assessment']['mae'] < min(
            control['assessment']['mae'] for control in controls
        )
        and sum(
            value > 0
            for value in correlated['eight_chronological_block_improvements']
        ) >= 6
    )
    result = {
        'protocol': (
            'Frozen original Student; equal 256-parameter own-change, random-neighbor, '
            'and correlated-neighbor paths; Train-only graphs and fitting; Validation '
            'prefix selects epoch; 12-origin embargo; Validation suffix assesses; Test '
            'not evaluated.'
        ),
        'source_checkpoint': str(source_path),
        'source_checkpoint_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'graph_k': 4,
        'train_graph_delta_correlation': correlated_corr,
        'random_graph_delta_correlation': random_corr,
        'optimizer': {'learning_rate': cli.lr, 'weight_decay': 1e-4},
        'source_full_validation': source_full,
        'source_selection_mae': source_selection,
        'source_assessment_mae': source_assessment,
        'selection_endpoints': [
            int(dataloader['val_loader'].idx[0]),
            int(dataloader['val_loader'].idx[split - 13]),
        ],
        'assessment_endpoints': [
            int(dataloader['val_loader'].idx[split]),
            int(dataloader['val_loader'].idx[-1]),
        ],
        'arms': reports,
        'predeclared_gate': (
            'Correlated-neighbor suffix improvement >=0.01, beats both equal-capacity '
            'controls, and improves >=6/8 chronological blocks.'
        ),
        'passed': bool(passed),
        'limitations': [
            'The source Student checkpoint was selected on full Validation; this is a '
            'mechanism screen rather than an untouched confirmatory evaluation.',
            'One seed and one predeclared learning rate are used.',
        ],
    }
    cli.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('Wrote %s; passed=%s', cli.output, passed)


if __name__ == '__main__':
    main()
