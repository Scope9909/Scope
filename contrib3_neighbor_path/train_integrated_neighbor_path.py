"""Integrated lightweight neighbor-innovation path in the Student time trunk.

This is a bounded PEMS04 pilot. It wraps the frozen existing Student instead
of changing production model code. A zero gate makes epoch 0 exactly the
source checkpoint. Both arms update the same predictor with the same data;
only the neighbor arm also learns a 256-parameter temporal input path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from models import Scope
from utils import datasets_utils
from contrib3_horizon_conflict.audit_horizon_gradients import tensorize_model_input
from contrib3_horizon_structure.audit_predictable_structure import (
    EXPERIMENT, ROOT, metrics, subset_loader,
)
from contrib3_neighbor_path.screen_neighbor_signal import graph_from_train


class NeighborInnovation(nn.Module):
    def __init__(self, neighbor_indices, channels):
        super().__init__()
        self.register_buffer('neighbors', torch.as_tensor(neighbor_indices, dtype=torch.long))
        self.temporal = nn.Conv1d(1, channels, kernel_size=3, padding=1, bias=False)
        self.gate = nn.Parameter(torch.zeros(channels))

    def forward(self, history):
        # Every entry in history is observed at the forecast origin.
        own = history[..., 0]
        neighbor_mean = own[:, :, self.neighbors].mean(dim=-1)
        difference = neighbor_mean - own
        innovation = torch.cat((torch.zeros_like(difference[:, :1]),
                                difference[:, 1:] - difference[:, :-1]), dim=1)
        b, t, n = innovation.shape
        temporal = self.temporal(innovation.permute(0, 2, 1).reshape(b * n, 1, t))
        temporal = temporal.reshape(b, n, -1, t).permute(0, 2, 1, 3)
        return torch.tanh(self.gate)[None, :, None, None] * temporal


class IntegratedStudent(nn.Module):
    def __init__(self, base, neighbors, enabled):
        super().__init__()
        self.base = base
        self.path = NeighborInnovation(neighbors, base.hid_dim)
        self.enabled = enabled
        self.current_history = None
        self.base.time_conv.register_forward_pre_hook(self._inject)

    def _inject(self, module, inputs):
        if self.enabled:
            assert self.current_history is not None
            return (inputs[0] + self.path(self.current_history),)
        return inputs

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
    check = np.arange(0, len(loader.idx), max(1, len(loader.idx)//40))
    expected = loader.data[loader.idx[check, None] + np.arange(1, 13)[None, :], :, 0]
    assert np.allclose(truth[check], expected.transpose(0, 2, 1), atol=1e-3)
    return prediction, truth


def train_arm(name, args, state, graph, dataloader, scaler, device, epochs, lr,
              baseline_prefix, checkpoint_dir, logger):
    torch.manual_seed(31)
    np.random.seed(31)
    base = Scope.Model(args).to(device)
    base.load_state_dict(state, strict=True)
    model = IntegratedStudent(base, graph, name == 'neighbor').to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.base.time_wise_predictor.parameters():
        parameter.requires_grad_(True)
    if name == 'neighbor':
        for parameter in model.path.parameters():
            parameter.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{'params': list(model.base.time_wise_predictor.parameters()), 'lr': lr}]
    if name == 'neighbor':
        groups.append({'params': list(model.path.parameters()), 'lr': 1e-3})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    train_loader = subset_loader(dataloader['train_loader'], np.arange(dataloader['train_loader'].size))
    val_loader = dataloader['val_loader']
    split = len(val_loader.idx) // 2
    selection, assessment = slice(0, split - 12), slice(split, len(val_loader.idx))
    anchor_p, anchor_y = assess(model, val_loader, scaler, device)
    anchor = metrics(anchor_p[selection], anchor_y[selection])['mae']
    assert abs(anchor - baseline_prefix) < 3e-5
    best, chosen_epoch, checkpoint = anchor, 0, None
    logs = []
    for epoch in range(1, epochs + 1):
        np.random.seed(1000 + epoch)
        train_loader.shuffle()
        # The frozen backbone must use its original inference-time behavior.
        model.eval()
        total, batches = 0., 0
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
        scores = {'epoch': epoch, 'train_batch_mae': total / batches,
                  'selection_mae': metrics(p[selection], y[selection])['mae'],
                  'full_validation_mae': metrics(p, y)['mae']}
        logs.append(scores)
        logger.info('%s epoch %d Train %.6f Select %.6f FullVal %.6f',
                    name, epoch, scores['train_batch_mae'], scores['selection_mae'], scores['full_validation_mae'])
        if scores['selection_mae'] < best:
            best, chosen_epoch = scores['selection_mae'], epoch
            checkpoint = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if checkpoint is not None:
        model.load_state_dict(checkpoint, strict=True)
        checkpoint_path = checkpoint_dir / f'{name}_selected.pth'
        torch.save(checkpoint, checkpoint_path)
    else:
        checkpoint_path = None
    p, y = assess(model, val_loader, scaler, device)
    report = {'selected_epoch': chosen_epoch, 'selection_mae': best,
              'assessment': metrics(p[assessment], y[assessment]),
              'full_validation': metrics(p, y), 'epochs': logs,
              'trainable_parameters': sum(v.numel() for v in trainable),
              'neighbor_parameters': sum(v.numel() for v in model.path.parameters()) if name == 'neighbor' else 0,
              'checkpoint': str(checkpoint_path) if checkpoint_path else None,
              'gate_magnitude': float(model.path.gate.abs().max().detach())}
    return report, p, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('pems04_s1_integrated_neighbor.json'))
    cli = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('integrated-neighbor')
    torch.set_num_threads(4)
    args = SimpleNamespace(**json.loads((EXPERIMENT / 'config.json').read_text(encoding='utf-8')))
    args.device, args.batch_size, args.cache_data_path = cli.device, 32, 'datasets'
    args.kd_alpha, args.teacher_predictions_path, args.teacher_predictions_strict_alignment = 0., 'none', 0
    dataloader, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.num_nodes, args.pearson_adj = nodes, None
    args.prob_factor = getattr(args, 'prob_factor', 5)
    if isinstance(args.k_hops, str):
        args.k_hops = [int(v) for v in args.k_hops.split(',')]
    with np.load(ROOT / 'datasets/PEMS04/data.npz', allow_pickle=False) as archive:
        flow = np.asarray(archive['data'][..., 0], dtype=np.float32)
    graph, _, graph_corr, _ = graph_from_train(flow, int(dataloader['train_loader'].idx.max()) + 13, 4, 41)
    device = torch.device(cli.device)
    state = torch.load(EXPERIMENT / 'best.pth', map_location='cpu', weights_only=True)
    source = Scope.Model(args).to(device).eval()
    source.load_state_dict(state, strict=True)
    baseline_p, baseline_y = assess(source, dataloader['val_loader'], scaler, device)
    baseline_full = metrics(baseline_p, baseline_y)
    assert abs(baseline_full['mae'] - 17.8334026) < 2e-4
    split = len(baseline_p)//2
    selection, assessment = slice(0, split-12), slice(split, len(baseline_p))
    baseline_prefix = metrics(baseline_p[selection], baseline_y[selection])['mae']
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    prediction_cache = {}
    for name in ('predictor_control', 'neighbor'):
        report, prediction, truth = train_arm(name, args, state, graph, dataloader, scaler, device,
                                              cli.epochs, cli.lr, baseline_prefix,
                                              cli.output.parent, logger)
        reports[name] = report
        prediction_cache[name] = prediction
        assert np.array_equal(baseline_y, truth)
    base_suffix = metrics(baseline_p[assessment], baseline_y[assessment])['mae']
    for name, report in reports.items():
        candidate = prediction_cache[name][assessment]
        report['assessment_improvement_vs_original'] = base_suffix - report['assessment']['mae']
        report['three_chronological_block_improvements'] = [
            metrics(baseline_p[assessment][ix], baseline_y[assessment][ix])['mae'] -
            metrics(candidate[ix], baseline_y[assessment][ix])['mae']
            for ix in np.array_split(np.arange(len(candidate)), 3)]
    neighbor = reports['neighbor']
    control = reports['predictor_control']
    pass_gate = (neighbor['selected_epoch'] > 0 and
                 neighbor['assessment']['mae'] <= base_suffix - .01 and
                 neighbor['assessment']['mae'] <= control['assessment']['mae'] - .01 and
                 sum(g > 0 for g in neighbor['three_chronological_block_improvements']) >= 2)
    result = {'protocol': 'Train-only fixed neighbor graph; same Student checkpoint and 8-epoch MAE fine-tune for both arms; Validation prefix selects epoch; 12-origin embargo; suffix assesses; Test not evaluated',
              'limits': ['Source Student selected on full Validation, so suffix is a mechanism screen only.',
                         'Only one seed and one learning rate were tested; novelty and multi-dataset generality are not established.'],
              'source_checkpoint': str(EXPERIMENT / 'best.pth'),
              'source_checkpoint_sha256': hashlib.sha256((EXPERIMENT / 'best.pth').read_bytes()).hexdigest(),
              'train_graph_k': 4, 'train_graph_delta_correlation': graph_corr,
              'source_full_validation': baseline_full,
              'source_selection_mae': baseline_prefix, 'source_assessment_mae': base_suffix,
              'selection_endpoints': [int(dataloader['val_loader'].idx[0]), int(dataloader['val_loader'].idx[split-13])],
              'assessment_endpoints': [int(dataloader['val_loader'].idx[split]), int(dataloader['val_loader'].idx[-1])],
              'arms': reports,
              'optimizer': {'predictor_learning_rate': cli.lr, 'neighbor_path_learning_rate': 1e-3},
              'predeclared_gate': 'Neighbor improves original and matched predictor-control suffix MAE by >=0.01, with positive gains in >=2/3 chronological blocks.',
              'passed': bool(pass_gate)}
    cli.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('Wrote %s; passed=%s', cli.output, pass_gate)


if __name__ == '__main__':
    main()
