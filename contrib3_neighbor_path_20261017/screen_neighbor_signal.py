"""Train-only probe: do observed correlated neighbors add Student MAE value?

This probe fits a separate tiny residual MLP solely to screen the information
source. It is not the proposed Student architecture or a paper result.
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

from models import TLAST_improved
from utils import datasets_utils
from contrib3_horizon_structure_20261017.audit_predictable_structure import (
    EXPERIMENT, ROOT, extract, metrics, paired_blocks, subset_loader,
)


def graph_from_train(flow, end_of_train, k, seed):
    # All graph statistics use samples whose observed values/labels lie in Train.
    changes = np.diff(flow[:end_of_train].astype(np.float64), axis=0)
    changes -= changes.mean(axis=0)
    scale = np.sqrt(np.square(changes).sum(axis=0)).clip(1e-8)
    corr = (changes.T @ changes) / np.outer(scale, scale)
    np.fill_diagonal(corr, -np.inf)
    neighbors = np.argsort(corr, axis=1)[:, -k:][:, ::-1].copy()
    mean_corr = float(np.mean(corr[np.arange(flow.shape[1])[:, None], neighbors]))
    rng = np.random.default_rng(seed)
    random_neighbors = np.stack([
        rng.choice(np.delete(np.arange(flow.shape[1]), node), size=k, replace=False)
        for node in range(flow.shape[1])
    ])
    random_corr = float(np.mean(corr[np.arange(flow.shape[1])[:, None], random_neighbors]))
    return neighbors, random_neighbors, mean_corr, random_corr


def neighbor_features(flow, endpoints, graph, scaler):
    raw = flow[endpoints[:, None] + np.arange(-11, 1)[None, :]]
    # Avoid materializing (origin, time, node, neighbor) at once.
    mean = np.zeros_like(raw)
    for member in range(graph.shape[1]):
        mean += raw[:, :, graph[:, member]] / graph.shape[1]
    signal = (mean - raw) / float(scaler.std)
    return signal.transpose(0, 2, 1).astype(np.float32)


def make_input(base, neighbor):
    # 28 base features, 12 equal-width neighbor slots, 32 fixed hidden features.
    return np.concatenate((base[..., :28], neighbor, base[..., 28:]), axis=-1)


def tiny_probe(train_x, train_residual, val_x, val_prediction, val_truth,
               device, seed, logger):
    split = len(val_prediction) // 2
    selection, assessment = slice(0, split - 12), slice(split, len(val_prediction))
    torch.manual_seed(seed)
    dim = train_x.shape[-1]
    x = train_x.reshape(-1, dim).copy()
    r = train_residual.reshape(-1, 12).copy()
    valid = np.isfinite(r).all(axis=1)
    x, r = x[valid], r[valid]
    center = x.mean(axis=0)
    scale = x.std(axis=0).clip(1e-5)
    x = torch.as_tensor((x - center) / scale, device=device, dtype=torch.float32)
    r = torch.as_tensor(r, device=device, dtype=torch.float32)
    val = torch.as_tensor((val_x - center) / scale, device=device, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.GELU(),
                              torch.nn.Linear(64, 64), torch.nn.GELU(),
                              torch.nn.Linear(64, 12)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    baseline_selection = metrics(val_prediction[selection], val_truth[selection])['mae']
    best, selected, chosen = baseline_selection, 0, None
    epochs = []
    for epoch in range(1, 9):
        net.train()
        order = torch.randperm(len(x), device=device)
        train_loss = 0.
        for indices in order.split(4096):
            optimizer.zero_grad(set_to_none=True)
            loss = (net(x[indices]) - r[indices]).abs().mean()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(indices)
        net.eval()
        with torch.inference_mode():
            correction = []
            for batch in val[selection].reshape(-1, dim).split(16384):
                correction.append(net(batch).cpu().numpy())
            correction = np.concatenate(correction, axis=0).reshape(val_prediction[selection].shape)
        score = metrics(val_prediction[selection] + correction, val_truth[selection])['mae']
        epochs.append({'epoch': epoch, 'train_mae': train_loss / len(x), 'selection_mae': score})
        logger.info('Epoch %d Train %.6f Selection %.6f', epoch, train_loss / len(x), score)
        if score < best:
            best, selected = score, epoch
            chosen = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
    candidate = val_prediction[assessment].copy()
    if selected:
        net.load_state_dict(chosen)
        net.eval()
        with torch.inference_mode():
            correction = []
            for batch in val[assessment].reshape(-1, dim).split(16384):
                correction.append(net(batch).cpu().numpy())
            candidate += np.concatenate(correction, axis=0).reshape(candidate.shape)
    base_mae = metrics(val_prediction[assessment], val_truth[assessment])['mae']
    value = metrics(candidate, val_truth[assessment])
    block_gains = [metrics(val_prediction[assessment][ix], val_truth[assessment][ix])['mae'] -
                   metrics(candidate[ix], val_truth[assessment][ix])['mae']
                   for ix in np.array_split(np.arange(len(candidate)), 3)]
    return {'parameters': sum(p.numel() for p in net.parameters()),
            'selected_epoch': selected, 'selection_mae': best, 'epochs': epochs,
            'assessment': value, 'mae_improvement': base_mae - value['mae'],
            'three_block_improvements': block_gains,
            'candidate_predictions': candidate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('pems04_s1_neighbor_screen.json'))
    cli = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('neighbor-screen')
    torch.set_num_threads(4)
    torch.manual_seed(41)
    np.random.seed(41)
    args = SimpleNamespace(**json.loads((EXPERIMENT / 'config.json').read_text(encoding='utf-8')))
    args.device, args.batch_size, args.cache_data_path = cli.device, 32, 'datasets'
    args.kd_alpha, args.teacher_predictions_path, args.teacher_predictions_strict_alignment = 0., 'none', 0
    loaders, nodes, scaler, _ = datasets_utils.load_dataset(args, logger)
    args.num_nodes, args.pearson_adj = nodes, None
    args.prob_factor = getattr(args, 'prob_factor', 5)
    if isinstance(args.k_hops, str):
        args.k_hops = [int(v) for v in args.k_hops.split(',')]
    device = torch.device(cli.device)
    model = TLAST_improved.Model(args).to(device).eval()
    model.load_state_dict(torch.load(EXPERIMENT / 'best.pth', map_location=device, weights_only=True), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection = torch.randn(args.d_out, 32, device=device) / np.sqrt(args.d_out)
    original_train, val_loader = loaders['train_loader'], loaders['val_loader']
    train_positions = np.flatnonzero(original_train.idx + 12 <= val_loader.idx[0])[::4]
    train_loader = subset_loader(original_train, train_positions)
    train_prediction, train_truth, train_base = extract(model, train_loader, scaler, projection, device, logger)
    val_prediction, val_truth, val_base = extract(model, val_loader, scaler, projection, device, logger)
    whole = metrics(val_prediction, val_truth)
    assert abs(whole['mae'] - 17.8334026) < 2e-4, whole
    with np.load(ROOT / 'datasets/PEMS04/data.npz', allow_pickle=False) as archive:
        flow = np.asarray(archive['data'][..., 0], dtype=np.float32)
    graph, random_graph, mean_corr, random_corr = graph_from_train(flow, int(original_train.idx.max()) + 13, 4, 41)
    logger.info('Train neighbor delta correlations correlated=%.4f random=%.4f', mean_corr, random_corr)
    # Own-history control has exactly the same input width/parameter count.
    train_groups = {'own': np.zeros((*train_base.shape[:2], 12), dtype=np.float32),
                    'correlated_neighbor': neighbor_features(flow, train_loader.idx, graph, scaler),
                    'random_neighbor': neighbor_features(flow, train_loader.idx, random_graph, scaler)}
    val_groups = {'own': np.zeros((*val_base.shape[:2], 12), dtype=np.float32),
                  'correlated_neighbor': neighbor_features(flow, val_loader.idx, graph, scaler),
                  'random_neighbor': neighbor_features(flow, val_loader.idx, random_graph, scaler)}
    residual = train_truth - train_prediction
    results = {}
    for name in ('own', 'correlated_neighbor', 'random_neighbor'):
        logger.info('Training %s equal-width MAE probe', name)
        result = tiny_probe(make_input(train_base, train_groups[name]), residual,
                            make_input(val_base, val_groups[name]), val_prediction, val_truth,
                            device, 50, logger)
        result['paired_day_blocks'] = paired_blocks(val_prediction[len(val_prediction)//2:],
                                                    result.pop('candidate_predictions'),
                                                    val_truth[len(val_truth)//2:],
                                                    val_loader.idx[len(val_prediction)//2:])
        results[name] = result
        logger.info('%s chosen epoch=%d suffix MAE=%.6f gain=%.6f',
                    name, result['selected_epoch'], result['assessment']['mae'], result['mae_improvement'])
    gains = results['correlated_neighbor']
    pass_gate = (gains['selected_epoch'] > 0 and gains['mae_improvement'] >= .01 and
                 sum(v > 0 for v in gains['three_block_improvements']) >= 2 and
                 gains['assessment']['mae'] < min(results['own']['assessment']['mae'],
                                                 results['random_neighbor']['assessment']['mae']))
    split = len(val_loader.idx) // 2
    result = {'protocol': 'Train-only node graph and probe fit; Validation prefix selects epochs; 12-origin embargo; suffix assessment; Test not evaluated',
              'limitations': ['Student was trained on Train, so Train residuals are in-sample rather than OOF.',
                              'Student checkpoint was selected on full Validation; suffix is a mechanism screen, not fresh confirmation.',
                              'Probe is a separate error-prediction model, not the proposed integrated Student mechanism.',
                              'The common data loader initializes Test arrays but Test loader is never iterated.'],
              'checkpoint': str(EXPERIMENT / 'best.pth'),
              'checkpoint_sha256': hashlib.sha256((EXPERIMENT / 'best.pth').read_bytes()).hexdigest(),
              'train_sampled_origins': len(train_loader.idx), 'train_graph_delta_correlation': mean_corr,
              'random_graph_delta_correlation': random_corr, 'graph_k': 4,
              'full_validation_student_mae': whole['mae'],
              'selection_endpoint_range': [int(val_loader.idx[0]), int(val_loader.idx[split-13])],
              'assessment_endpoint_range': [int(val_loader.idx[split]), int(val_loader.idx[-1])],
              'suffix_baseline_student_mae': metrics(val_prediction[split:], val_truth[split:])['mae'],
              'predeclared_gate': 'Correlated-neighbor suffix improvement >=0.01, positive in >=2/3 chronological blocks, better than equal-width own and random-neighbor controls.',
              'probes': results, 'integrated_student_pilot_warranted': bool(pass_gate)}
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('Wrote %s; integrated pilot warranted=%s', cli.output, pass_gate)


if __name__ == '__main__':
    main()
