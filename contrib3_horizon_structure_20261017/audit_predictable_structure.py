"""Bounded horizon-structure screen, not a proposed residual-correction method.

Fit all statistics/probes on Train. Select probe hyperparameters on Validation
prefix and assess once on its suffix, separated by a 12-origin embargo.
The source Student has seen all Train labels and was selected on full Validation:
this is not an out-of-fold or confirmatory experiment. Test is not evaluated.
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

from models import TLAST_improved
from utils import datasets_utils
from contrib3_horizon_conflict_20261016.audit_horizon_gradients import tensorize_model_input

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'EXP_experiments/TLAST_improved/PEMS04/TLAST_improved_PEMS04_12to12_exppems04_joint3_lite_r256_strictalign_s1'


def metrics(pred, truth):
    # Arrays are (origins, nodes, horizons). Match official mean-horizon MAE.
    mask = np.isfinite(truth) & (truth != 0)
    error = np.where(mask, np.abs(pred - truth), 0).astype(np.float64)
    counts = mask.sum(axis=(0, 1))
    horizon = error.sum(axis=(0, 1)) / counts
    return {'mae': float(horizon.mean()), 'horizon_mae': horizon.tolist(),
            'element_mae': float(error.sum() / mask.sum()), 'valid_counts': counts.tolist()}


def correlation_stats(residual, truth):
    complete = (np.isfinite(truth) & (truth != 0)).all(axis=-1)
    rows = residual[complete].astype(np.float64)
    covariance = np.cov(rows, rowvar=False)
    corr = np.corrcoef(rows, rowvar=False)
    eig = np.linalg.eigvalsh(covariance)[::-1]
    off = corr[np.triu_indices(12, 1)]
    return {'complete_rows': len(rows), 'mean_offdiagonal_correlation': float(off.mean()),
            'adjacent_correlation': float(np.diag(corr, 1).mean()),
            'first_component_variance_fraction': float(eig[0] / eig.sum()),
            'first_four_variance_fraction': float(eig[:4].sum() / eig.sum()),
            'correlation_matrix': corr.tolist()}


def subset_loader(loader, positions):
    result = copy.copy(loader)
    # Every sample-aligned field must follow exactly the same subset.
    for name in ('idx', 'temporal_position', 'retrieval_data', 'exact_retrieval_data',
                 'exact_available_data', 'teacher_predictions_data',
                 'analog_candidate_index_data', 'analog_candidate_code_data',
                 'analog_candidate_parent_position'):
        value = getattr(result, name, None)
        if value is not None:
            setattr(result, name, value[positions])
    result.size = len(positions)
    result.drop_last = False
    result.num_batch = int(np.ceil(result.size / result.bs))
    return result


def extract(model, loader, scaler, projection, device, logger):
    predictions, truths, features = [], [], []
    with torch.inference_mode():
        for step, (inputs, targets) in enumerate(loader.get_iterator()):
            y, y_time = targets
            model_input = tensorize_model_input(inputs, y_time, device)
            output, _ = model(model_input)
            prediction = scaler.inverse_transform(output)[..., 0].transpose(1, 2)
            hidden = model.time_wise_predictor.last_shared_hidden
            hidden_features = hidden @ projection
            history = model_input[0][..., 0].transpose(1, 2)
            delta_prediction = output[..., 0].transpose(1, 2) - history[..., -1:]
            calendar = model_input[1][:, -1].float()
            phase = torch.stack((calendar[:, 0] * (2 * np.pi / 288),
                                 calendar[:, 1] * (2 * np.pi / 7)), dim=-1)
            periodic = torch.cat((phase.sin(), phase.cos()), dim=-1)
            periodic = periodic[:, None, :].expand(-1, history.shape[1], -1)
            x = torch.cat((history, delta_prediction, periodic, hidden_features), dim=-1)
            predictions.append(prediction.cpu().numpy())
            truths.append(np.asarray(y, dtype=np.float32)[..., 0].transpose(0, 2, 1))
            features.append(x.cpu().numpy())
            if step % 80 == 0:
                logger.info('Extracted batch %d/%d', step + 1, loader.num_batch)
    p, y, x = (np.concatenate(values, axis=0) for values in (predictions, truths, features))
    assert len(p) == len(loader.idx)
    check = np.arange(0, len(p), max(1, len(p) // 40))
    expected = loader.data[loader.idx[check, None] + np.arange(1, 13)[None, :], :, 0]
    assert np.allclose(y[check], expected.transpose(0, 2, 1), atol=1e-3), 'Target/index mismatch'
    assert np.isfinite(p).all() and np.isfinite(x).all()
    return p, y, x


def ridge(x, residual, regularizer):
    design = np.concatenate((x.astype(np.float64), np.ones((len(x), 1))), axis=1)
    gram = design.T @ design / len(x)
    cross = design.T @ residual.astype(np.float64) / len(x)
    penalty = np.eye(design.shape[1]) * regularizer
    penalty[-1, -1] = 0
    return np.linalg.solve(gram + penalty, cross).astype(np.float32)


def apply_ridge(x, weight):
    return x @ weight[:-1] + weight[-1]


def paired_blocks(base, candidate, truth, endpoints):
    blocks = []
    for day in np.unique(endpoints // 288):
        ix = endpoints // 288 == day
        blocks.append({'day_index': int(day), 'origins': int(ix.sum()),
                       'mae_improvement': metrics(base[ix], truth[ix])['mae'] - metrics(candidate[ix], truth[ix])['mae']})
    # Descriptive only: few days and overlapping windows preclude a strong claim.
    return blocks


def mae_probe(fit_x, fit_r, val_x, val_p, val_y, basis, device, logger):
    """Same observable features, but optimize MAE to check the Ridge mismatch.

    The zero predictor is included as epoch zero. Only the Validation prefix
    selects epochs. Every candidate sees identical minibatches and update count.
    """
    split = len(val_p) // 2
    select, assess = slice(0, split - 12), slice(split, len(val_p))
    results = {}
    for name, dim, structured in (('observed_full', 28, False),
                                  ('hidden_full', fit_x.shape[-1], False),
                                  ('hidden_pca4', fit_x.shape[-1], True)):
        torch.manual_seed(41)
        x = torch.as_tensor(fit_x[:, :dim], device=device)
        r = torch.as_tensor(fit_r, device=device)
        transform = torch.as_tensor(basis.T if structured else np.eye(12, dtype=np.float32), device=device)
        head = torch.nn.Linear(dim, len(transform), device=device)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=1e-4)
        best = metrics(val_p[select], val_y[select])['mae']
        selected_epoch, best_weight, best_bias = 0, None, None
        logs = []
        for epoch in range(1, 9):
            order = torch.randperm(len(x), device=device)
            train_error = 0.
            for indices in order.split(8192):
                optimizer.zero_grad(set_to_none=True)
                loss = (head(x[indices]) @ transform - r[indices]).abs().mean()
                loss.backward()
                optimizer.step()
                train_error += float(loss.detach()) * len(indices)
            weight = (head.weight.T @ transform).detach().cpu().numpy()
            bias = (head.bias @ transform).detach().cpu().numpy()
            score = metrics(val_p[select] + val_x[select, :, :dim] @ weight + bias, val_y[select])['mae']
            logs.append({'epoch': epoch, 'train_mae': train_error / len(x), 'selection_mae': score})
            if score < best:
                best, selected_epoch, best_weight, best_bias = score, epoch, weight.copy(), bias.copy()
        assessed = val_p[assess].copy()
        if selected_epoch:
            assessed += val_x[assess, :, :dim] @ best_weight + best_bias
        score = metrics(assessed, val_y[assess])
        block_gains = [metrics(val_p[assess][ix], val_y[assess][ix])['mae'] - metrics(assessed[ix], val_y[assess][ix])['mae']
                       for ix in np.array_split(np.arange(len(assessed)), 3)]
        results[name] = {'parameters': sum(p.numel() for p in head.parameters()), 'selected_epoch': selected_epoch,
                         'selection_mae': best, 'assessment': score, 'epochs': logs,
                         'mae_improvement': metrics(val_p[assess], val_y[assess])['mae'] - score['mae'],
                         'three_block_improvements': block_gains,
                         'final_weight_norm': float(head.weight.detach().norm())}
        logger.info('Direct-MAE %s selected epoch %d; suffix %.8f; final weight norm %.4f',
                    name, selected_epoch, score['mae'], results[name]['final_weight_norm'])
        del x, r, head, optimizer
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--train-stride', type=int, default=4)
    parser.add_argument('--projection-dim', type=int, default=32)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('pems04_s1_structure.json'))
    cli = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger('horizon-structure')
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
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
    for p in model.parameters():
        p.requires_grad_(False)
    projection = torch.randn(args.d_out, cli.projection_dim, device=device) / np.sqrt(args.d_out)
    train_loader, val_loader = loaders['train_loader'], loaders['val_loader']
    # Remove Train targets overlapping the first Validation target interval.
    train_positions = np.flatnonzero(train_loader.idx + 12 <= val_loader.idx[0])[::cli.train_stride]
    train_loader = subset_loader(train_loader, train_positions)
    logger.info('Train sampled origins=%d; Validation origins=%d', len(train_loader.idx), len(val_loader.idx))
    pt, yt, xt = extract(model, train_loader, scaler, projection, device, logger)
    pv, yv, xv = extract(model, val_loader, scaler, projection, device, logger)
    baseline = metrics(pv, yv)
    assert abs(baseline['mae'] - 17.8334026) < 2e-4, baseline
    logger.info('Reproduced full Validation baseline %.8f', baseline['mae'])

    # Demonstrate that fixed linear horizon mixing adds no function class to
    # arbitrary linear readouts of one shared hidden vector.
    heads = model.time_wise_predictor.predict_unit_list
    w = torch.cat([h.weight for h in heads], dim=0).double()
    b = torch.cat([h.bias for h in heads], dim=0).double()
    h = torch.randn(7, w.shape[1], device=device, dtype=torch.float64)
    mix = torch.eye(12, device=device, dtype=torch.float64) + .02 * torch.randn(12, 12, device=device, dtype=torch.float64)
    folding_error = float(((h @ w.T + b) @ mix.T - (h @ (mix @ w).T + mix @ b)).abs().max())
    assert folding_error < 1e-10

    rt = yt - pt
    complete = (np.isfinite(yt) & (yt != 0)).all(axis=-1)
    fit_x, fit_r = xt[complete], rt[complete]
    mean, scale = fit_x.mean(axis=0), fit_x.std(axis=0).clip(1e-5)
    fit_x = (fit_x - mean) / scale
    xv = (xv - mean) / scale
    residual_mean = fit_r.mean(axis=0)
    covariance = np.cov(fit_r.astype(np.float64), rowvar=False)
    eig, basis = np.linalg.eigh(covariance)
    basis = basis[:, np.argsort(eig)[::-1][:4]].astype(np.float32)
    train_blocks = [correlation_stats(rt[ix], yt[ix]) for ix in np.array_split(np.arange(len(rt)), 3)]
    initial = {'protocol': 'Train-fitted diagnostic probes; Validation prefix selection; 12-origin embargo; suffix assessment; no Test evaluation',
               'limitations': ['Frozen Student fitted on all Train; Train residuals are in-sample, not OOF.',
                              'Source checkpoint selected on full Validation; suffix is not a fresh confirmatory holdout.',
                              'Only a fixed random projection of hidden features is probed; a negative result is not an impossibility proof.',
                              'Shared loader initializes Test-related arrays, but Test loader is never iterated or scored.'],
               'checkpoint': str(EXPERIMENT / 'best.pth'),
               'checkpoint_sha256': hashlib.sha256((EXPERIMENT / 'best.pth').read_bytes()).hexdigest(),
               'train_origins': len(train_loader.idx), 'fit_complete_node_windows': int(complete.sum()),
               'full_validation_baseline': baseline,
               'linear_horizon_mixing_fold_max_abs_error': folding_error,
               'train_error_structure': correlation_stats(rt, yt), 'train_three_blocks': train_blocks,
               'feature_dimensions': {'history_prediction_calendar': 28, 'with_hidden_projection': 28 + cli.projection_dim},
               'predeclared_screen_gate': 'Require suffix MAE improvement >=0.01, positive gain in >=2/3 chronological blocks, and outperform Train-median bias. PCA probe must beat full-output ridge to motivate structured decoding; these are engineering gates, not significance tests.'}
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.with_name('pems04_s1_structure_diagnostic.json').write_text(json.dumps(initial, indent=2), encoding='utf-8')
    logger.info('Train residual adjacent correlation %.4f; first-four variance %.4f', initial['train_error_structure']['adjacent_correlation'], initial['train_error_structure']['first_four_variance_fraction'])

    split = len(pv) // 2
    selection, assessment = slice(0, split - 12), slice(split, len(pv))
    assert val_loader.idx[split - 13] + 12 < val_loader.idx[split] + 1
    base_select = metrics(pv[selection], yv[selection])['mae']
    base_assess = metrics(pv[assessment], yv[assessment])['mae']
    median = np.array([np.median(rt[..., j][yt[..., j] != 0]) for j in range(12)], dtype=np.float32)
    bias_score = metrics(pv[assessment] + median, yv[assessment])['mae']
    results = {}
    choices = {}
    for feature_name, dim in (('observed', 28), ('hidden', fit_x.shape[-1])):
        for target_name in ('full', 'pca4'):
            name = feature_name + '_' + target_name
            records = []
            best = {'selection_mae': base_select, 'alpha': 0., 'ridge': None}
            best_weight = None
            for regularizer in (0.01, 0.1, 1.):
                target = fit_r if target_name == 'full' else (fit_r - residual_mean) @ basis
                weight = ridge(fit_x[:, :dim], target, regularizer)
                if target_name == 'pca4':
                    weight = weight @ basis.T
                    weight[-1] += residual_mean
                correction = apply_ridge(xv[selection, :, :dim], weight)
                for alpha in (0.1, 0.25, 0.5, 1.):
                    score = metrics(pv[selection] + alpha * correction, yv[selection])['mae']
                    record = {'ridge': regularizer, 'alpha': alpha, 'selection_mae': score}
                    records.append(record)
                    if score < best['selection_mae']:
                        best, best_weight = record, weight.copy()
            if best_weight is None:
                assessed = pv[assessment].copy()
            else:
                assessed = pv[assessment] + best['alpha'] * apply_ridge(xv[assessment, :, :dim], best_weight)
            score = metrics(assessed, yv[assessment])
            block_gains = [metrics(pv[assessment][ix], yv[assessment][ix])['mae'] - metrics(assessed[ix], yv[assessment][ix])['mae']
                           for ix in np.array_split(np.arange(len(assessed)), 3)]
            gain = base_assess - score['mae']
            results[name] = {'selection': best, 'selection_grid': records, 'assessment': score,
                             'mae_improvement': gain, 'three_block_improvements': block_gains,
                             'paired_day_blocks': paired_blocks(pv[assessment], assessed, yv[assessment], val_loader.idx[assessment]),
                             'passes_recoverability_screen': bool(gain >= .01 and sum(g > 0 for g in block_gains) >= 2 and score['mae'] < bias_score)}
            choices[name] = best['selection_mae']
            logger.info('%s selected=%s; suffix MAE %.8f gain %.8f blocks=%s', name, best, score['mae'], gain, block_gains)
    selected = min(choices, key=choices.get)
    structure_supported = any(results[group + '_pca4']['passes_recoverability_screen'] and
                              results[group + '_pca4']['assessment']['mae'] < results[group + '_full']['assessment']['mae']
                              for group in ('observed', 'hidden'))
    initial.update({'selection_origins': [int(val_loader.idx[0]), int(val_loader.idx[split-13])],
                    'assessment_origins': [int(val_loader.idx[split]), int(val_loader.idx[-1])],
                    'baseline_assessment': metrics(pv[assessment], yv[assessment]),
                    'train_median_bias_assessment': bias_score, 'probes': results,
                    'prefix_selected_probe': selected,
                    'structured_decoder_screen_passed': bool(structure_supported),
                    'decision': 'Consider a separately controlled decoder experiment' if structure_supported else 'Do not launch another decoder based on this screen'})
    # This control was added after the Ridge screen to resolve the specific
    # squared-loss-versus-MAE confound. It is reported separately, not hidden
    # inside the original predeclared Ridge gate.
    initial['mae_controls_protocol'] = 'Added direct-MAE control; 8 epochs, AdamW lr .01, identical batches, prefix-selected epoch including unchanged epoch 0; diagnostic only.'
    initial['mae_controls'] = mae_probe(fit_x, fit_r, xv, pv, yv, basis, device, logger)
    structured = initial['mae_controls']['hidden_pca4']
    plain = initial['mae_controls']['hidden_full']
    mae_supported = (structured['mae_improvement'] >= .01 and
                     sum(g > 0 for g in structured['three_block_improvements']) >= 2 and
                     structured['assessment']['mae'] < min(bias_score, plain['assessment']['mae']))
    initial['structured_decoder_screen_passed'] = bool(structure_supported or mae_supported)
    initial['decision'] = 'Consider a separately controlled decoder experiment' if initial['structured_decoder_screen_passed'] else 'Do not launch another decoder based on this screen'
    cli.output.write_text(json.dumps(initial, indent=2), encoding='utf-8')
    logger.info('Wrote %s; structured screen passed=%s', cli.output, initial['structured_decoder_screen_passed'])


if __name__ == '__main__':
    main()
