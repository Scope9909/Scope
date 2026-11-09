"""User-requested Test re-evaluation of a post-Test lossless storage extension.

Freeze per-field recipes from Validation before packaging/evaluating Test.
Keep the previously selected Q8 nodes, checkpoints and calibration protocol.
This is a reproducibility check, not a new independent blind Test experiment.
"""
from __future__ import annotations

import argparse
import bz2
import gc
import hashlib
import json
import logging
import lzma
import pickle
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaptive_lossless_state_codec as codec
import eval_hybrid_codec_test_once as previous
from composite_q8_delta_codec import decode_composite_online_state
from online_codec_runtime import decode_online_state
from pilot_reversible_predictive_coding import normalize_state, best_bytes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ONLINE = HERE / 'packages' / 'online'
VALIDATION = ONLINE / 'adaptive_lossless_val_v2'


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe(path):
    blob = Path(path).read_bytes()
    n = len(codec.MAGIC)
    length = struct.unpack('<I', blob[n:n+4])[0]
    rows = json.loads(zlib.decompress(blob[n+4:n+4+length]))
    return {r['key']: {k: r[k] for k in ('order', 'predictor', 'layout', 'compression')}
            for r in rows}


def pack_fixed(state, rules):
    rows, payloads = [], []
    compressors = {'raw': lambda x: x, 'zlib': lambda x: zlib.compress(x, 9),
                   'xz': lambda x: lzma.compress(x, preset=6),
                   'bz2': lambda x: bz2.compress(x, compresslevel=9)}
    for key in sorted(state):
        a = np.asarray(state[key])
        rule = rules.get(key, dict(order='C', predictor='none', layout='plain', compression='zlib'))
        data, dtype = codec.transform(a, rule['order'], rule['predictor'], rule['layout'])
        payload = compressors[rule['compression']](data)
        row = dict(key=key, dtype=a.dtype.str, shape=list(a.shape), encoded_dtype=dtype,
                   **rule, length=len(payload), sha256=codec.digest(a))
        rows.append(row); payloads.append(payload)
    return codec.assemble(rows, payloads)


def same(left, right):
    if left.keys() != right.keys():
        raise AssertionError('State fields changed')
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            raise AssertionError(f'State field changed: {key}')


def evaluate(dataset, val_file, test_file):
    runner = previous._load_runner(dataset)
    exp_dir, calibration_path, _, _, selector = runner.inputs()
    with (exp_dir / 'config.pkl').open('rb') as stream:
        args = pickle.load(stream)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.batch_size = 32
    args.cache_data_path = 'datasets'
    args.prob_factor = getattr(args, 'prob_factor', 5)
    if isinstance(getattr(args, 'k_hops', '1,2'), str):
        args.k_hops = [int(v) for v in args.k_hops.split(',')]
    args.pearson_adj = None
    if not (args.use_analog_lowrank_retrieval_residual and args.use_analog_exact_match_residual):
        raise ValueError('Checkpoint does not contain expected adapters')
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = 'none'
    args.kd_alpha = 0
    logger = logging.getLogger('adaptive_lossless_test')
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = runner.datasets_utils.load_dataset(args, logger)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(exp_dir/'best.pth', map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False
    with np.load(ROOT/'datasets'/dataset/f'{dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz', allow_pickle=False) as z:
        val_idx = np.asarray(z['idx_val'], dtype=np.int64)
        test_idx = np.asarray(z['idx_test'], dtype=np.int64)
    val_memory = runner.OnlineResidualMemory(codec.load_runtime_state(val_file))
    test_memory = runner.OnlineResidualMemory(codec.load_runtime_state(test_file))
    if val_memory.cold_start_endpoint != int(val_idx[0]) or test_memory.cold_start_endpoint is not None:
        raise ValueError('Unexpected state alignment')
    print(dataset, 'Validation forward for frozen calibration', flush=True)
    val_pred, val_truth = runner.collect(loaders['val_loader'], val_idx, model, scaler, val_memory, device, 32)
    print(dataset, 'Test forward', flush=True)
    test_pred, test_truth = runner.collect(loaders['test_loader'], test_idx, model, scaler, test_memory, device, 32)
    if len(test_truth) != len(test_idx) or len(val_truth) != len(val_idx):
        raise AssertionError('Repeated padding samples included')
    parameters = json.loads(calibration_path.read_text(encoding='utf-8'))
    kwargs = {k: parameters[k] for k in ('num_bins', 'smooth_shrinkage', 'num_time_bins', 'time_shrinkage', 'slices_per_day')}
    _, calibrated, _ = runner.fit_and_apply(val_pred, val_truth, val_idx, test_pred, test_idx, **kwargs)
    return dict(dataset=dataset, seed=1, selector=selector, device=str(device),
                test_raw_metrics=runner.metrics(test_pred, test_truth),
                test_calibrated_metrics=runner.metrics(calibrated, test_truth),
                validation_count=len(val_truth), test_count=len(test_truth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    cli = ap.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    original_manifest = json.loads(previous.MANIFEST.read_text(encoding='utf-8'))
    original_results = {r['dataset']: r for r in json.loads(previous.REPORT.read_text(encoding='utf-8'))['results']}
    rules = {ds: {kind: recipe(VALIDATION/f'{ds.lower()}_val_{kind}.alsc')
                  for kind in ('selected', 'exact', 'generic')} for ds in previous.CASES}
    freeze = dict(scope='User-requested Test reproducibility check for new lossless coding',
                  not_independent_blind_test=True, test_metric_tuning=False,
                  test_byte_based_recipe_selection=False, q8_nodes_changed=False,
                  rules=rules, unseen_field_rule=dict(order='C', predictor='none', layout='plain', compression='zlib'),
                  codec_sha256=checksum(Path(codec.__file__)), evaluator_sha256=checksum(__file__),
                  old_manifest=str(previous.MANIFEST), original_jobs=original_manifest['jobs'])
    (cli.output/'frozen_recipes.json').write_text(json.dumps(freeze, indent=2), encoding='utf-8')
    # Recipes are already written before reading any Test state or targets.
    package_rows = {}
    for ds, case in previous.CASES.items():
        exact_source = ONLINE / f'{ds.lower()}_s1_test_u16.npz'
        exact, _ = normalize_state(exact_source)
        source = ONLINE / case['test_source']
        if case['kind'] == 'mixed':
            with np.load(source, allow_pickle=False) as z:
                selected = {k: z[k] for k in z.files}
        else:
            selected = exact
        sizes = {}
        for kind, state in (('selected', selected), ('exact', exact), ('generic', exact)):
            blob = pack_fixed(state, rules[ds][kind])
            path = cli.output/f'{ds.lower()}_test_{kind}.alsc'
            path.write_bytes(blob)
            same(state, codec.decode(path.read_bytes()))
            sizes[kind] = len(blob)
        old = ONLINE / case['test_package']
        old_runtime = decode_composite_online_state(old) if case['kind']=='mixed' else decode_online_state(old)
        new_path = cli.output/f'{ds.lower()}_test_selected.alsc'
        same(old_runtime, codec.load_runtime_state(new_path))
        baseline, _ = best_bytes(exact)
        package_rows[ds] = dict(package_bytes=sizes, legacy_baseline_bytes=len(baseline),
                                source_sha256=checksum(source), package_sha256=checksum(new_path),
                                runtime_state_bitwise_equal=True)
        print(ds, 'Test packages', sizes, 'state equality passed', flush=True)
    (cli.output/'packages.json').write_text(json.dumps(package_rows, indent=2), encoding='utf-8')
    results = []
    for ds in previous.CASES:
        job = next(j for j in original_manifest['jobs'] if j['dataset'] == ds)
        for key in ('checkpoint', 'calibration'):
            if checksum(job[key]) != job[key+'_sha256']:
                raise ValueError(f'{ds} frozen {key} changed')
        t0 = time.perf_counter()
        row = evaluate(ds, VALIDATION/f'{ds.lower()}_val_selected.alsc', cli.output/f'{ds.lower()}_test_selected.alsc')
        row.update(package_rows[ds])
        row['raw_mae_change_vs_previous'] = row['test_raw_metrics']['mae'] - original_results[ds]['test_raw_mae']
        row['calibrated_mae_change_vs_previous'] = row['test_calibrated_metrics']['mae'] - original_results[ds]['test_val_calibrated_mae']
        row['elapsed_seconds'] = time.perf_counter()-t0
        (cli.output/f'{ds.lower()}_metrics.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        results.append(row)
        report = dict(scope=freeze['scope'], not_independent_blind_test=True,
                      test_metric_tuning=False, results=results,
                      complete=len(results)==4)
        (cli.output/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(ds, 'FINISHED', row['test_raw_metrics'], row['test_calibrated_metrics'], flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
