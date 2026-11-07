"""Time frozen Student + online SRM on the official Test query stream.

Measures retrieval plus student forward pass. Calibration fitting is offline
and is not included in these timings.
"""
import argparse
import importlib.util
import json
import logging
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'))
cli = parser.parse_args()
sys.argv = ['benchmark', cli.dataset]
spec = importlib.util.spec_from_file_location('online_codec', HERE / 'eval_online_srm_codec.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runner = module.runner
nodes = {'PEMS03': '22', 'PEMS04': '11', 'PEMS07': '117',
         'PEMS08': '7_14_19'}[cli.dataset]


def main():
    experiment, _, _, original_path, _ = runner.inputs()
    out = HERE / 'packages' / 'online'
    paths = {
        'original': original_path,
        'best_lossless': out / f'{cli.dataset.lower()}_s1_test_best_lossless.npz',
        'sensitivity': out / f'{cli.dataset.lower()}_s1_test_fixed_{nodes}_idx.npz',
    }
    with (experiment / 'config.pkl').open('rb') as stream:
        args = pickle.load(stream)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.batch_size = 32
    args.cache_data_path = 'datasets'
    args.prob_factor = getattr(args, 'prob_factor', 5)
    if isinstance(getattr(args, 'k_hops', '1,2'), str):
        args.k_hops = [int(value) for value in args.k_hops.split(',')]
    args.pearson_adj = None
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = 'none'
    args.kd_alpha = 0
    loaders, node_count, scaler, _ = runner.datasets_utils.load_dataset(
        args, logging.getLogger('system_inference_benchmark'))
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = node_count
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(
        experiment / 'best.pth', map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False
    index_path = (runner.ROOT / 'datasets' / cli.dataset /
                  f'{cli.dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz')
    with np.load(index_path, allow_pickle=False) as archive:
        endpoints = np.asarray(archive['idx_test'], dtype=np.int64)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def collect(name):
        memory = (runner.OnlineResidualMemory.from_npz(paths[name]) if name == 'original'
                  else module.load_memory(paths[name]))
        sync()
        start = time.perf_counter()
        prediction, truth = runner.collect(
            loaders['test_loader'], endpoints, model, scaler, memory, device, 32)
        sync()
        return (time.perf_counter()-start, prediction, truth)
    # Warm model kernels and data loader before collecting comparative times.
    _ = collect('original')
    timings = {name: [] for name in paths}
    for order in (('original', 'best_lossless', 'sensitivity'),
                  ('sensitivity', 'best_lossless', 'original')):
        for name in order:
            seconds, _, _ = collect(name)
            timings[name].append(seconds)
            print(name, seconds, flush=True)
    result = dict(dataset=cli.dataset, device=str(device),
                  test_queries=len(endpoints), batch_size=32,
                  measured_path='online retrieval + Student forward + output inverse scaling',
                  excluded='one-time Validation calibration fitting and file load/decode',
                  seconds_by_run=timings,
                  median_seconds={name: statistics.median(values)
                                  for name, values in timings.items()})
    target = out / f'{cli.dataset.lower()}_system_inference_benchmark.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
