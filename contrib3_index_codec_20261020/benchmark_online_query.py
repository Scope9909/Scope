"""Check lossless query equivalence and measure retrieval time on real histories."""
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

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'))
cli = parser.parse_args()
sys.argv = ['benchmark', cli.dataset]
spec = importlib.util.spec_from_file_location('online_codec', HERE / 'eval_online_srm_codec.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
runner = module.runner
chosen = {'PEMS03': '22', 'PEMS04': '11', 'PEMS07': '117',
          'PEMS08': '7_14_19'}[cli.dataset]


def main():
    experiment, _, _, original_path, _ = runner.inputs()
    out = HERE / 'packages' / 'online'
    lossless_path = out / f'{cli.dataset.lower()}_s1_test_best_lossless.npz'
    coded_path = out / f'{cli.dataset.lower()}_s1_test_fixed_{chosen}_idx.npz'
    with (experiment / 'config.pkl').open('rb') as stream:
        args = pickle.load(stream)
    args.device = 'cpu'
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
    loaders, _, scaler, _ = runner.datasets_utils.load_dataset(
        args, logging.getLogger('query_benchmark'))
    original = runner.OnlineResidualMemory.from_npz(original_path)
    lossless = module.load_memory(lossless_path)
    coded = module.load_memory(coded_path)
    if not (np.array_equal(original.train_raw, lossless.train_raw)
            and np.array_equal(original.idx_train, lossless.idx_train)
            and np.array_equal(original.quant_scale, lossless.quant_scale)):
        raise AssertionError('Lossless state differs after decoding')
    loader = loaders['test_loader']
    loader.bs = 32
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / 32))
    batches = []
    for loader_input, _ in loader.get_iterator():
        x, x_time, *extra = runner.unpack_dataloader_input(loader_input)
        history = np.rint(scaler.inverse_transform(x)[..., 0][
            :, :, original.node_ids]).astype(np.float32)
        tod = np.asarray(x_time[:, -1, 0], dtype=np.int16)
        batches.append((history, tod))
        if len(batches) == 4:
            break
    base = [original.query(*batch) for batch in batches]
    exact = [lossless.query(*batch) for batch in batches]
    for base_result, exact_result in zip(base, exact):
        if not all(np.array_equal(a, b) for a, b in zip(base_result, exact_result)):
            raise AssertionError('Lossless retrieval output differs')
    memories = {'original': original, 'lossless': lossless, 'sensitivity': coded}
    timings = {}
    for name, memory in memories.items():
        for batch in batches:
            memory.query(*batch)
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for batch in batches:
                memory.query(*batch)
            samples.append((time.perf_counter() - start) * 1000 / len(batches))
        timings[name] = statistics.median(samples)
    result = dict(dataset=cli.dataset, batch_size=32, measured_batches=len(batches),
                  repeats=5, lossless_outputs_exact=True,
                  median_query_ms_per_batch=timings,
                  caveat='CPU retrieval only; excludes student model and calibration')
    target = out / f'{cli.dataset.lower()}_query_benchmark.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
