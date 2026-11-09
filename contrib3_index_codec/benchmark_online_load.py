"""Measure local online-state decode/load cost, separate from model inference."""
import gc
import importlib.util
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.argv = ['audit', 'PEMS04']
spec = importlib.util.spec_from_file_location('online_codec', HERE / 'eval_online_srm_codec.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OUT = HERE / 'packages' / 'online'


def measure(path, native, repeats=5):
    func = (module.runner.OnlineResidualMemory.from_npz if native else module.load_memory)
    func(path)
    times = []
    peaks = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        begin = time.perf_counter()
        memory = func(path)
        times.append((time.perf_counter()-begin)*1000)
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        assert memory.train_raw.ndim == 2
        del memory
    return dict(bytes=path.stat().st_size, median_load_ms=statistics.median(times),
                peak_python_alloc_bytes=max(peaks), repeats=repeats)


def main():
    nodes = {'PEMS03': '22', 'PEMS04': '11', 'PEMS07': '117',
             'PEMS08': '7_14_19'}
    results = {}
    for dataset, choice in nodes.items():
        source = json.loads((ROOT / 'srm_test_once' /
                             f'{dataset.lower()}_s1_sparse_test_once.json').read_text())
        variants = {
            'original': (Path(source['test_state']), True),
            'best_lossless': (OUT / f'{dataset.lower()}_s1_test_best_lossless.npz', False),
            'sensitivity': (OUT / f'{dataset.lower()}_s1_test_fixed_{choice}_idx.npz', False),
        }
        results[dataset] = {name: measure(*args) for name, args in variants.items()}
        print(dataset, json.dumps(results[dataset]), flush=True)
    (OUT / 'load_benchmark.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
