"""Read-only final-run preflight; no training, pickle loading or downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from prepare_local_inputs import required_paths

ROOT = Path(__file__).resolve().parents[1]


def check_retrieval_caches(root):
    """Check frozen-run caches against the prepared dataset split indices.

    Only small metadata/index arrays are loaded; retrieval tensors are not
    materialized, and no pickle data is accepted.
    """
    failures = []
    try:
        import numpy as np
    except ImportError:
        return ['Cannot inspect retrieval caches without numpy'], 0

    runs = json.loads((root / 'tools/final_runs.json').read_text(encoding='utf-8'))
    checked = 0
    for dataset, run_path in runs.items():
        config_path = root / run_path / 'config.json'
        if not config_path.is_file():
            failures.append(f'Missing frozen run config: {config_path.relative_to(root)}')
            continue
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            failures.append(f'Invalid frozen run config for {dataset}: {exc}')
            continue
        if config.get('dataset_name') != dataset:
            failures.append(f'Frozen run dataset mismatch: {dataset}')

        cache_rel = config.get('analog_retrieval_cache')
        if not cache_rel or cache_rel == 'none':
            failures.append(f'Frozen run has no retrieval cache: {dataset}')
            continue
        cache_path = (root / cache_rel).resolve()
        try:
            cache_path.relative_to(root.resolve())
        except ValueError:
            failures.append(f'Retrieval cache escapes repository: {dataset}')
            continue
        if not cache_path.is_file():
            failures.append(f'Missing frozen retrieval cache: {cache_rel}')
            continue

        index_rel = config.get('index_scaler_path')
        if not index_rel or index_rel == 'none':
            suffix = '_712' if dataset == 'PEMSBAY' else ''
            index_rel = f'datasets/{dataset}/{dataset}_12to12_index_scaler{suffix}.npz'
        index_path = (root / index_rel).resolve()
        try:
            index_path.relative_to(root.resolve())
        except ValueError:
            failures.append(f'Dataset index escapes repository: {dataset}')
            continue
        if not index_path.is_file():
            failures.append(f'Missing dataset index for retrieval cache: {index_rel}')
            continue

        try:
            with np.load(cache_path, allow_pickle=False) as cache, np.load(index_path, allow_pickle=False) as index:
                required = {'k', 'input_length', 'predict_length'}
                for split in ('train', 'val', 'test'):
                    required.update((f'idx_{split}', f'retrieval_{split}',
                                     f'exact_retrieval_{split}', f'exact_available_{split}'))
                missing = required.difference(cache.files)
                if missing:
                    failures.append(f'Retrieval cache missing fields for {dataset}: {", ".join(sorted(missing))}')
                    continue
                if int(cache['k']) != int(config['analog_retrieval_k']):
                    failures.append(f'Retrieval K mismatch for {dataset}')
                for field in ('input_length', 'predict_length'):
                    if int(cache[field]) != int(config[field]):
                        failures.append(f'Retrieval {field} mismatch for {dataset}')
                for split in ('train', 'val', 'test'):
                    key = f'idx_{split}'
                    if key not in index.files or not np.array_equal(cache[key], index[key]):
                        failures.append(f'Retrieval {split} split mismatch for {dataset}')
                checked += 1
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f'Cannot inspect retrieval cache for {dataset}: {exc}')
    return failures, checked


def main():
    failures = []
    for package in ('numpy', 'torch', 'einops', 'pandas'):
        if importlib.util.find_spec(package) is None:
            failures.append('Missing Python package: ' + package)
    manifest = json.loads((ROOT / 'tools/final_assets_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = ROOT / row['path']
        if not path.is_file():
            failures.append('Missing final asset: ' + row['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            failures.append('Final asset hash mismatch: ' + row['path'])
    for ds in ('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'PEMSBAY'):
        index = ds + '_12to12_index_scaler' + ('_712' if ds == 'PEMSBAY' else '') + '.npz'
        for name in ('data.npz', ds + '_time.npz', index):
            if not (ROOT / 'datasets' / ds / name).is_file():
                failures.append('Missing prepared data: datasets/' + ds + '/' + name)
    for item in manifest.get('unresolved', []):
        failures.append('Unresolved source asset: ' + item)
    for rel in required_paths():
        if not (ROOT / rel).is_file():
            failures.append('Missing replay input: ' + rel)
    local = ROOT / '_preparation/local_inputs_manifest.json'
    if local.exists():
        for row in json.loads(local.read_text(encoding='utf-8'))['files']:
            path = ROOT / row['path']
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                failures.append('Local input hash mismatch: ' + row['path'])
    cache_failures, cache_count = check_retrieval_caches(ROOT)
    failures.extend(cache_failures)
    for failure in failures:
        print('FAIL:', failure)
    print('Final assets checked:', len(manifest['files']))
    print('Frozen retrieval caches checked:', cache_count)
    print('NOT A FULL REPRODUCTION CHECK: numerical outputs and end-to-end deployment still require validation.')
    print('BLOCKED' if failures else 'STATIC CHECKS PASSED; complete runtime validation still required')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
