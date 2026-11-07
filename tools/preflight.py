"""Read-only final-run preflight; no training, pickle loading or downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from prepare_local_inputs import required_paths

ROOT = Path(__file__).resolve().parents[1]


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
    for failure in failures:
        print('FAIL:', failure)
    print('Final assets checked:', len(manifest['files']))
    print('NOT A FULL REPRODUCTION CHECK: dynamic retrieval-cache dependencies and numerical outputs still require validation.')
    print('BLOCKED' if failures else 'STATIC CHECKS PASSED; complete runtime validation still required')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
