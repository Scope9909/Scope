"""Copy frozen inputs from an existing trusted workspace; preview by default.

Never regenerates data or runs a model. Conflicting destinations are rejected.
Raw data and legacy predictions are local-only, not GitHub release assets.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def literal(name, path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(name + ' not found')


def required_paths():
    paths = set()
    for ds in ('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'PEMSBAY'):
        index = ds + '_12to12_index_scaler' + ('_712' if ds == 'PEMSBAY' else '') + '.npz'
        paths.update('datasets/' + ds + '/' + n for n in ('data.npz', ds + '_time.npz', index))
    configs = literal('CONFIGS', ROOT / 'srm_test_once_20261014/run_frozen_srm_test_once.py')
    for ds, c in configs.items():
        paths.add(c['calibration'].format(seed=1))
        paths.add(c['validation_state_root'] + '/' + c['validation_state_pattern'].format(mode='sparse'))
        paths.add(c['test_states']['sparse'] if 'test_states' in c else
                  c['test_state_root'] + '/' + c['test_state_pattern'].format(mode='sparse'))
    base = 'contrib3_index_codec_20261020/packages/online/'
    cases = literal('CASES', ROOT / 'contrib3_index_codec_20261020/eval_hybrid_codec_test_once.py')
    for ds, c in cases.items():
        paths.add(base + ds.lower() + '_s1_test_u16.npz')
        paths.update(base + c[k] for k in ('test_source', 'test_package', 'roundtrip_test_source'))
    paths.update((
        'paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal/calibration.json',
        'paper_results/pemsbay712_srm_validation_pilot_20261027/validation_screen.json',
        'paper_results/pemsbay712_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz',
        'pemsbay_full_20261027/full_validation_replayed.npz',
        'pemsbay_full_20261027/validation_predictions.npz',
    ))
    return sorted(paths)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    if source == ROOT.resolve():
        parser.error('Source must differ from the destination repository')
    rows = []
    for rel in required_paths():
        src, dst = source / rel, ROOT / rel
        src.resolve().relative_to(source)
        dst.resolve().relative_to(ROOT.resolve())
        if not src.is_file():
            raise FileNotFoundError(src)
        h = digest(src)
        if dst.exists() and digest(dst) != h:
            raise RuntimeError('Conflicting destination; refusing overwrite: ' + rel)
        rows.append({'path': rel, 'sha256': h, 'size': src.stat().st_size, 'copy_needed': not dst.exists()})
    print(json.dumps({'files': len(rows), 'copy_count': sum(x['copy_needed'] for x in rows),
                      'copy_bytes': sum(x['size'] for x in rows if x['copy_needed']), 'apply': args.apply}, indent=2))
    if not args.apply:
        return
    for row in rows:
        src, dst = source / row['path'], ROOT / row['path']
        if row['copy_needed']:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        if digest(dst) != row['sha256'] or digest(src) != row['sha256']:
            raise RuntimeError('Hash changed: ' + row['path'])
    out = ROOT / '_preparation/local_inputs_manifest.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'files': rows, 'scope': 'local-only prepared inputs; no model evaluation'}, indent=2), encoding='utf-8')
    print('Copied and verified. No model execution or remote publication occurred.')


if __name__ == '__main__':
    main()
