"""Copy the 15 prepared dataset files from a trusted workspace.

Never regenerates data or runs a model. Conflicting destinations are rejected.
Prepared data is local-only and is not a repository release asset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def required_paths():
    paths = set()
    for ds in ('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'PEMSBAY'):
        index = ds + '_12to12_index_scaler' + ('_712' if ds == 'PEMSBAY' else '') + '.npz'
        paths.update('datasets/' + ds + '/' + n for n in ('data.npz', ds + '_time.npz', index))
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
    print('Copied and verified. No model execution or publication occurred.')


if __name__ == '__main__':
    main()
