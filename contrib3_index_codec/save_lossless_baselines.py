"""Materialize and verify the fair lossless Train-state baselines."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exact_dictionary import best_layout

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'packages' / 'online'


def main():
    rows = []
    for dataset in ('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'):
        manifest = json.loads((ROOT / 'srm_test_once' /
                               f'{dataset.lower()}_s1_sparse_test_once.json').read_text())
        source = Path(manifest['test_state'])
        with np.load(source, allow_pickle=False) as archive:
            state = {key: archive[key] for key in archive.files}
        original = state.copy()
        raw = state.pop('train_raw')
        if not (np.isfinite(raw).all() and np.array_equal(raw, np.rint(raw))
                and raw.min() >= 0 and raw.max() <= 65535):
            raise ValueError(f'{dataset}: raw history is not uint16-exact')
        state['train_raw'] = raw.astype(np.uint16)
        idx = state.pop('idx_train')
        if not np.array_equal(idx, np.arange(idx[0], idx[0]+len(idx))):
            raise ValueError(f'{dataset}: candidate indices are not contiguous')
        state['idx_train_start'] = np.asarray(idx[0], dtype=np.int32)
        state['idx_train_count'] = np.asarray(len(idx), dtype=np.int32)
        blob, order = best_layout(state)
        target = OUT / f'{dataset.lower()}_s1_test_best_lossless.npz'
        target.write_bytes(blob)
        with np.load(target, allow_pickle=False) as archive:
            decoded = {key: archive[key] for key in archive.files}
        start = int(decoded.pop('idx_train_start'))
        count = int(decoded.pop('idx_train_count'))
        decoded['idx_train'] = np.arange(start, start+count, dtype=np.int32)
        if decoded.keys() != original.keys():
            raise AssertionError(f'{dataset}: keys changed')
        for key, value in original.items():
            if not np.array_equal(decoded[key], value):
                raise AssertionError(f'{dataset}: {key} changed')
        rows.append(dict(dataset=dataset, original_bytes=source.stat().st_size,
                         lossless_bytes=len(blob), layout=order,
                         all_fields_exact=True, path=str(target)))
    report = OUT / 'best_lossless_baselines.json'
    report.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
