"""Validation-only archive experiment; frozen Q8 nodes and old outputs preserved."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np

import adaptive_lossless_state_codec as codec
from pilot_reversible_predictive_coding import normalize_state, best_bytes
from online_codec_runtime import decode_online_state
from composite_q8_delta_codec import decode_composite_online_state

HERE = Path(__file__).resolve().parent
ONLINE = HERE / 'packages' / 'online'
CASES = {'pems03': 'pems03_s1_val_fixed_22_idx.npz',
         'pems04': None, 'pems07': 'pems07_s1_val_fixed_117_idx.npz', 'pems08': None}


def check(left, right):
    assert left.keys() == right.keys()
    for k in left:
        assert left[k].shape == right[k].shape and left[k].dtype == right[k].dtype, k
        assert left[k].tobytes() == right[k].tobytes(), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for ds, mixed_source in CASES.items():
        exact_source = ONLINE / f'{ds}_s1_val_u16.npz'
        exact, _ = normalize_state(exact_source)
        baseline_blob, _ = best_bytes(exact)
        selected = exact
        source = exact_source
        if mixed_source:
            source = ONLINE / mixed_source
            with np.load(source, allow_pickle=False) as archive:
                selected = {k: archive[k] for k in archive.files}
        t0 = time.perf_counter()
        exact_blob, generic_blob, exact_diagnostics = codec.encode(exact)
        print(ds, 'exact', len(exact_blob), 'generic', len(generic_blob), flush=True)
        if mixed_source:
            blob, _, diagnostics = codec.encode(selected)
        else:
            blob, diagnostics = exact_blob, exact_diagnostics
        for name, data, expected in [('selected', blob, selected),
                                     ('exact', exact_blob, exact), ('generic', generic_blob, exact)]:
            path = args.output / f'{ds}_val_{name}.alsc'
            path.write_bytes(data)
            check(expected, codec.decode(path.read_bytes()))
        path = args.output / f'{ds}_val_selected.alsc'
        old_package = (ONLINE / f'{ds}_s1_val_q8_delta_pilot.npz' if mixed_source
                       else ONLINE / 'bitpacked_val_20261023' / f'{ds}_s1_val_delta_bitpacked.npz')
        runtime_old = (decode_composite_online_state(old_package) if mixed_source
                       else decode_online_state(old_package))
        runtime_new = codec.load_runtime_state(path)
        check(runtime_old, runtime_new)
        # Independently reading from disk also validates the standalone decoder.
        decode_start = time.perf_counter()
        codec.load_runtime_state(path)
        elapsed_decode = time.perf_counter() - decode_start
        row = dict(dataset=ds.upper(), validation_only=True,
                   source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   exact_source_sha256=hashlib.sha256(exact_source.read_bytes()).hexdigest(),
                   old_package=str(old_package.resolve()),
                   old_package_sha256=hashlib.sha256(old_package.read_bytes()).hexdigest(),
                   old_package_bytes=old_package.stat().st_size,
                   old_package_outer_zlib_bytes=len(zlib.compress(old_package.read_bytes(), 9)),
                   frozen_semantic_choice='mixed_Q8' if mixed_source else 'exact',
                   legacy_baseline_bytes=len(baseline_blob),
                   legacy_baseline_outer_zlib_bytes=len(zlib.compress(baseline_blob, 9)),
                   generic_same_container_bytes=len(generic_blob),
                   new_exact_bytes=len(exact_blob), new_selected_bytes=len(blob),
                   saved_vs_legacy_percent=100*(1-len(blob)/len(baseline_blob)),
                   saved_vs_generic_percent=100*(1-len(blob)/len(generic_blob)),
                   all_fields_byte_exact=True, runtime_state_byte_exact=True,
                   decode_seconds=elapsed_decode, experiment_seconds=time.perf_counter()-t0,
                   package=str(path.resolve()), sha256=hashlib.sha256(blob).hexdigest(),
                   fields=diagnostics)
        rows.append(row)
        report = dict(scope='Validation-only lossless coding exploration',
                      test_used=False, mae_rerun=False, q8_nodes_changed=False,
                      codec_sha256=hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest(),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      results=rows)
        totals = {k: sum(r[k] for r in rows) for k in ('legacy_baseline_bytes',
                  'legacy_baseline_outer_zlib_bytes', 'generic_same_container_bytes',
                  'new_exact_bytes', 'new_selected_bytes', 'old_package_bytes', 'old_package_outer_zlib_bytes')}
        totals['saved_vs_legacy_percent'] = 100*(1-totals['new_selected_bytes']/totals['legacy_baseline_bytes'])
        totals['saved_vs_generic_percent'] = 100*(1-totals['new_selected_bytes']/totals['generic_same_container_bytes'])
        report['totals'] = totals
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(ds, 'selected', len(blob), 'lossless/runtime verified', flush=True)
    print(json.dumps(report['totals'], indent=2))


if __name__ == '__main__':
    main()
