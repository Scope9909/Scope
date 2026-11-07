"""Train-state-only feasibility audit; never reads Validation/Test labels."""
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / 'packages' / 'exact_dictionary_pilot'
SOURCE = ROOT / 'paper_results/icde_retrieval_free_pilot_20261012/online_state_budget_pems04_s1_cap2905847_20261013/online_train_state.npz'


def serialize(state):
    stream = io.BytesIO()
    np.savez_compressed(stream, **state)
    return stream.getvalue()


def best_layout(state):
    variants = []
    for order in ('C', 'F'):
        arrays = {k: np.array(v, order=order) if v.ndim > 1 else v
                  for k, v in state.items()}
        variants.append((serialize(arrays), order))
    return min(variants, key=lambda v: len(v[0]))


def main():
    with np.load(SOURCE, allow_pickle=False) as archive:
        source = {k: archive[k] for k in archive.files}
    raw = source['train_raw']
    assert np.isfinite(raw).all() and np.array_equal(raw, np.rint(raw))
    assert raw.min() >= 0 and raw.max() <= 65535
    raw = raw.astype(np.uint16)
    endpoints = source['idx_train'].astype(np.int64)
    tod = source['train_tod']
    il = int(source['input_length'])
    pl = int(source['predict_length'])
    min_valid = int(source['exact_min_valid_history'])
    assert np.all(np.diff(endpoints) == 1)
    common = {k: v for k, v in source.items() if k not in ('train_raw', 'idx_train')}
    common['idx_train_start'] = np.asarray(endpoints[0], dtype=np.int32)
    common['idx_train_count'] = np.asarray(len(endpoints), dtype=np.int32)
    # Preserve the original scale, including dtype: the baseline is truly lossless.
    baseline, baseline_order = best_layout(dict(common, train_raw=raw))
    hist_index = endpoints[:, None] + np.arange(-il + 1, 1)[None]
    fut_index = endpoints[:, None] + np.arange(1, pl + 1)[None]
    histories, times, sums, counts, pointers = [], [], [], [], [0]
    groups_by_node = []
    checked = 0
    rng = np.random.default_rng(20261021)
    for node in range(raw.shape[1]):
        history = raw[hist_index, node]
        future = raw[fut_index, node]
        keys = np.column_stack((tod, history)).astype(np.uint16)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        total = np.zeros((len(unique), pl), dtype=np.uint64)
        count = np.zeros((len(unique), pl), dtype=np.uint32)
        np.add.at(total, inverse, future)
        np.add.at(count, inverse, future != 0)
        assert total.max() <= np.iinfo(np.uint32).max
        assert count.max() <= np.iinfo(np.uint16).max
        # Verify masked matching against the original individual candidates.
        # Include altered zero masks; a hash-only exact lookup would fail here.
        samples = rng.choice(len(endpoints), size=min(64, len(endpoints)), replace=False)
        for position in samples:
            for masked in (False, True):
                query = history[position].copy()
                if masked:
                    query[::3] = 0
                original_valid = (history != 0) & (query[None] != 0)
                matches = ((original_valid.sum(axis=1) >= min_valid)
                           & np.all((history == query[None]) | ~original_valid, axis=1)
                           & (tod == tod[position]))
                dict_history = unique[:, 1:]
                dict_valid = (dict_history != 0) & (query[None] != 0)
                dict_matches = ((dict_valid.sum(axis=1) >= min_valid)
                                & np.all((dict_history == query[None]) | ~dict_valid, axis=1)
                                & (unique[:, 0] == tod[position]))
                assert np.array_equal(future[matches].sum(axis=0, dtype=np.uint64),
                                      total[dict_matches].sum(axis=0, dtype=np.uint64))
                assert np.array_equal((future[matches] != 0).sum(axis=0),
                                      count[dict_matches].sum(axis=0, dtype=np.uint64))
                checked += 1
        histories.append(unique[:, 1:])
        times.append(unique[:, 0])
        sums.append(total.astype(np.uint32))
        counts.append(count.astype(np.uint16))
        pointers.append(pointers[-1] + len(unique))
        groups_by_node.append(len(unique))
    exact = dict(common, exact_history=np.concatenate(histories),
                 exact_tod=np.concatenate(times), exact_future_sum=np.concatenate(sums),
                 exact_future_count=np.concatenate(counts),
                 exact_node_pointers=np.asarray(pointers, dtype=np.uint32),
                 codec_version=np.asarray(1, dtype=np.uint8))
    exact_blob, exact_order = best_layout(exact)
    scale = np.where(raw.max(axis=0) > 0, raw.max(axis=0) / 255., 1.).astype(np.float32)
    codes = np.clip(np.rint(raw / scale[None]), 0, 255).astype(np.uint8)
    combined_blob, combined_order = best_layout(dict(exact, nearest_codes=codes,
                                                    nearest_scale=scale))
    # Verify independent archive roundtrip for all fields, not just array sizes.
    with np.load(io.BytesIO(combined_blob), allow_pickle=False) as decoded:
        for key, value in exact.items():
            assert np.array_equal(decoded[key], value), key
    with np.load(io.BytesIO(baseline), allow_pickle=False) as decoded:
        assert np.array_equal(decoded['train_raw'], source['train_raw'])
        assert np.array_equal(np.arange(int(decoded['idx_train_start']),
                                       int(decoded['idx_train_start']) + int(decoded['idx_train_count'])), endpoints)
        for key, value in common.items():
            assert np.array_equal(decoded[key], value), key
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'pems04_lossless_train.npz').write_bytes(baseline)
    (OUT / 'pems04_exact_dictionary_train.npz').write_bytes(exact_blob)
    (OUT / 'pems04_exact_dictionary_q8_train.npz').write_bytes(combined_blob)
    result = dict(dataset='PEMS04', scope='immutable Train state only; no held-out labels read',
                  original_bytes=SOURCE.stat().st_size, lossless_baseline_bytes=len(baseline),
                  exact_dictionary_bytes=len(exact_blob), combined_q8_bytes=len(combined_blob),
                  layouts=dict(baseline=baseline_order, exact=exact_order, combined=combined_order),
                  original_candidate_node_pairs=len(endpoints) * raw.shape[1],
                  unique_groups=sum(groups_by_node), groups_by_node=groups_by_node,
                  masked_match_checks=checked, checks_passed=True,
                  storage_gate_passed=len(combined_blob) < len(baseline),
                  validation_evaluated=False, test_evaluated=False,
                  limitation='Cold-start pending-history support not implemented; would require additional state.')
    (OUT / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
