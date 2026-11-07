"""Audit lossless codecs for the analog candidate-index cache.

Motivation
----------
The deployment chain stores three payloads per dataset: model weights (already
compressed by the Student contribution), the raw flow series (input data, not
compressible by us), and two precomputed retrieval artefacts:

    candidate_indices  (query, K, node) int32   -- 8x the raw series
    retrieval cache    (query, H, node) x 4     -- 40x the raw series

The retrieval cache is priced by the field-wise law.  The candidate index was
never priced.  It stores global time endpoints in int32 even though the value
range is tiny and the TOD partition is *reconstructible for free* from
``idx_train`` + ``time_data``.  This script prices three lossless codecs and
verifies bit-exact recovery for every one of them.

Codecs
------
    T0 raw        int32 global endpoints, ``-1`` = unavailable      (4 B)
    T1 int16      order-preserving narrowing, ``-1`` sentinel        (2 B)
    T2 uint8      TOD-group offset, 255 = unavailable                (1 B)
    T3 bitpack    T2 offsets packed at ceil(log2(G_max + 1)) bits

T1 needs no side information.  T2/T3 need the TOD partition, which the loader
can rebuild from ``idx_train`` and ``time_data`` with zero stored bytes: each
train endpoint belongs to exactly one TOD group, so a single static table
``position_of_index`` (length = number of timesteps) covers all splits.

Gate
----
Every codec must decode to an array ``np.array_equal`` to the original.  Any
mismatch aborts without reporting numbers.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASETS = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
K_EXPECTED = 8
UNAVAILABLE = 255


def dataset_paths(name):
    base = os.path.join(ROOT, "datasets", name)
    return {
        "indices": os.path.join(
            base, f"{name}_12to12_analog_k8_candidate_indices.npz"),
        "index_scaler": os.path.join(
            base, f"{name}_12to12_index_scaler.npz"),
        "time": os.path.join(base, f"{name}_time.npz"),
    }


def build_tod_groups(time_data, idx_train):
    """Return (train_tod, groups) where groups[tod] is the sorted endpoint list."""
    train_tod = np.asarray(time_data[idx_train, 0], dtype=np.int64)
    groups = {}
    for tod in np.unique(train_tod):
        groups[int(tod)] = np.ascontiguousarray(
            idx_train[train_tod == tod], dtype=np.int64)
    return train_tod, groups


def build_position_table(num_steps, groups):
    """Global endpoint -> position within its TOD group (-1 if not a candidate).

    Zero stored bytes: fully determined by (idx_train, time_data).
    """
    table = np.full(num_steps, -1, dtype=np.int64)
    for endpoints in groups.values():
        table[endpoints] = np.arange(endpoints.size, dtype=np.int64)
    return table


def encode_t2(arr, position_table, sentinel):
    """Group offsets, with ``sentinel`` marking unavailable slots.

    ``sentinel`` must be one past the largest legal offset, so the code fits
    exactly ``ceil(log2(G_max + 1))`` bits and survives bit packing.
    """
    invalid = arr < 0
    positions = position_table[arr]          # -1 where unavailable
    if np.any(positions > sentinel - 1):
        raise AssertionError("group position exceeds the sentinel range")
    return np.where(invalid, sentinel, positions).astype(np.uint8)


def decode_t2(code, query_tod, groups, sentinel):
    out = np.full(code.shape, -1, dtype=np.int32)
    for tod in np.unique(query_tod):
        rows = np.flatnonzero(query_tod == tod)
        block = code[rows]
        group = groups[int(tod)]
        valid = block != sentinel
        if np.any(block[valid] >= group.size):
            raise AssertionError(f"TOD={tod}: offset outside its group")
        out[rows] = np.where(valid, group[np.where(valid, block, 0)], -1)
    return out


def bitpack(values, bits, chunk_rows=200000):
    """Pack the last axis of ``values`` at ``bits`` bits per element.

    ``values`` must be uint8 with the packed axis last, e.g. (query, node, K).
    Returns a uint8 tensor whose last axis holds the packed bytes.
    """
    rows = values.reshape(-1, values.shape[-1])
    count = rows.shape[1]
    weights = np.arange(bits - 1, -1, -1).astype(np.uint8)
    out_cols = (count * bits + 7) // 8
    packed = np.empty((rows.shape[0], out_cols), dtype=np.uint8)
    for start in range(0, rows.shape[0], chunk_rows):
        stop = min(start + chunk_rows, rows.shape[0])
        block = rows[start:stop]
        bit_matrix = ((block[:, :, None] >> weights) & 1).astype(np.uint8)
        packed[start:stop] = np.packbits(
            bit_matrix.reshape(block.shape[0], count * bits), axis=1)
    return packed.reshape(values.shape[:-1] + (out_cols,))


def bitunpack(packed, count, bits, chunk_rows=200000):
    """Inverse of :func:`bitpack`; returns uint8 values with count last."""
    rows = packed.reshape(-1, packed.shape[-1])
    need = count * bits
    weights = (np.uint64(1) << np.arange(bits - 1, -1, -1, dtype=np.uint64))
    out = np.empty((rows.shape[0], count), dtype=np.uint8)
    for start in range(0, rows.shape[0], chunk_rows):
        stop = min(start + chunk_rows, rows.shape[0])
        bits_flat = np.unpackbits(rows[start:stop], axis=1)[:, :need]
        words = bits_flat.reshape(stop - start, count, bits)
        out[start:stop] = (words.astype(np.uint64) * weights).sum(
            axis=2).astype(np.uint8)
    return out.reshape(packed.shape[:-1] + (count,))


def run_dataset(name):
    paths = dataset_paths(name)
    with np.load(paths["index_scaler"], allow_pickle=False) as z:
        idx_train = np.asarray(z["idx_train"], dtype=np.int64)
        idx_val = np.asarray(z["idx_val"], dtype=np.int64)
        idx_test = np.asarray(z["idx_test"], dtype=np.int64)
    with np.load(paths["time"], allow_pickle=False) as z:
        time_data = np.asarray(z["time"])
    num_steps = int(time_data.shape[0])

    train_tod, groups = build_tod_groups(time_data, idx_train)
    if train_tod.size != idx_train.size:
        raise AssertionError("TOD partition lost train entries")
    position_table = build_position_table(num_steps, groups)
    group_sizes = np.array([g.size for g in groups.values()], dtype=np.int64)
    g_max = int(group_sizes.max())
    if g_max > 254:
        raise AssertionError("TOD group too large for a uint8 offset code")
    sentinel = g_max                            # first value past legal offsets
    bits = int(np.ceil(np.log2(g_max + 1)))     # covers 0..g_max inclusive

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    with np.load(paths["indices"], allow_pickle=False) as z:
        arrays = {s: np.asarray(z[f"candidate_index_{s}"], dtype=np.int32)
                  for s in splits}
        k_meta = int(np.asarray(z["k"]).item()) if "k" in z.files else -1

    report = {
        "dataset": name,
        "num_steps": num_steps,
        "num_train": int(idx_train.size),
        "num_tod_groups": int(group_sizes.size),
        "group_size_min": int(group_sizes.min()),
        "group_size_max": g_max,
        "group_size_mean": float(group_sizes.mean()),
        "bits_per_offset": bits,
        "k_meta": k_meta,
        "splits": {},
        "gate": {},
    }

    totals = {"t0": 0, "t1": 0, "t2": 0, "t3": 0}
    for split, query_indices in splits.items():
        arr = arrays[split]
        query_tod = np.asarray(time_data[query_indices, 0], dtype=np.int64)
        n_queries, k, _ = arr.shape
        if k != K_EXPECTED:
            raise AssertionError(f"{split}: K={k}")
        total_slots = arr.size
        unavailable = int((arr < 0).sum())

        # T0
        t0_bytes = int(arr.nbytes)
        # T1
        if arr.max() > 32766:
            raise AssertionError(f"{split}: int16 overflow")
        t1 = arr.astype(np.int16)
        t1_ok = bool(np.array_equal(t1.astype(np.int32), arr))
        t1_bytes = int(t1.nbytes)
        # T2
        t2 = encode_t2(arr, position_table, sentinel)
        t2_back = decode_t2(t2, query_tod, groups, sentinel)
        t2_ok = bool(np.array_equal(t2_back, arr))
        t2_bytes = int(t2.nbytes)
        # T3: pack the K axis, which is the one holding the offsets
        t2_k_last = np.ascontiguousarray(t2.transpose(0, 2, 1))
        t3 = bitpack(t2_k_last, bits)
        t3_back = bitunpack(t3, k, bits)
        t3_ok = bool(np.array_equal(t3_back, t2_k_last))
        t3_bytes = int(t3.nbytes)
        # end-to-end: T3 bytes -> int32 endpoints, must equal the original
        t3_e2e = decode_t2(
            np.ascontiguousarray(t3_back.transpose(0, 2, 1)),
            query_tod, groups, sentinel)
        t3_e2e_ok = bool(np.array_equal(t3_e2e, arr))

        if not (t1_ok and t2_ok and t3_ok and t3_e2e_ok):
            raise AssertionError(
                f"{split}: lossless gate failed "
                f"(t1={t1_ok}, t2={t2_ok}, t3={t3_ok}, e2e={t3_e2e_ok})")

        report["splits"][split] = {
            "shape": [int(n_queries), k, int(arr.shape[2])],
            "queries": int(n_queries),
            "slots": int(total_slots),
            "unavailable_slots": unavailable,
            "unavailable_fraction": unavailable / total_slots,
            "endpoint_min": int(arr.min()),
            "endpoint_max": int(arr.max()),
            "distinct_endpoints": int(np.unique(arr[arr >= 0]).size),
            "t0_bytes": t0_bytes,
            "t1_bytes": t1_bytes,
            "t2_bytes": t2_bytes,
            "t3_bytes": t3_bytes,
            "t1_lossless": t1_ok,
            "t2_lossless": t2_ok,
            "t3_lossless": t3_ok,
            "t3_end_to_end_lossless": t3_e2e_ok,
        }
        totals["t0"] += t0_bytes
        totals["t1"] += t1_bytes
        totals["t2"] += t2_bytes
        totals["t3"] += t3_bytes

    report["totals_bytes"] = totals
    report["reduction_vs_t0"] = {
        key: 1.0 - totals[key] / totals["t0"] for key in ("t1", "t2", "t3")
    }
    report["gate"]["all_lossless"] = all(
        v["t1_lossless"] and v["t2_lossless"] and v["t3_lossless"]
        for v in report["splits"].values()
    )
    return report


def main():
    reports = {}
    for name in DATASETS:
        print(f"=== {name} ===", flush=True)
        rep = run_dataset(name)
        reports[name] = rep
        print(
            "  groups: n=%d min=%d max=%d mean=%.1f bits=%d | K=%d"
            % (rep["num_tod_groups"], rep["group_size_min"],
               rep["group_size_max"], rep["group_size_mean"],
               rep["bits_per_offset"], rep["k_meta"]), flush=True)
        for split, s in rep["splits"].items():
            print(
                "  %-5s shape=%-20s unavail=%6.2f%%  t0=%8.2f MB "
                "t1=%8.2f t2=%8.2f t3=%8.2f  lossless=%s/%s/%s"
                % (split, str(tuple(s["shape"])), 100 * s["unavailable_fraction"],
                   s["t0_bytes"] / 1e6, s["t1_bytes"] / 1e6,
                   s["t2_bytes"] / 1e6, s["t3_bytes"] / 1e6,
                   s["t1_lossless"], s["t2_lossless"], s["t3_lossless"]),
                flush=True)
        t = rep["totals_bytes"]
        r = rep["reduction_vs_t0"]
        print("  TOTAL t0=%.2f MB -> t1=%.2f (-%.2f%%) t2=%.2f (-%.2f%%) "
              "t3=%.2f (-%.2f%%)"
              % (t["t0"] / 1e6, t["t1"] / 1e6, 100 * r["t1"],
                 t["t2"] / 1e6, 100 * r["t2"], t["t3"] / 1e6, 100 * r["t3"]),
              flush=True)

    if not all(r["gate"]["all_lossless"] for r in reports.values()):
        raise SystemExit("GATE FAILED - not reporting")

    out = os.path.join(ROOT, "contrib3_index_codec_20261020",
                       "index_codec_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "scope": "four datasets, all splits, lossless round-trip gate",
            "codecs": {
                "T0": "int32 global endpoints (-1 = unavailable)",
                "T1": "int16 order-preserving narrowing",
                "T2": "uint8 TOD-group offset (255 = unavailable), "
                      "group rebuilt from idx_train + time_data at zero cost",
                "T3": "T2 offsets bit-packed at ceil(log2(G_max+1)) bits",
            },
            "datasets": reports,
        }, f, indent=2)
    print(f"\nGATE PASSED (bit-exact on every split/codec). Wrote {out}")


if __name__ == "__main__":
    main()
