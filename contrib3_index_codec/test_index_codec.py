"""Independent round-trip test of the shipped codec on all splits/datasets.

The audit script proved the *idea* is lossless with inline code.  This test
exercises the actual module a loader would import, so the paper can claim
"the shipped codec round-trips bit-exactly", not just "an experiment did".
"""

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location(
    "index_codec", os.path.join(HERE, "index_codec.py"))
index_codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index_codec)

DATASETS = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
SPLITS = ("train", "val", "test")


def run(name):
    base = os.path.join(ROOT, "datasets", name)
    with np.load(os.path.join(base, f"{name}_12to12_index_scaler.npz"),
                 allow_pickle=False) as z:
        idx = {s: np.asarray(z[f"idx_{s}"], dtype=np.int64) for s in SPLITS}
    with np.load(os.path.join(base, f"{name}_time.npz"),
                 allow_pickle=False) as z:
        time_data = np.asarray(z["time"])

    codec = index_codec.TodIndexCodec(time_data, idx["train"])
    rows = {}
    for split in SPLITS:
        with np.load(
                os.path.join(
                    base, f"{name}_12to12_analog_k8_candidate_indices.npz"),
                allow_pickle=False) as z:
            original = np.asarray(z[f"candidate_index_{split}"], dtype=np.int32)
        q = idx[split]
        k = original.shape[1]

        code = codec.encode(original)
        decoded8 = codec.decode(code, q)
        ok8 = bool(np.array_equal(decoded8, original))

        back, packed = codec.pack_transposed_roundtrip(code, k)
        decoded6 = codec.decode(back, q)
        ok6 = bool(np.array_equal(decoded6, original))

        if not (ok8 and ok6):
            raise AssertionError(f"{name}/{split}: round-trip failed")

        rows[split] = {
            "shape": [int(v) for v in original.shape],
            "original_bytes": int(original.nbytes),
            "offset8_bytes": int(code.nbytes),
            "offset6_bytes": int(packed.nbytes),
            "offset8_lossless": ok8,
            "offset6_lossless": ok6,
        }
    return {
        "dataset": name,
        "bits": codec.bits,
        "group_size_max": codec.group_size_max,
        "splits": rows,
    }


def main():
    results = [run(d) for d in DATASETS]
    for r in results:
        for split, s in r["splits"].items():
            print("%-7s %-5s shape=%-20s %8.2f -> %6.2f / %6.2f MB  "
                  "lossless=%s/%s"
                  % (r["dataset"], split, str(tuple(s["shape"])),
                     s["original_bytes"] / 1e6, s["offset8_bytes"] / 1e6,
                     s["offset6_bytes"] / 1e6,
                     s["offset8_lossless"], s["offset6_lossless"]))
    out = os.path.join(HERE, "index_codec_roundtrip.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"module": "index_codec.TodIndexCodec", "results": results},
                  f, indent=2)
    print("\nALL PASS - shipped codec is bit-exact on 4 datasets x 3 splits")
    print("wrote", out)


if __name__ == "__main__":
    main()
