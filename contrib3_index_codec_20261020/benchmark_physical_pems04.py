"""Materialize and benchmark a PEMS04 6-bit candidate-index artifact.

This is isolated from the training loader and never overwrites source caches.
"""

from pathlib import Path
from time import perf_counter
import importlib.util

import numpy as np

CODEC_PATH = Path(__file__).resolve().parent / "index_codec.py"
spec = importlib.util.spec_from_file_location("index_codec", CODEC_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TodIndexCodec = module.TodIndexCodec


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets" / "PEMS04"
OUT = Path(__file__).resolve().parent / "PEMS04_candidate_index_6bit_trial.npz"
SPLITS = ("train", "val", "test")


def main():
    with np.load(BASE / "PEMS04_12to12_index_scaler.npz") as z:
        indices = {s: z[f"idx_{s}"].astype(np.int64) for s in SPLITS}
    with np.load(BASE / "PEMS04_time.npz") as z:
        time_data = z["time"]
    codec = TodIndexCodec(time_data, indices["train"])
    source = BASE / "PEMS04_12to12_analog_k8_candidate_indices.npz"
    arrays = {}
    original_bytes = packed_bytes = 0
    with np.load(source) as z:
        for split in SPLITS:
            original = z[f"candidate_index_{split}"]
            t0 = perf_counter()
            packed = codec.pack(codec.encode(original))
            encode_s = perf_counter() - t0
            t0 = perf_counter()
            decoded = codec.decode(
                codec.unpack(packed, original.shape[1]).transpose(0, 2, 1),
                indices[split],
            )
            decode_s = perf_counter() - t0
            if not np.array_equal(decoded, original):
                raise AssertionError(f"{split}: decoded index differs")
            arrays[f"packed_{split}"] = packed
            original_bytes += original.nbytes
            packed_bytes += packed.nbytes
            print(f"{split}: raw={original.nbytes/1e6:.2f} MB, "
                  f"packed={packed.nbytes/1e6:.2f} MB, "
                  f"encode={encode_s:.3f}s, decode={decode_s:.3f}s, "
                  "bit_exact=True", flush=True)
    np.savez(OUT, **arrays)
    with np.load(OUT) as z, np.load(source) as src:
        for split in SPLITS:
            original = src[f"candidate_index_{split}"]
            t0 = perf_counter()
            decoded = codec.decode(
                codec.unpack(z[f"packed_{split}"], original.shape[1])
                .transpose(0, 2, 1), indices[split]
            )
            load_decode_s = perf_counter() - t0
            if not np.array_equal(decoded, original):
                raise AssertionError(f"{split}: persisted artifact differs")
            print(f"{split}: persisted_load_decode={load_decode_s:.3f}s", flush=True)
    print(f"source_file={source.stat().st_size/1e6:.2f} MB, "
          f"trial_file={OUT.stat().st_size/1e6:.2f} MB", flush=True)
    print(f"raw_array={original_bytes/1e6:.2f} MB, "
          f"packed_array={packed_bytes/1e6:.2f} MB, "
          f"reduction={1-packed_bytes/original_bytes:.2%}", flush=True)
    print("Note: the current loader still expands candidates to int32; "
          "this benchmark does not claim resident-memory savings.", flush=True)


if __name__ == "__main__":
    main()
