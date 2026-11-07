"""Read-only audit of saved offline retrieval versus physical Train-state retrieval."""

import argparse
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from contrib3_index_codec_20261020.adaptive_lossless_state_codec import decode
from utils.online_residual_memory import OnlineResidualMemory

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=("val", "test"), default="val")
args = parser.parse_args()
state_name = "selected_val_cold_state.alsc" if args.split == "val" else "selected_train_state.alsc"
state = decode((here / state_name).read_bytes())
memory = OnlineResidualMemory(state)
with np.load(root / "datasets/PEMSBAY/PEMSBAY_12to12_index_scaler_712.npz") as file:
    endpoints = file[f"idx_{args.split}"]
with np.load(root / "datasets/PEMSBAY/data.npz") as file:
    raw = file["data"][..., 0]
with np.load(root / "datasets/PEMSBAY/PEMSBAY_time.npz") as file:
    tod = file["time"][..., 0]
with np.load(root / "datasets/PEMSBAY/PEMSBAY_12to12_analog_k8_exact_mean_712.npz") as file:
    positions = (np.arange(len(endpoints)) if args.split == "val" else
                 np.linspace(0, len(endpoints) - 1, 512, dtype=np.int64))
    cached = file[f"retrieval_{args.split}"][positions][:, :, memory.node_ids, 0]
    cached_exact = file[f"exact_retrieval_{args.split}"][positions][:, :, memory.node_ids, 0]
    cached_available = file[f"exact_available_{args.split}"][positions][:, :, memory.node_ids, 0]
endpoints = endpoints[positions]
offsets = np.arange(-11, 1)
history = raw[endpoints[:, None] + offsets[None, :]][:, :, memory.node_ids].astype(np.float32)
retrieved, exact, available, _ = memory.query_stream(history, endpoints, tod[endpoints].astype(np.int16))
for name, actual, reference in (
    ("retrieval", retrieved[..., 0], cached),
    ("exact", exact[..., 0], cached_exact),
    ("available", available[..., 0], cached_available),
):
    delta = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
    point = np.unravel_index(np.argmax(delta), delta.shape)
    print(name, "max", delta.max(), "mean", delta.mean(), "nonzero", np.count_nonzero(delta),
          "location", (point[0], point[1] + 1, int(memory.node_ids[point[2]])),
          "values", actual[point], reference[point])
    print("sample_count_gt_1e-5", np.sum(np.max(delta, axis=(1, 2)) > 1e-5))
