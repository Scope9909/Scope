"""Measure batch-1 PEMS07 gate feature + 5-model decision CPU cost.

Uses Validation input and Student forecasts only. Random weights have identical
architecture/operations to the five trained gate models; no accuracy is read.
This is gate-only cost, not Student/Teacher end-to-end serving latency.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from run_pems07_teacher_escalation import UtilityGate, build_features


def main() -> None:
    torch.set_num_threads(1)
    with np.load(ROOT / "paper_results/pems07_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only/selected_predictions.npz", allow_pickle=False) as z:
        endpoints = np.asarray(z["val_endpoints"], dtype=np.int64)
        predictions = np.asarray(z["val_prediction"], dtype=np.float32)
    with np.load(ROOT / "datasets/PEMS07/data.npz", allow_pickle=False) as z:
        raw = np.asarray(z["data"], dtype=np.float32)[..., 0]
    starts = np.linspace(0, len(endpoints)-1, 200, dtype=np.int64)
    features, _ = build_features(raw, endpoints[starts[:1]], predictions[starts[:1]])
    gates = [UtilityGate(features.shape[1]).eval() for _ in range(5)]
    times_ms = []
    with torch.no_grad():
        for position in starts[:20]:
            x, _ = build_features(raw, endpoints[position:position+1], predictions[position:position+1])
            tensor = torch.from_numpy(x)
            _ = np.mean([model(tensor).item() for model in gates])
        for position in starts:
            start = time.perf_counter()
            x, _ = build_features(raw, endpoints[position:position+1], predictions[position:position+1])
            tensor = torch.from_numpy(x)
            _ = np.mean([model(tensor).item() for model in gates])
            times_ms.append((time.perf_counter()-start)*1000)
    report = {
        "protocol": "PEMS07_Validation_batch1_CPU_gate_only_feature_plus_5_MLP_ensemble",
        "device": "CPU",
        "samples": len(times_ms),
        "warmup": 20,
        "torch_threads": 1,
        "input_dimension": features.shape[1],
        "weights": "random_same_architecture_timing_only",
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "mean_ms": float(np.mean(times_ms)),
        "caveat": "Gate decision only; excludes Student, Teacher, retrieval, data loader, host-device transfer and deployment overhead.",
    }
    out = HERE / "pems07_gate_decision_cpu_latency.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
