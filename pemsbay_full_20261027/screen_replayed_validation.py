"""Re-screen SRM budgets against checkpoint-replayed Validation outputs only."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from calibrate_dualscale_quantile_residual import metrics

BASE = ROOT / "paper_results/pemsbay712_srm_validation_pilot_20261027/no_memory_validation_predictions.npz"
FULL = HERE / "full_validation_replayed.npz"
OUT = HERE / "validation_screen_replayed.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    with np.load(BASE) as archive:
        base = archive["val_prediction"]
        truth = archive["val_truth"]
        endpoints = archive["val_endpoints"]
    with np.load(FULL) as archive:
        full = archive["val_prediction"]
        if not np.array_equal(truth, archive["val_truth"]) or not np.array_equal(endpoints, archive["val_endpoints"]):
            raise ValueError("Validation alignment mismatch")
    n, _, nodes = truth.shape
    prefix = int(0.55 * n)
    valid = np.rint(truth[:prefix]) > 0
    count = valid.sum((0, 1))
    gain = (np.abs(base[:prefix] - truth[:prefix]) * valid).sum((0, 1)) / count - (
        np.abs(full[:prefix] - truth[:prefix]) * valid).sum((0, 1)) / count
    order = np.argsort(-gain, kind="stable")
    report = {
        "dataset": "PEMSBAY", "protocol": "checkpoint_replayed_validation_screen_v2",
        "test_evaluated": False, "selection_prefix_end": prefix,
        "node_count": nodes, "positive_gain_nodes": int((gain > 0).sum()),
        "prefix_gain_per_node": gain.tolist(), "budgets": [],
    }
    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        selected = np.sort(order[:round(fraction * nodes)])
        prediction = base.copy()
        prediction[:, :, selected] = full[:, :, selected]
        result = {"fraction": fraction, "selected_nodes": selected.tolist(),
                  "prefix": metrics(prediction[:prefix], truth[:prefix]),
                  "suffix": metrics(prediction[prefix:], truth[prefix:])}
        report["budgets"].append(result)
        print(f"fraction={fraction:.1f} nodes={len(selected)} "
              f"suffix_MAE={result['suffix']['mae']:.6f}", flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    old = json.loads((ROOT / "paper_results/pemsbay712_srm_validation_pilot_20261027/validation_screen.json").read_text())
    old_ids = set(next(x for x in old["budgets"] if x["fraction"] == 0.8)["selected_nodes"])
    new_ids = set(next(x for x in report["budgets"] if x["fraction"] == 0.8)["selected_nodes"])
    print(f"80pct overlap={len(old_ids & new_ids)}/{len(new_ids)}", flush=True)


if __name__ == "__main__":
    main()
