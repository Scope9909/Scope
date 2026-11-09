"""Accept physically evaluated Validation after checkpoint-consistent replay."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "validation_predictions.npz"
if OUTPUT.exists():
    raise FileExistsError(OUTPUT)
ids = np.asarray(json.loads((HERE / "freeze_manifest.json").read_text())["selected_nodes"])
others = np.setdiff1d(np.arange(325), ids)
with np.load(HERE / "validation_parity_debug.npz") as archive:
    actual, truth, endpoints = (archive[key] for key in ("prediction", "truth", "endpoints"))
with np.load(HERE / "full_validation_replayed.npz") as archive:
    full = archive["val_prediction"]
    if not np.array_equal(truth, archive["val_truth"]) or not np.array_equal(endpoints, archive["val_endpoints"]):
        raise ValueError("Full Validation alignment mismatch")
with np.load(ROOT / "paper_results/pemsbay712_srm_validation_pilot/no_memory_validation_predictions.npz") as archive:
    base = archive["val_prediction"]
    if not np.array_equal(truth, archive["val_truth"]) or not np.array_equal(endpoints, archive["val_endpoints"]):
        raise ValueError("Base Validation alignment mismatch")
selected_error = np.abs(actual[:, :, ids] - full[:, :, ids])
unselected_error = np.abs(actual[:, :, others] - base[:, :, others])
print("selected parity", selected_error.max(), selected_error.mean())
print("unselected parity", unselected_error.max(), unselected_error.mean())
if selected_error.max() > 1e-5 or unselected_error.max() > 1e-5:
    raise ValueError("Physical sparse predictions are not reproducible")
np.savez_compressed(OUTPUT, val_prediction=actual, val_truth=truth,
                    val_endpoints=endpoints)
print("accepted", OUTPUT)
