"""Post-Validation confirmation of PEMS07 seed 2 with frozen final codec.

Seed selection is fixed in pems07_single_seed_selection_val_20261024.json.
This Test is not a fresh blind holdout because earlier project runs viewed Test.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ONLINE = HERE / "packages/online"
sys.path[:0] = [str(HERE), str(ROOT)]
from adaptive_lossless_state_codec import load_runtime_state
from select_single_seed_validation import load_runner

SELECTION = ONLINE / "pems07_single_seed_selection_val_20261024.json"
RESULT = ONLINE / "pems07_selected_seed2_codec_test_20261024.json"
STARTED = ONLINE / "pems07_selected_seed2_codec_test_20261024.started"


def main():
    if RESULT.exists() or STARTED.exists():
        raise FileExistsError("Selected Test already started or completed; refusing to repeat")
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selected["selected_seed"] != 2 or selected["test_used"]:
        raise ValueError("Unexpected seed-selection record")
    runner = load_runner(2)
    experiment, calibration_path, _, _, selector = runner.inputs()
    with (experiment / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(v) for v in args.k_hops.split(",")]
    args.pearson_adj = None
    if not (args.use_analog_lowrank_retrieval_residual and
            args.use_analog_exact_match_residual):
        raise ValueError("Expected frozen retrieval-aware Student")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("selected_seed2_test")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = runner.datasets_utils.load_dataset(args, logger)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(
        experiment / "best.pth", map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False
    with np.load(ROOT / "datasets/PEMS07/PEMS07_12to12_index_scaler.npz",
                 allow_pickle=False) as index:
        val_endpoints = np.asarray(index["idx_val"], dtype=np.int64)
        test_endpoints = np.asarray(index["idx_test"], dtype=np.int64)
    val_path = ONLINE / "adaptive_lossless_val_20261023_v2/pems07_val_selected.alsc"
    test_path = ONLINE / "adaptive_lossless_test_20261023_v1/pems07_test_selected.alsc"
    val_memory = runner.OnlineResidualMemory(load_runtime_state(val_path))
    test_memory = runner.OnlineResidualMemory(load_runtime_state(test_path))
    if val_memory.cold_start_endpoint != int(val_endpoints[0]):
        raise ValueError("Validation state misaligned")
    if test_memory.cold_start_endpoint is not None:
        raise ValueError("Test state must be fully observed")
    val_pred, val_truth = runner.collect(
        loaders["val_loader"], val_endpoints, model, scaler, val_memory, device, 32)
    known_val_mae = next(row["validation_raw"]["mae"] for row in selected["rows"]
                         if row["seed"] == 2)
    measured_val_mae = runner.metrics(val_pred, val_truth)["mae"]
    if abs(known_val_mae - measured_val_mae) > 1e-8:
        raise AssertionError("Validation selection cannot be reproduced")
    parameters = json.loads(calibration_path.read_text(encoding="utf-8"))
    kwargs = {key: parameters[key] for key in (
        "num_bins", "smooth_shrinkage", "num_time_bins",
        "time_shrinkage", "slices_per_day")}
    STARTED.write_text(json.dumps(dict(selected_seed=2, validation_mae=measured_val_mae)),
                       encoding="utf-8")
    test_pred, test_truth = runner.collect(
        loaders["test_loader"], test_endpoints, model, scaler, test_memory, device, 32)
    _, calibrated, _ = runner.fit_and_apply(
        val_pred, val_truth, val_endpoints, test_pred, test_endpoints, **kwargs)
    report = dict(dataset="PEMS07", seed=2, selection_split="Validation",
                  previous_test_seen_in_project=True, new_blind_test=False,
                  selector=selector, checkpoint=str(experiment / "best.pth"),
                  validation_raw=runner.metrics(val_pred, val_truth),
                  test_raw=runner.metrics(test_pred, test_truth),
                  test_val_calibrated=runner.metrics(calibrated, test_truth),
                  validation_count=len(val_truth), test_count=len(test_truth),
                  val_state=str(val_path), test_state=str(test_path))
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE PEMS07 seed 2", report["test_val_calibrated"]["mae"], flush=True)


if __name__ == "__main__":
    main()
