"""PEMS07 dense Teacher base forecast under the frozen Student evaluation split.

This is a base-forecast control, not the full Hist2Res Teacher (whose K and
retrieval modules differ from the final Student). No Test-based fitting occurs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "contrib3_index_codec_20261020"))
import eval_hybrid_codec_test_once as hybrid  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = HERE / "pems07_teacher_base_metrics.json"
    if output.exists():
        raise FileExistsError(output)
    runner = hybrid._load_runner("PEMS07")
    teacher = (ROOT / "EXP_experiments" / "TLAST_improved" / "PEMS07" /
               "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_final_joint")
    _, calibration_path, _, _, _ = runner.inputs()
    with (teacher / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    args.pearson_adj = None
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    flags = {key: value for key, value in vars(args).items()
             if key.startswith("use_analog_")}
    for key in flags:
        setattr(args, key, 0)
    logger = logging.getLogger("pems07_teacher_base")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = runner.datasets_utils.load_dataset(args, logger)
    for key, value in flags.items():
        setattr(args, key, value)
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    checkpoint = teacher / "best.pth"
    engine.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    for key in flags:
        if hasattr(model, key):
            setattr(model, key, False)
    index_path = ROOT / "datasets" / "PEMS07" / "PEMS07_12to12_index_scaler.npz"
    with np.load(index_path, allow_pickle=False) as archive:
        val_idx = np.asarray(archive["idx_val"], dtype=np.int64)
        test_idx = np.asarray(archive["idx_test"], dtype=np.int64)
    print("PEMS07 dense Teacher base Validation", flush=True)
    val_pred, val_truth = runner.collect(loaders["val_loader"], val_idx, model, scaler, None, device, 32)
    print("PEMS07 dense Teacher base Test", flush=True)
    test_pred, test_truth = runner.collect(loaders["test_loader"], test_idx, model, scaler, None, device, 32)
    parameters = json.loads(calibration_path.read_text(encoding="utf-8"))
    kwargs = {key: parameters[key] for key in (
        "num_bins", "smooth_shrinkage", "num_time_bins", "time_shrinkage", "slices_per_day")}
    _, calibrated, _ = runner.fit_and_apply(
        val_pred, val_truth, val_idx, test_pred, test_idx, **kwargs)
    result = {
        "protocol": "PEMS07 dense Teacher base-only control; Validation-only calibration",
        "scope_warning": "Not the full retrieval-augmented Hist2Res Teacher",
        "dataset": "PEMS07", "seed": 1,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "validation_count": len(val_truth), "test_count": len(test_truth),
        "test_raw_metrics": runner.metrics(test_pred, test_truth),
        "test_calibrated_metrics": runner.metrics(calibrated, test_truth),
        "test_based_selection": False,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
