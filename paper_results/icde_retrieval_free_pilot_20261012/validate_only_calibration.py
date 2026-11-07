"""Audit a retrieval-free checkpoint with the frozen PEMS04 ValCal protocol.

Only Validation predictions are forwarded and scored. Test is not evaluated.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    return parser.parse_args()


CLI_ARGS = parse_args()
# run.py parses sys.argv at import time; hide this script's CLI flags.
sys.argv = [sys.argv[0]]

from run import trainer, unpack_dataloader_input  # noqa: E402
from utils import datasets_utils  # noqa: E402
from calibrate_dualscale_smooth_tod import fit_and_apply  # noqa: E402
from calibrate_dualscale_quantile_residual import metrics  # noqa: E402


def get_logger() -> logging.Logger:
    log = logging.getLogger("retrieval_free_validation_calibration")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler(sys.stdout))
    return log


@torch.no_grad()
def collect_validation(model, loader, scaler, device, sample_count):
    predictions, truths = [], []
    original_drop_last = loader.drop_last
    original_num_batch = loader.num_batch
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    try:
        model.eval()
        for loader_input, loader_target in loader.get_iterator():
            (
                x,
                x_time,
                analog_retrieval,
                exact_retrieval,
                exact_available,
                candidate_future,
                candidate_available,
                candidate_features,
            ) = unpack_dataloader_input(loader_input)
            y, y_time = loader_target
            model_input = [
                torch.as_tensor(x, dtype=torch.float32, device=device),
                torch.as_tensor(x_time, dtype=torch.long, device=device),
                torch.as_tensor(y_time, dtype=torch.long, device=device),
            ]
            for value in (
                analog_retrieval,
                exact_retrieval,
                exact_available,
                candidate_future,
                candidate_available,
                candidate_features,
            ):
                if value is not None:
                    model_input.append(torch.as_tensor(value, device=device))
            prediction, _ = model(model_input)
            predictions.append(prediction.cpu())
            truths.append(torch.as_tensor(y, dtype=torch.float32))
    finally:
        loader.drop_last = original_drop_last
        loader.num_batch = original_num_batch

    prediction = torch.cat(predictions, dim=0)[:sample_count]
    truth = torch.cat(truths, dim=0)[:sample_count]
    prediction = scaler.inverse_transform(prediction).numpy()[..., 0]
    truth = truth.numpy()[..., 0]
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Validation prediction/truth shape or finiteness check failed")
    if not np.isfinite(truth).all():
        raise ValueError("Validation truth has non-finite values")
    return prediction.astype(np.float32), truth.astype(np.float32)


def main() -> None:
    exp_dir = CLI_ARGS.exp_dir.resolve()
    output_dir = CLI_ARGS.output_dir.resolve()
    reference_path = CLI_ARGS.reference.resolve()
    report_path = output_dir / "validation_calibration.json"
    predictions_path = output_dir / "validation_predictions.npz"
    if report_path.exists() or predictions_path.exists():
        raise FileExistsError("Refusing to overwrite existing validation results")
    with (exp_dir / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    needs_graph = bool(getattr(args, "use_graph_conv", 0)) or (
        getattr(args, "spatial_type", None) == "agcrn"
    )
    pearson_path = getattr(args, "pearson_adj_path", None)
    default_pearson_path = Path("datasets") / args.dataset_name / (
        f"{args.dataset_name}_pearson_adj.npy"
    )
    if needs_graph and pearson_path and Path(pearson_path).is_file():
        args.pearson_adj = torch.as_tensor(np.load(pearson_path), dtype=torch.float32)
    elif needs_graph and default_pearson_path.is_file():
        args.pearson_adj = torch.as_tensor(
            np.load(default_pearson_path), dtype=torch.float32
        )
    else:
        args.pearson_adj = None

    index_path = Path("datasets") / args.dataset_name / (
        f"{args.dataset_name}_{args.input_length}to{args.predict_length}_index_scaler.npz"
    )
    with np.load(index_path, allow_pickle=False) as archive:
        endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
    if not np.all(np.diff(endpoints) > 0):
        raise ValueError("Validation endpoints are not strictly increasing")

    log = get_logger()
    device = torch.device(args.device)
    loaders, num_nodes, scaler, _ = datasets_utils.load_dataset(args, log)
    args.num_nodes = num_nodes
    engine = trainer(scaler=scaler, device=device, args=args, inverse=True)
    state = torch.load(exp_dir / "best.pth", map_location=device)
    engine.model.load_state_dict(state, strict=True)
    log.info("Forward Validation once from frozen checkpoint ...")
    prediction, truth = collect_validation(
        engine.model, loaders["val_loader"], scaler, device, len(endpoints)
    )
    if len(prediction) != len(endpoints):
        raise ValueError("Validation predictions and endpoints are not aligned")
    with reference_path.open("r", encoding="utf-8") as stream:
        reference = json.load(stream)
    parameters = {
        "num_bins": reference["num_bins"],
        "smooth_shrinkage": reference["smooth_shrinkage"],
        "num_time_bins": reference["num_time_bins"],
        "time_shrinkage": reference["time_shrinkage"],
        "slices_per_day": reference["slices_per_day"],
    }
    reference_folds = reference["validation_folds"]
    folds = []
    for train_fraction, end_fraction in ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0)):
        train_end = int(len(prediction) * train_fraction)
        eval_end = int(len(prediction) * end_fraction)
        smooth, final, _ = fit_and_apply(
            prediction[:train_end].astype(np.float64),
            truth[:train_end].astype(np.float64),
            endpoints[:train_end],
            prediction[train_end:eval_end].astype(np.float64),
            endpoints[train_end:eval_end],
            **parameters,
        )
        baseline_metrics = metrics(
            prediction[train_end:eval_end], truth[train_end:eval_end]
        )
        smooth_metrics = metrics(smooth, truth[train_end:eval_end])
        final_metrics = metrics(final, truth[train_end:eval_end])
        folds.append({
            "train_samples": train_end,
            "evaluation_start": train_end,
            "evaluation_end": eval_end,
            "baseline": baseline_metrics,
            "smooth_level": smooth_metrics,
            "smooth_plus_target_time": final_metrics,
            "smooth_gate_passed": smooth_metrics["mae"] < baseline_metrics["mae"],
            "target_time_gate_passed": final_metrics["mae"] < smooth_metrics["mae"],
        })
        log.info(
            "Validation fold %d: raw %.4f, smooth %.4f, final %.4f",
            len(folds), baseline_metrics["mae"], smooth_metrics["mae"],
            final_metrics["mae"],
        )
    for new_fold, old_fold in zip(folds, reference_folds, strict=True):
        if (new_fold["train_samples"], new_fold["evaluation_start"], new_fold["evaluation_end"]) != (
            old_fold["train_samples"], old_fold["evaluation_start"], old_fold["evaluation_end"]
        ):
            raise ValueError("Reference and pilot Validation folds are not aligned")
    new_oos = float(np.mean([f["smooth_plus_target_time"]["mae"] for f in folds]))
    reference_oos = float(np.mean([
        f["smooth_plus_target_time"]["mae"] for f in reference_folds
    ]))
    report = {
        "protocol": "PEMS04_frozen_ValCal_expanding_folds_validation_only",
        "checkpoint": str(exp_dir / "best.pth"),
        "reference_report": str(reference_path),
        "test_evaluated": False,
        "parameters": parameters,
        "baseline_validation": metrics(prediction, truth),
        "validation_folds": folds,
        "all_validation_gates_passed": all(
            f["smooth_gate_passed"] and f["target_time_gate_passed"] for f in folds
        ),
        "pilot_calibrated_oos_fold_mean_mae": new_oos,
        "reference_calibrated_oos_fold_mean_mae": reference_oos,
        "pilot_minus_reference_calibrated_oos_mae": new_oos - reference_oos,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        val_prediction=prediction,
        val_truth=truth,
        val_endpoints=endpoints,
    )
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    log.info("Saved Validation-only report: %s", report_path)
    log.info("Pilot minus reference calibrated OOS MAE: %+0.4f", new_oos - reference_oos)


if __name__ == "__main__":
    main()
