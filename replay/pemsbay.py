"""Evaluate the frozen PEMS-BAY Student with physical sparse float32 history.

Validation is run first and saved separately. Test is a later one-shot run
using the already frozen node set, lossless state and calibration parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Match the frozen Validation reference on CUDA across runs. With cuDNN TF32
# enabled, the selected-node parity gate can exceed its 0.005 tolerance.
torch.backends.cudnn.allow_tf32 = False

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="New directory for Validation predictions and Test report")
    return parser.parse_args()


CLI = arguments()
sys.argv = [sys.argv[0]]
OUT = CLI.output_dir.resolve()
from replay.calibration import fit_and_apply, metrics  # noqa: E402
from replay.state_codec import decode  # noqa: E402
from replay.model_runtime import build_model, unpack_dataloader_input  # noqa: E402
from utils import datasets_utils  # noqa: E402
from utils.online_residual_memory import OnlineResidualMemory  # noqa: E402

EXPERIMENT = ROOT / "assets/checkpoints/PEMSBAY"
INDEX = ROOT / "datasets/PEMSBAY/PEMSBAY_12to12_index_scaler_712.npz"
RAW = ROOT / "datasets/PEMSBAY/data.npz"
TIME = ROOT / "datasets/PEMSBAY/PEMSBAY_time.npz"
CALIBRATION = ROOT / "configs/frozen_calibration.json"
SCREEN = ROOT / "assets/states/PEMSBAY/validation_screen.json"
MANIFEST = ROOT / "assets/states/PEMSBAY/freeze_manifest.json"
VAL_OUTPUT = OUT / "validation_predictions.npz"
TEST_OUTPUT = OUT / "test_results.json"


def digest(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hash_obj.update(block)
    return hash_obj.hexdigest()


def sparse_correction(model, prediction, ids, retrieval, exact, available):
    low = model.analog_lowrank_retrieval_residual
    exact_adapter = model.analog_exact_match_residual
    selected = prediction.index_select(2, ids)
    # Preserve the original full-node factor contraction before selecting
    # sparse nodes; changing the GEMM shape changes CUDA accumulation slightly.
    full_logits = low.horizon_logits[:, None] + torch.einsum(
        "nr,hr->hn", low.node_factor, low.horizon_factor)
    blend = (low.max_blend * torch.tanh(full_logits)).index_select(1, ids)
    blend = blend.view(1, blend.shape[0], len(ids), 1)
    selected = selected + blend * (retrieval - selected)
    exact_blend = (exact_adapter.max_blend *
                   torch.tanh(exact_adapter.horizon_logits)).view(1, -1, 1, 1)
    selected = selected + exact_blend * available.to(selected.dtype) * (exact - selected)
    return prediction.index_copy(2, ids, selected)


def lead_metrics(prediction, truth, lead):
    return metrics(prediction[:, lead - 1:lead], truth[:, lead - 1:lead])


def main():
    if CLI.split == "val":
        OUT.mkdir(parents=True, exist_ok=True)
    elif not OUT.is_dir():
        raise FileNotFoundError(f"Validation output directory is missing: {OUT}")
    if CLI.split == "val" and VAL_OUTPUT.exists():
        raise FileExistsError(VAL_OUTPUT)
    if CLI.split == "test" and TEST_OUTPUT.exists():
        raise FileExistsError(TEST_OUTPUT)
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if frozen["test_evaluated"] or frozen["checkpoint_sha256"] != digest(EXPERIMENT / "best.pth"):
        raise ValueError("Frozen checkpoint mismatch")
    if frozen["validation_screen_sha256"] != digest(SCREEN):
        raise ValueError("Frozen node selection mismatch")
    if screen["test_evaluated"]:
        raise ValueError("Screen unexpectedly used Test")
    with np.load(INDEX, allow_pickle=False) as archive:
        endpoints = np.asarray(archive[f"idx_{CLI.split}"], dtype=np.int64)
    state_file = ROOT / "assets/states/PEMSBAY" / (
        "validation.alsc" if CLI.split == "val" else "test.alsc"
    )
    state = decode(state_file.read_bytes())
    memory = OnlineResidualMemory(state)
    if not np.array_equal(memory.node_ids, frozen["selected_nodes"]):
        raise ValueError("Encoded node set mismatch")
    if CLI.split == "test" and not VAL_OUTPUT.exists():
        raise FileNotFoundError("Validation prediction must be frozen before Test")

    with (EXPERIMENT / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    if not (args.use_analog_lowrank_retrieval_residual and args.use_analog_exact_match_residual):
        raise ValueError("Expected retrieval-enabled Student checkpoint")
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.pearson_adj = None
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    log = logging.getLogger("pemsbay_frozen")
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(logging.INFO)
    loaders, nodes, scaler, _ = datasets_utils.load_dataset(args, log)
    if nodes != 325:
        raise ValueError("Unexpected PEMS-BAY node count")
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    model = build_model(args, device)
    model.load_state_dict(torch.load(EXPERIMENT / "best.pth", map_location=device,
                                     weights_only=True), strict=True)
    model.eval()
    ids = torch.as_tensor(memory.node_ids, dtype=torch.long, device=device)
    node_mask = torch.zeros((1, 1, nodes, 1), dtype=torch.float32, device=device)
    node_mask[:, :, ids, :] = 1.0
    adapter = model.analog_lowrank_retrieval_residual
    original_forward = adapter.forward

    def masked_forward(base, retrieved):
        correction = original_forward(base, retrieved) - base
        return base + node_mask * correction

    adapter.forward = masked_forward
    with np.load(RAW, allow_pickle=False) as archive:
        raw = np.asarray(archive["data"])[..., 0]
    with np.load(TIME, allow_pickle=False) as archive:
        tod = np.asarray(archive["time"])[..., 0]
    loader = loaders[f"{CLI.split}_loader"]
    previous = (loader.drop_last, loader.num_batch, loader.bs)
    loader.drop_last = False
    loader.bs = 32
    loader.num_batch = int(np.ceil(loader.size / loader.bs))
    predictions, truths = [], []
    cursor = 0
    offsets = np.arange(1 - args.input_length, 1, dtype=np.int64)
    try:
        with torch.inference_mode():
            for loader_input, loader_target in loader.get_iterator():
                if cursor >= len(endpoints):
                    break
                x, x_time, *extras = unpack_dataloader_input(loader_input)
                if any(value is not None for value in extras):
                    raise ValueError("Query cache unexpectedly loaded")
                y, y_time = loader_target
                take = min(len(x), len(endpoints) - cursor)
                batch_endpoints = endpoints[cursor:cursor + take]
                history = raw[batch_endpoints[:, None] + offsets[None, :]][:, :, memory.node_ids]
                history = np.asarray(history, dtype=np.float32)
                query_tod = np.asarray(tod[batch_endpoints], dtype=np.int16)
                retrieval, exact, available, _ = memory.query_stream(
                    history, batch_endpoints, query_tod)
                model_input = [
                    torch.as_tensor(x[:take], dtype=torch.float32, device=device),
                    torch.as_tensor(x_time[:take], dtype=torch.long, device=device),
                    torch.as_tensor(y_time[:take], dtype=torch.long, device=device),
                ]
                dense_retrieval = torch.zeros((take, args.predict_length, nodes, 1),
                                              dtype=torch.float32, device=device)
                dense_exact = torch.zeros_like(dense_retrieval)
                dense_available = torch.zeros_like(dense_retrieval, dtype=torch.uint8)
                dense_retrieval[:, :, ids, :] = torch.as_tensor(
                    retrieval, dtype=torch.float32, device=device)
                dense_exact[:, :, ids, :] = torch.as_tensor(
                    exact, dtype=torch.float32, device=device)
                dense_available[:, :, ids, :] = torch.as_tensor(
                    available, device=device)
                model_input.extend((dense_retrieval, dense_exact, dense_available))
                prediction, _ = model(model_input)
                predictions.append(scaler.inverse_transform(prediction).cpu().numpy()[..., 0])
                truths.append(np.asarray(y[:take])[..., 0])
                cursor += take
                if cursor % 1024 < take:
                    print(f"{CLI.split}: {cursor}/{len(endpoints)}", flush=True)
    finally:
        loader.drop_last, loader.num_batch, loader.bs = previous
    if cursor != len(endpoints):
        raise ValueError("Loader did not cover all official samples")
    prediction = np.concatenate(predictions).astype(np.float32)
    truth = np.concatenate(truths).astype(np.float32)
    if prediction.shape != truth.shape or prediction.shape[0] != len(endpoints):
        raise ValueError("Output shape mismatch")
    if CLI.split == "val":
        np.savez_compressed(VAL_OUTPUT, val_prediction=prediction, val_truth=truth,
                            val_endpoints=endpoints)
        validation_report = {
            "dataset": "PEMSBAY", "split": "val",
            "sample_count": len(endpoints),
            "raw_metrics": metrics(prediction, truth),
            "state_sha256": digest(state_file),
        }
        (OUT / "validation_metrics.json").write_text(
            json.dumps(validation_report, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(json.dumps(validation_report, ensure_ascii=False), flush=True)
        return
    with np.load(VAL_OUTPUT, allow_pickle=False) as archive:
        val_pred = np.asarray(archive["val_prediction"], dtype=np.float64)
        val_truth = np.asarray(archive["val_truth"], dtype=np.float64)
        val_endpoints = np.asarray(archive["val_endpoints"], dtype=np.int64)
    config = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    params = {key: config[key] for key in (
        "num_bins", "smooth_shrinkage", "num_time_bins",
        "time_shrinkage", "slices_per_day")}
    _, calibrated, _ = fit_and_apply(val_pred, val_truth, val_endpoints,
                                     prediction.astype(np.float64), endpoints, **params)
    report = {
        "protocol": "pemsbay_sparse_float32_exact_test_once_v1",
        "dataset": "PEMSBAY", "selected_nodes": len(memory.node_ids),
        "test_samples": len(endpoints), "test_evaluations": 1,
        "test_based_selection": False, "checkpoint_sha256": frozen["checkpoint_sha256"],
        "state_sha256": digest(state_file), "validation_screen_sha256": digest(SCREEN),
        "test_raw": metrics(prediction, truth),
        "test_calibrated": metrics(calibrated, truth),
        "horizons": {str(h): {"raw": lead_metrics(prediction, truth, h),
                              "calibrated": lead_metrics(calibrated, truth, h)}
                     for h in (3, 6, 12)},
    }
    TEST_OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(TEST_OUTPUT), "test_calibrated": report["test_calibrated"],
                      "horizons": report["horizons"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
