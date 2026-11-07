"""Benchmark frozen Student + ALSC retrieval + loaded calibrator on Validation.

Uses the first fixed Validation queries; no Test state, predictions, or labels
are read. Each pass reloads the immutable state to reset cold-start streaming.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import eval_hybrid_codec_test_once as previous
from frozen_deployment_runtime import FrozenAssets

VAL = HERE / "packages/online/adaptive_lossless_val_20261023_v2"
CAL = HERE / "packages/online/frozen_calibrators_20261024_v3"


def prepare(dataset):
    runner = previous._load_runner(dataset)
    experiment, _, _, _, _ = runner.inputs()
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
        raise ValueError("Expected frozen Student retrieval adapters")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("frozen_deployment_benchmark")
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
    with np.load(ROOT / "datasets" / dataset /
                 f"{dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz",
                 allow_pickle=False) as archive:
        endpoints = np.asarray(archive["idx_val"], dtype=np.int64)
    return runner, loaders["val_loader"], model, scaler, device, endpoints


def run_pass(dataset, setup, batch_size, samples):
    runner, loader, model, scaler, device, all_endpoints = setup
    endpoints = all_endpoints[:samples]
    assets = FrozenAssets.load(
        VAL / f"{dataset.lower()}_val_selected.alsc",
        CAL / f"{dataset.lower()}_calibrator.npz")
    if assets.memory.cold_start_endpoint != int(endpoints[0]):
        raise ValueError("Cold-start Validation state is misaligned")
    old = loader.bs, loader.drop_last, loader.num_batch
    loader.bs = batch_size
    loader.drop_last = False
    loader.num_batch = int(np.ceil(loader.size / batch_size))
    ids = torch.as_tensor(assets.memory.node_ids, dtype=torch.long, device=device)
    times, outputs = [], []
    cursor = 0
    try:
        with torch.inference_mode():
            for loader_input, loader_target in loader.get_iterator():
                if cursor >= samples:
                    break
                x, x_time, *extra = runner.unpack_dataloader_input(loader_input)
                if any(v is not None for v in extra):
                    raise ValueError("Unexpected precomputed query cache")
                _, y_time = loader_target
                take = min(len(x), samples - cursor)
                x, x_time, y_time = x[:take], x_time[:take], y_time[:take]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                raw_history = np.rint(scaler.inverse_transform(x)[..., 0][
                    :, :, assets.memory.node_ids]).astype(np.float32)
                tod = np.asarray(x_time[:, -1, 0], dtype=np.int16)
                retrieval, exact, available, _ = assets.memory.query_stream(
                    raw_history, endpoints[cursor:cursor + take], tod)
                model_input = [
                    torch.as_tensor(x, dtype=torch.float32, device=device),
                    torch.as_tensor(x_time, dtype=torch.long, device=device),
                    torch.as_tensor(y_time, dtype=torch.long, device=device),
                ]
                prediction, _ = model(model_input)
                prediction = runner.sparse_correction(
                    model, prediction, ids,
                    torch.as_tensor(retrieval, device=device),
                    torch.as_tensor(exact, device=device).float(),
                    torch.as_tensor(available, device=device))
                raw = scaler.inverse_transform(prediction).cpu().numpy()[..., 0]
                calibrated = assets.calibrate(raw, endpoints[cursor:cursor + take])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append((time.perf_counter() - started) * 1000)
                outputs.append(calibrated)
                cursor += take
    finally:
        loader.bs, loader.drop_last, loader.num_batch = old
    if cursor != samples:
        raise ValueError(f"Only processed {cursor}/{samples} Validation queries")
    return times, np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("samples and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        setup = prepare(dataset)
        n = min(args.samples, len(setup[-1]))
        for batch_size in (1, 32):
            run_pass(dataset, setup, batch_size, n)  # warm-up
            trials = []
            reference = None
            for repeat in range(args.repeats):
                times, output = run_pass(dataset, setup, batch_size, n)
                if reference is None:
                    reference = output
                elif not np.allclose(reference, output, rtol=0, atol=1e-8):
                    raise AssertionError("Repeated output changed")
                trials.append(times)
            flat = sorted(value for times in trials for value in times)
            row = dict(dataset=dataset, batch_size=batch_size, samples=n,
                       device=str(setup[-2]), repeats=args.repeats,
                       calls_per_repeat=len(trials[0]),
                       median_call_ms=statistics.median(flat),
                       p95_call_ms=flat[min(len(flat)-1, int(np.ceil(0.95*len(flat)))-1)],
                       median_ms_per_sample=statistics.median(
                           sum(t)/n for t in trials),
                       per_repeat_total_ms=[sum(t) for t in trials],
                       outputs_repeat_identical=True)
            rows.append(row)
            (args.output / "report.json").write_text(json.dumps(
                dict(scope="Sequential warm Validation inference with loaded ALSC state and calibrator",
                     test_used=False, labels_used=False, loader_time_excluded=True,
                     model_setup_and_asset_load_excluded=True, results=rows), indent=2),
                encoding="utf-8")
            print(dataset, "batch", batch_size, "median", round(row["median_call_ms"], 3),
                  "p95", round(row["p95_call_ms"], 3), flush=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
