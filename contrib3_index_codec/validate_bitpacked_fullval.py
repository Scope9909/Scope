"""Full Validation prediction equivalence test for exact delta-bitpacked state."""
from __future__ import annotations

import importlib.util
import json
import logging
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ONLINE = HERE / "packages" / "online"
OUT = ONLINE / "bitpacked_fullval_formal_decoder.json"
DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")


def load_module(name: str, path: Path, argv: list[str]):
    previous = sys.argv
    try:
        sys.argv = argv
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = previous


base_codec = load_module("bitpack_base_codec", HERE / "pilot_reversible_predictive_coding.py", [str(HERE)])
bit_codec = load_module("bitpack_codec", HERE / "pilot_delta_bitpacking.py", [str(HERE)])
sys.path.insert(0, str(HERE))
from online_codec_runtime import decode_online_state  # noqa: E402


def load_eval(dataset: str, seed: int = 1):
    script = HERE / "eval_online_srm_codec.py"
    return load_module("online_eval_for_bitpack", script,
                       [str(script), dataset, "--seed", str(seed), "--validation-only"])


def make_candidate(source: Path, target: Path) -> tuple[int, dict[str, np.ndarray]]:
    state, raw = base_codec.normalize_state(source)
    stream, widths, prefix = bit_codec.encode(raw)
    state.pop("train_raw")
    state["train_raw_delta_prefix"] = prefix
    state["train_raw_delta_bits"] = stream
    state["train_raw_delta_widths"] = widths
    state["train_raw_delta_shape"] = np.asarray(raw.shape, dtype=np.uint32)
    state["train_raw_delta_codec_version"] = np.asarray(1, dtype=np.uint8)
    blob, _ = base_codec.best_bytes(state)
    target.write_bytes(blob)
    return len(blob), {"raw": raw, "compact_state": state}


def decode_candidate(path: Path) -> tuple[dict[str, np.ndarray], float]:
    start = time.perf_counter()
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    shape = tuple(int(v) for v in state.pop("train_raw_delta_shape"))
    version = int(state.pop("train_raw_delta_codec_version"))
    if version != 1:
        raise ValueError("Unsupported bitpacked codec version")
    raw = bit_codec.decode(state.pop("train_raw_delta_bits"),
                           state.pop("train_raw_delta_widths"),
                           state.pop("train_raw_delta_prefix"), shape[0])
    if raw.shape != shape:
        raise ValueError("Decoded history shape mismatch")
    state["train_raw"] = raw.astype(np.float32)
    if "idx_train_start" in state:
        index_start = int(np.asarray(state.pop("idx_train_start")).item())
        index_count = int(np.asarray(state.pop("idx_train_count")).item())
        state["idx_train"] = np.arange(index_start, index_start + index_count,
                                        dtype=np.int32)
    return state, time.perf_counter() - start


def run_dataset(dataset: str, seed: int = 1) -> dict:
    eval_module = load_eval(dataset, seed)
    runner = eval_module.runner
    exp_dir, _, val_path, _, _ = runner.inputs()
    with (exp_dir / "config.pkl").open("rb") as stream:
        args = pickle.load(stream)
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.batch_size = 32
    args.cache_data_path = "datasets"
    args.prob_factor = getattr(args, "prob_factor", 5)
    if isinstance(getattr(args, "k_hops", "1,2"), str):
        args.k_hops = [int(value) for value in args.k_hops.split(",")]
    args.pearson_adj = None
    if not (args.use_analog_lowrank_retrieval_residual and
            args.use_analog_exact_match_residual):
        raise ValueError("Expected Student low-rank plus exact-history checkpoint")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger(f"bitpacked_validate_{dataset}")
    logger.addHandler(logging.NullHandler())
    loaders, nodes, scaler, _ = runner.datasets_utils.load_dataset(args, logger)
    args.use_analog_lowrank_retrieval_residual = 1
    args.use_analog_exact_match_residual = 1
    args.num_nodes = nodes
    device = torch.device(args.device)
    engine = runner.trainer(scaler=scaler, device=device, args=args, inverse=True)
    engine.model.load_state_dict(torch.load(
        exp_dir / "best.pth", map_location=device, weights_only=True), strict=True)
    model = engine.model
    model.eval()
    model.use_analog_lowrank_retrieval_residual = False
    model.use_analog_exact_match_residual = False

    index_path = (HERE.parent / "datasets" / dataset /
                  f"{dataset}_{args.input_length}to{args.predict_length}_index_scaler.npz")
    with np.load(index_path, allow_pickle=False) as index:
        endpoints = np.asarray(index["idx_val"], dtype=np.int64)
    prefix = f"{dataset.lower()}_s{seed}"
    source = ONLINE / f"{prefix}_val_u16.npz"
    candidate_path = (ONLINE / "bitpacked_val" /
                      f"{dataset.lower()}_s{seed}_val_delta_bitpacked.npz")
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    start = time.perf_counter()
    decoded_state = decode_online_state(candidate_path)
    decode_seconds = time.perf_counter() - start
    with np.load(source, allow_pickle=False) as archive:
        baseline_state = {key: archive[key] for key in archive.files}
    baseline_state["train_raw"] = np.rint(
        baseline_state["train_raw"]).astype(np.float32)

    baseline_memory = runner.OnlineResidualMemory(baseline_state)
    candidate_memory = runner.OnlineResidualMemory(decoded_state)
    if (baseline_memory.cold_start_endpoint != int(endpoints[0]) or
            candidate_memory.cold_start_endpoint != int(endpoints[0])):
        raise ValueError("Cold-start endpoint mismatch")
    start = time.perf_counter()
    base_prediction, truth = runner.collect(
        loaders["val_loader"], endpoints, model, scaler, baseline_memory,
        device, 32)
    baseline_seconds = time.perf_counter() - start
    start = time.perf_counter()
    coded_prediction, coded_truth = runner.collect(
        loaders["val_loader"], endpoints, model, scaler, candidate_memory,
        device, 32)
    coded_seconds = time.perf_counter() - start
    if not np.array_equal(truth, coded_truth):
        raise AssertionError("Validation truth differs between passes")
    equal = np.array_equal(base_prediction, coded_prediction)
    base_mae = runner.metrics(base_prediction, truth)["mae"]
    coded_mae = runner.metrics(coded_prediction, coded_truth)["mae"]
    return {
        "dataset": dataset,
        "seed": seed,
        "validation_samples": int(len(endpoints)),
        "baseline_val_mae": base_mae,
        "bitpacked_val_mae": coded_mae,
        "delta_mae": coded_mae - base_mae,
        "prediction_arrays_bitwise_equal": equal,
        "bitpacked_state_bytes": candidate_path.stat().st_size,
        "decode_seconds": decode_seconds,
        "baseline_full_val_seconds": baseline_seconds,
        "bitpacked_full_val_seconds": coded_seconds,
        "test_evaluated": False,
    }


def main() -> None:
    results = [run_dataset(dataset, 1) for dataset in DATASETS]
    output = {
        "title": "Exact delta-bitpacked state full Validation equivalence",
        "date": "2026-10-23",
        "scope": "Frozen s1 Student checkpoints and Validation only; Test not loaded/evaluated.",
        "results": results,
        "limitations": ["Single full Validation pass per state", "No independent checkpoint confirmation", "Runtime seconds are single-run diagnostics"],
    }
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
