"""Choose one PEMS07 checkpoint using only the frozen final Validation state."""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from adaptive_lossless_state_codec import load_runtime_state


def load_runner(seed):
    path = ROOT / "srm_test_once/run_frozen_srm_test_once.py"
    previous = sys.argv
    sys.argv = [str(path), "--dataset", "PEMS07", "--seed", str(seed),
                "--mode", "sparse"]
    try:
        spec = importlib.util.spec_from_file_location(f"seed_runner_{seed}", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        return runner
    finally:
        sys.argv = previous


def evaluate(seed, state, endpoints):
    runner = load_runner(seed)
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
        raise ValueError("Expected frozen retrieval-aware Student")
    args.use_analog_lowrank_retrieval_residual = 0
    args.use_analog_exact_match_residual = 0
    args.teacher_predictions_path = "none"
    args.kd_alpha = 0
    logger = logging.getLogger("single_seed_selection")
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
    memory = runner.OnlineResidualMemory(state)
    if memory.cold_start_endpoint != int(endpoints[0]):
        raise ValueError("Validation state alignment mismatch")
    prediction, truth = runner.collect(
        loaders["val_loader"], endpoints, model, scaler, memory, device, 32)
    return dict(seed=seed, checkpoint=str(experiment / "best.pth"),
                validation_raw=runner.metrics(prediction, truth),
                validation_count=len(truth))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state_path = HERE / "packages/online/adaptive_lossless_val_v2/pems07_val_selected.alsc"
    state = load_runtime_state(state_path)
    with np.load(ROOT / "datasets/PEMS07/PEMS07_12to12_index_scaler.npz",
                 allow_pickle=False) as index:
        endpoints = np.asarray(index["idx_val"], dtype=np.int64)
    rows = []
    for seed in (1, 2, 3):
        row = evaluate(seed, state, endpoints)
        rows.append(row)
        print("PEMS07 seed", seed, "Validation raw MAE", row["validation_raw"]["mae"],
              flush=True)
    winner = min(rows, key=lambda row: (row["validation_raw"]["mae"], row["seed"]))
    args.output.write_text(json.dumps(dict(
        selection_split="Validation", test_used=False,
        selection_metric="raw MAE of current frozen final codec state",
        state_path=str(state_path), rows=rows, selected_seed=winner["seed"]),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("SELECTED", winner["seed"], flush=True)


if __name__ == "__main__":
    main()
