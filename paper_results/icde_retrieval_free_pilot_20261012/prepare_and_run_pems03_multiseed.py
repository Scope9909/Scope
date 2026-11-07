"""Prepare causal PEMS03 float32 memory, verify parity, then run three seeds."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / "_runtime_deps" / "python311_embed" / "python.exe"
SOURCE = ROOT / "datasets" / "PEMS03" / "PEMS03_12to12_analog_k4_exact_mean.npz"
FULL_REFERENCE = (ROOT / "paper_results" /
                  "pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_network_only" /
                  "selected_predictions.npz")
EXPERIMENT = (ROOT / "EXP_experiments" / "TLAST_improved" / "PEMS03" /
              "TLAST_improved_PEMS03_12to12_exp10101500_s1")
CALIBRATION = (ROOT / "paper_results" /
               "pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s1_valcal" /
               "calibration.json")


def run_stage(name: str, command: list[str], required: list[Path],
              output_dir: Path) -> None:
    if all(path.is_file() for path in required):
        print(f"existing {name}", flush=True)
        return
    print(f"start {name}", flush=True)
    log = output_dir / f"{name}.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([str(PYTHON), "-B", *command], cwd=ROOT,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   check=False)
    if completed.returncode != 0 or not all(path.is_file() for path in required):
        raise RuntimeError(f"{name} failed; inspect {log}")
    print(f"done {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    benchmark = str(HERE / "benchmark_online_end_to_end.py")
    train_state = str(HERE / "build_online_train_state.py")
    cold_state = str(HERE / "build_cold_start_online_state.py")
    backbone_json = out / "pems03_backbone_screen_b32.json"
    backbone_predictions = out / "pems03_backbone_screen_predictions.npz"
    run_stage("pems03_backbone_screen", [
        benchmark, "--exp-dir", str(EXPERIMENT), "--mode", "backbone",
        "--batch-size", "32", "--warmup-passes", "0", "--repeats", "1",
        "--output", str(backbone_json),
        "--predictions-output", str(backbone_predictions),
    ], [backbone_json, backbone_predictions], out)

    selection = out / "pems03_frozen_selection"
    run_stage("pems03_frozen_selection", [
        str(HERE / "select_pems03_frozen_sparse.py"),
        "--backbone-predictions", str(backbone_predictions),
        "--full-predictions", str(FULL_REFERENCE),
        "--full-cache", str(SOURCE),
        "--output-dir", str(selection),
    ], [selection / "plan.json", selection / "screen.json"], out)

    for mode in ("sparse", "full"):
        source = out / f"pems03_{mode}_online_state.npz"
        source_report = out / f"pems03_{mode}_online_state.json"
        build_command = [
            train_state, "--dataset", "PEMS03", "--full-cache", str(SOURCE),
            "--output-state", str(source), "--output-report", str(source_report),
        ]
        if mode == "full":
            build_command.append("--all-nodes")
        else:
            build_command.extend(["--plan", str(selection / "plan.json")])
        run_stage(f"pems03_{mode}_online_state", build_command,
                  [source, source_report], out)
        cold = out / f"pems03_{mode}_cold_state.npz"
        cold_report = out / f"pems03_{mode}_cold_state.json"
        run_stage(f"pems03_{mode}_cold_state", [
            cold_state, "--dataset", "PEMS03",
            "--source-state", str(source),
            "--output-state", str(cold),
            "--output-report", str(cold_report),
        ], [cold, cold_report], out)

    parity_json = out / "pems03_full_parity_b32.json"
    run_stage("pems03_full_parity_b32", [
        benchmark, "--exp-dir", str(EXPERIMENT), "--mode", "full",
        "--state", str(out / "pems03_full_cold_state.npz"),
        "--batch-size", "32", "--warmup-passes", "0", "--repeats", "1",
        "--reference-predictions", str(FULL_REFERENCE),
        "--output", str(parity_json),
    ], [parity_json], out)
    parity = json.loads(parity_json.read_text(encoding="utf-8"))
    difference = parity["prediction_parity"]
    # The legacy float32 cache was produced by a separate offline reduction.
    # Equal-distance candidates can be accumulated in a different order during
    # live exact scanning.  Audit both pointwise drift and the metric impact.
    reference_mae = json.loads(CALIBRATION.read_text(encoding="utf-8"))["baseline_val"]["mae"]
    online_mae = parity["raw_validation_metrics"]["mae"]
    if (difference["mean_absolute_difference"] > 0.005 or
            difference["max_absolute_difference"] > 0.15 or
            abs(online_mae - reference_mae) > 1e-4):
        raise ValueError(f"PEMS03 full online parity failed: {difference}")
    print("PEMS03 full-memory Validation numerical audit accepted:", difference,
          "MAE delta=", online_mae - reference_mae, flush=True)

    run_stage("pems03_three_seed_batch1", [
        str(HERE / "run_multiseed_single_request.py"),
        "--output-dir", str(out), "--datasets", "PEMS03",
        "--seeds", "1", "2", "3",
    ], [out / f"pems03_s{seed}_{mode}_b1_r3.json"
        for seed in (1, 2, 3) for mode in ("backbone", "sparse", "full")], out)
    print("PEMS03 three-seed Validation benchmark complete", flush=True)


if __name__ == "__main__":
    main()
