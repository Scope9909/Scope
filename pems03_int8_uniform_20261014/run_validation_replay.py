"""Run frozen PEMS03 INT8 sparse/full online Validation replay for three seeds."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
PYTHON = ROOT / "_runtime_deps" / "python311_embed" / "python.exe"
BENCHMARK = ROOT / "paper_results" / "icde_retrieval_free_pilot_20261012" / "benchmark_online_end_to_end.py"
EXPERIMENTS = {
    1: "TLAST_improved_PEMS03_12to12_exp10101500_s1",
    2: "TLAST_improved_PEMS03_12to12_exp10101506_s2",
    3: "TLAST_improved_PEMS03_12to12_exp10101513_s3",
}


def main() -> None:
    planned = [f"pems03_s{seed}_{mode}_int8_b1_r3.json"
               for seed in EXPERIMENTS for mode in ("sparse", "full")]
    for index, (seed, mode) in enumerate(
        (item for seed in EXPERIMENTS for item in ((seed, "sparse"), (seed, "full"))), 1
    ):
        output = OUT / f"pems03_s{seed}_{mode}_int8_b1_r3.json"
        if output.exists():
            report = json.loads(output.read_text(encoding="utf-8"))
            if not (report["dataset"] == "PEMS03" and report["mode"] == mode
                    and report["retrieval_format"] == "int8"
                    and report["batch_size"] == 1 and report["warmup_passes"] == 1
                    and report["repeats"] == 3 and not report["test_evaluated"]):
                raise ValueError(f"Existing report has the wrong protocol: {output}")
            print(f"[{index}/6] existing seed={seed} mode={mode}", flush=True)
            continue
        checkpoint = (ROOT / "EXP_experiments" / "TLAST_improved" / "PEMS03" /
                      EXPERIMENTS[seed])
        calibration = (ROOT / "paper_results" /
                       f"pems03_lite_r256_kdfix3_a30_lr1e4_fixedteacher_s{seed}_valcal" /
                       "calibration.json")
        state = OUT / f"{mode}_cold_state.npz"
        for path in (checkpoint / "best.pth", checkpoint / "config.pkl", calibration, state):
            if not path.is_file():
                raise FileNotFoundError(path)
        command = [str(PYTHON), "-B", str(BENCHMARK),
                   "--exp-dir", str(checkpoint), "--mode", mode,
                   "--state", str(state), "--batch-size", "1",
                   "--warmup-passes", "1", "--repeats", "3",
                   "--calibration", str(calibration), "--output", str(output)]
        log = OUT / f"pems03_s{seed}_{mode}_int8_b1_r3.log"
        print(f"[{index}/6] start seed={seed} mode={mode}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError(f"Validation replay failed: {log}")
        report = json.loads(output.read_text(encoding="utf-8"))
        if (report["retrieval_format"] != "int8" or report["test_evaluated"]
                or report["query_cache_loaded"]):
            raise ValueError(f"Result failed protocol check: {output}")
        print(f"[{index}/6] done seed={seed} mode={mode} "
              f"MAE={report['fixed_validation_calibration']['mean_fixed_calibrated_mae']:.6f} "
              f"p50={report['end_to_end_batch_latency_ms_p50']:.3f}ms", flush=True)
        (OUT / "run_manifest.json").write_text(json.dumps({
            "protocol": "pems03_int8_three_seed_validation_only_v1",
            "planned": planned,
            "completed": [name for name in planned if (OUT / name).is_file()],
            "test_evaluated": False,
        }, indent=2), encoding="utf-8")
    print("PEMS03 INT8 Validation replay complete", flush=True)


if __name__ == "__main__":
    main()
