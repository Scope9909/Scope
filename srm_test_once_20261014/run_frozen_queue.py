"""Resume-safe queue for the frozen 36-job SRM system Test comparison."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PYTHON = ROOT / "_runtime_deps" / "python311_embed" / "python.exe"
RUNNER = OUT / "run_frozen_srm_test_once.py"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    manifest = json.loads((OUT / "frozen_manifest.json").read_text(encoding="utf-8"))
    if digest(RUNNER) != manifest["runner_sha256"]:
        raise ValueError("Frozen Test runner changed after plan freeze")
    if len(manifest["jobs"]) != manifest["expected_jobs"] or manifest["expected_jobs"] != 36:
        raise ValueError("Frozen job count changed")
    for index, job in enumerate(manifest["jobs"], 1):
        for name in ("checkpoint", "calibration", "validation_state", "test_state"):
            path = job[name]
            if path is not None and digest(Path(path)) != job[f"{name}_sha256"]:
                raise ValueError(f"Frozen input changed: {path}")
        dataset, seed, mode = job["dataset"], job["seed"], job["mode"]
        output = OUT / f"{dataset.lower()}_s{seed}_{mode}_test_once.json"
        if output.exists():
            report = json.loads(output.read_text(encoding="utf-8"))
            if not (report["protocol"] == "frozen_srm_online_test_once_v1" and
                    report["dataset"] == dataset and report["seed_label"] == seed and
                    report["mode"] == mode and report["test_evaluations"] == 1):
                raise ValueError(f"Existing Test result failed protocol check: {output}")
            print(f"[{index}/36] existing {dataset} s{seed} {mode}", flush=True)
            continue
        command = [str(PYTHON), "-B", str(RUNNER), "--dataset", dataset,
                   "--seed", str(seed), "--mode", mode]
        log = OUT / f"{dataset.lower()}_s{seed}_{mode}_test_once.log"
        print(f"[{index}/36] start {dataset} s{seed} {mode}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0 or not output.exists():
            raise RuntimeError(f"Frozen Test job failed: {log}")
        report = json.loads(output.read_text(encoding="utf-8"))
        if not (report["test_evaluations"] == 1 and report["test_evaluated"] and
                not report["test_based_selection"] and not report["query_cache_loaded"]):
            raise ValueError(f"Frozen Test result failed integrity check: {output}")
        print(f"[{index}/36] done {dataset} s{seed} {mode} "
              f"raw={report['test_raw']['mae']:.6f} "
              f"cal={report['test_fixed_calibrated']['mae']:.6f}", flush=True)
    print("All 36 frozen SRM Test jobs complete", flush=True)


if __name__ == "__main__":
    main()
