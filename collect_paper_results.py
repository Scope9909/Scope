"""Collect named paper runs into per-run and mean/std CSV tables."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "paper_results"

TEST_PATTERN = re.compile(
    r"On average over 12 horizons, Test MAE: ([0-9.]+), "
    r"Test MAPE: ([0-9.]+), Test RMSE: ([0-9.]+)"
)
VALID_PATTERN = re.compile(r"The valid loss on best model is ([0-9.]+)")
PARAM_PATTERN = re.compile(r"trainable_parameters=(\d+)")
def last_match(pattern: re.Pattern[str], text: str):
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def read_run(directory: Path, run_pattern: re.Pattern[str]) -> dict | None:
    run_match = run_pattern.search(directory.name)
    if run_match is None:
        return None
    logs = sorted(directory.glob("*.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return None
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in logs)
    test = last_match(TEST_PATTERN, text)
    valid = last_match(VALID_PATTERN, text)
    trainable = last_match(PARAM_PATTERN, text)
    if test is None:
        return None
    return {
        "run_id": run_match.group(1),
        "seed": int(run_match.group(2)),
        "stage": run_match.group(3),
        "valid_mae": float(valid.group(1)) if valid else "",
        "test_mae": float(test.group(1)),
        "test_mape": float(test.group(2)),
        "test_rmse": float(test.group(3)),
        "trainable_parameters": int(trainable.group(1)) if trainable else "",
        "checkpoint": str((directory / "best.pth").relative_to(ROOT)),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["stage"], []).append(row)
    output = []
    for stage, group in sorted(grouped.items()):
        item = {"stage": stage, "runs": len(group)}
        for metric in ("valid_mae", "test_mae", "test_mape", "test_rmse"):
            values = [
                float(row[metric]) for row in group if row[metric] != ""
            ]
            item[f"{metric}_mean"] = (
                round(statistics.mean(values), 6) if values else ""
            )
            item[f"{metric}_std"] = (
                round(statistics.stdev(values), 6)
                if len(values) >= 2
                else ""
            )
        output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMS04")
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.upper()
    prefix = args.prefix or f"paper_{dataset.lower()}"
    experiment_root = ROOT / "EXP_experiments" / "TLAST_improved" / dataset
    run_pattern = re.compile(
        rf"exp({re.escape(prefix)}_s(\d+)_(.+))$"
    )
    directories = sorted(
        path
        for path in experiment_root.glob(f"*exp{prefix}*")
        if path.is_dir()
    )
    rows = [
        row
        for path in directories
        if (row := read_run(path, run_pattern)) is not None
    ]
    rows.sort(key=lambda row: (row["stage"], row["seed"]))

    run_fields = [
        "run_id",
        "seed",
        "stage",
        "valid_mae",
        "test_mae",
        "test_mape",
        "test_rmse",
        "trainable_parameters",
        "checkpoint",
    ]
    output_stem = dataset.lower()
    write_csv(OUTPUT_ROOT / f"{output_stem}_paper_runs.csv", rows, run_fields)

    aggregate_rows = aggregate(rows)
    aggregate_fields = [
        "stage",
        "runs",
        "valid_mae_mean",
        "valid_mae_std",
        "test_mae_mean",
        "test_mae_std",
        "test_mape_mean",
        "test_mape_std",
        "test_rmse_mean",
        "test_rmse_std",
    ]
    write_csv(
        OUTPUT_ROOT / f"{output_stem}_paper_aggregate.csv",
        aggregate_rows,
        aggregate_fields,
    )
    print(
        f"Collected {len(rows)} completed runs; "
        f"wrote {OUTPUT_ROOT / f'{output_stem}_paper_runs.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
