"""Reproduce a horizon-wise Test MAE figure for the frozen seed-1 DARC runs.

Run ``collect`` with the repository's Python 3.11/CUDA environment, then
``plot`` with any Python environment that has Matplotlib. No model selection
or parameter fitting is performed using Test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = (
    ROOT
    / "contrib3_index_codec_20261020"
    / "packages"
    / "online"
    / "adaptive_lossless_test_20261023_v1"
)
DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")


def horizon_mae(prediction, truth):
    import numpy as np

    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("Expected matching [sample, horizon, node] arrays")
    values = []
    for h in range(prediction.shape[1]):
        mask = np.round(truth[:, h]) > 0
        if not np.any(mask):
            raise ValueError(f"Horizon {h + 1} has no valid target")
        values.append(
            float(
                np.abs(
                    prediction[:, h][mask].astype(np.float64)
                    - truth[:, h][mask].astype(np.float64)
                ).mean()
            )
        )
    return values


def collect(output: Path):
    import numpy as np

    sys.path.insert(0, str(ROOT / "contrib3_index_codec_20261020"))
    import eval_adaptive_lossless_test as evaluator

    published = {
        item["dataset"]: item
        for item in json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))[
            "results"
        ]
    }
    original_loader = evaluator.previous._load_runner
    observations = {}

    for dataset in DATASETS:
        captured = {}

        def capture_loader(name):
            runner = original_loader(name)
            original_collect = runner.collect
            original_metrics = runner.metrics

            def collect_with_base(loader, endpoints, model, scaler, memory, device, batch_size):
                # Only the Test state has no cold-start endpoint. The same
                # frozen model and official loader are used for the base curve.
                if memory is not None and memory.cold_start_endpoint is None:
                    base_pred, base_truth = original_collect(
                        loader, endpoints, model, scaler, None, device, batch_size
                    )
                    captured["student_base"] = horizon_mae(base_pred, base_truth)
                prediction, truth = original_collect(
                    loader, endpoints, model, scaler, memory, device, batch_size
                )
                if memory is not None and memory.cold_start_endpoint is None:
                    if not np.array_equal(base_truth, truth):
                        raise AssertionError("Base and retrieval Test targets differ")
                return prediction, truth

            def metrics_with_horizons(prediction, truth):
                key = "darc_raw" if "darc_raw" not in captured else "darc_calibrated"
                captured[key] = horizon_mae(prediction, truth)
                return original_metrics(prediction, truth)

            runner.collect = collect_with_base
            runner.metrics = metrics_with_horizons
            return runner

        evaluator.previous._load_runner = capture_loader
        try:
            evaluated = evaluator.evaluate(
                dataset,
                evaluator.VALIDATION / f"{dataset.lower()}_val_selected.alsc",
                RESULTS / f"{dataset.lower()}_test_selected.alsc",
            )
        finally:
            evaluator.previous._load_runner = original_loader

        for curve, summary_key in (
            ("darc_raw", "test_raw_metrics"),
            ("darc_calibrated", "test_calibrated_metrics"),
        ):
            computed = float(np.mean(captured[curve]))
            expected = published[dataset][summary_key]["mae"]
            if abs(computed - expected) > 1e-5:
                raise AssertionError(
                    f"{dataset} {curve}: horizon mean {computed} != published {expected}"
                )
            if abs(evaluated[summary_key]["mae"] - expected) > 1e-5:
                raise AssertionError(f"{dataset} {curve}: rerun differs from frozen report")

        observations[dataset] = {
            "seed": 1,
            "test_count": evaluated["test_count"],
            "horizon_mae": captured,
            "verified_raw_mae": published[dataset]["test_raw_metrics"]["mae"],
            "verified_calibrated_mae": published[dataset]["test_calibrated_metrics"]["mae"],
        }
        print(dataset, "verified", observations[dataset]["verified_calibrated_mae"], flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "protocol": "frozen_seed1_darc_test_horizon_mae",
                "mask": "round(truth) > 0",
                "aggregation": "masked MAE at each horizon; equal-weight mean checks main table",
                "source_report": str(RESULTS / "results.json"),
                "datasets": observations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot(data_path: Path, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(data_path.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.3))
    curves = (
        ("student_base", "Student base", "#8C6BB1", (0, (2, 2)), "s"),
        ("darc_raw", "DARC raw", "#258B7B", "--", "o"),
        ("darc_calibrated", "DARC calibrated", "#174D94", "-", "D"),
    )
    steps = np.arange(1, 13)
    handles = []
    for idx, (dataset, ax) in enumerate(zip(DATASETS, axes.flat)):
        metrics = data["datasets"][dataset]["horizon_mae"]
        for key, label, color, style, marker in curves:
            values = np.asarray(metrics[key], dtype=float)
            if len(values) != 12:
                raise ValueError(f"{dataset} {key}: expected 12 steps")
            line = ax.plot(
                steps,
                values,
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.5,
                marker=marker,
                markersize=2.8,
                markevery=(0, 2),
                zorder=3,
            )[0]
            if idx == 0:
                handles.append(line)
        ax.set_title(f"({chr(ord('a') + idx)}) {dataset}", pad=5)
        ax.set_xlim(1, 12)
        ax.set_xticks([1, 3, 6, 9, 12])
        ax.grid(axis="y", color="#D8DEE7", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9099A6")
        ax.tick_params(length=2.5, colors="#374151")
        if idx // 2 == 1:
            ax.set_xlabel("Prediction step")
        if idx % 2 == 0:
            ax.set_ylabel("Test MAE")
    fig.legend(
        handles,
        [item[1] for item in curves],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.92, bottom=0.18, wspace=0.22, hspace=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "plot"))
    parser.add_argument("--data", type=Path, default=ROOT / "tmp/pdfs/darc_horizon_mae_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/darc_horizon_mae.pdf")
    args = parser.parse_args()
    if args.mode == "collect":
        collect(args.data)
    else:
        plot(args.data, args.output)


if __name__ == "__main__":
    main()
