"""Draw the paper figure containing MAE comparison and PEMS04 validation loss.

The left panel uses the MAE values reported in the paper table.  The right
panel reads the real validation loss from the PEMS04 training log.  If the log
is unavailable, it falls back to ``pems04_validation_loss.csv`` in this folder.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOG_FILE = (
    PROJECT_ROOT
    / "EXP_experiments"
    / "TLAST_improved"
    / "PEMS04"
    / "TLAST_improved_PEMS04_12to12_exp07170128_s1"
    / "TLAST_improved_PEMS04_12to12_exp07170128_s1.log"
)
CSV_FILE = SCRIPT_DIR / "pems04_validation_loss.csv"
OUTPUT_STEM = SCRIPT_DIR / "mae_and_validation_loss"


METHODS = [
    "HA",
    "DCRNN",
    "AGCRN",
    "STGCN",
    "MTGNN",
    "STNorm",
    "PDFormer",
    "GMAN",
    "STAEformer",
    "HyperD",
    "HARP",
]

MAE = {
    "PEMS03": [32.62, 15.54, 15.24, 15.83, 14.85, 15.32,
               14.94, 16.87, 15.35, 14.69, 14.19],
    "PEMS04": [42.35, 19.63, 19.38, 19.57, 19.17, 18.96,
               18.36, 19.14, 18.22, 18.20, 17.63],
    "PEMS07": [49.03, 21.16, 20.57, 21.74, 20.89, 20.50,
               19.97, 20.97, 19.14, 19.37, 18.57],
}

# Soft, print-friendly colors.
COLORS = {
    "PEMS03": "#6587AC",
    "PEMS04": "#C49A70",
    "PEMS07": "#819E82",
    "validation": "#B8403A",
}

LINE_STYLES = {
    "PEMS03": ("-", "o"),
    "PEMS04": ("--", "s"),
    "PEMS07": (":", "^"),
}


def read_validation_loss() -> tuple[np.ndarray, np.ndarray]:
    """Return epoch and validation-loss arrays from the real training output."""
    if LOG_FILE.exists():
        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
        rows = re.findall(
            r"Epoch:\s*(\d+),\s*Train Loss:.*?Valid Loss:\s*([0-9.]+)",
            text,
        )
        if not rows:
            raise RuntimeError(f"No validation-loss records found in {LOG_FILE}")
        epochs = np.asarray([int(epoch) for epoch, _ in rows], dtype=int)
        losses = np.asarray([float(loss) for _, loss in rows], dtype=float)
        return epochs, losses

    if CSV_FILE.exists():
        with CSV_FILE.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        epochs = np.asarray([int(row["epoch"]) for row in rows], dtype=int)
        losses = np.asarray(
            [float(row["validation_loss"]) for row in rows], dtype=float
        )
        return epochs, losses

    raise FileNotFoundError(
        "Neither the PEMS04 training log nor pems04_validation_loss.csv exists."
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "axes.edgecolor": "#52657A",
            "axes.linewidth": 0.8,
            "xtick.color": "#52657A",
            "ytick.color": "#52657A",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_figure() -> None:
    configure_style()
    epochs, validation_loss = read_validation_loss()

    fig, (ax_mae, ax_loss) = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.2),
        gridspec_kw={"width_ratios": [1.52, 1.0]},
    )
    fig.patch.set_facecolor("white")

    # (a) Three datasets in one line chart.
    x = np.arange(len(METHODS))
    for dataset, values in MAE.items():
        linestyle, marker = LINE_STYLES[dataset]
        ax_mae.plot(
            x,
            values,
            label=dataset,
            color=COLORS[dataset],
            linestyle=linestyle,
            linewidth=1.8,
            marker=marker,
            markersize=5.4,
            markerfacecolor="white",
            markeredgecolor=COLORS[dataset],
            markeredgewidth=1.3,
        )

    ax_mae.set_title("(a) MAE comparison on three datasets", loc="left", pad=14,
                     fontweight="bold")
    ax_mae.set_xlabel("Methods", labelpad=10)
    ax_mae.set_ylabel("MAE", rotation=90, labelpad=18)
    ax_mae.set_xlim(-0.25, len(METHODS) - 0.75)
    ax_mae.set_ylim(10, 52)
    ax_mae.set_yticks([10, 20, 30, 40, 50])
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(METHODS, rotation=48, ha="right")
    ax_mae.grid(axis="y", color="#D9E1EA", linestyle="--", linewidth=0.7)
    ax_mae.set_axisbelow(True)
    ax_mae.legend(
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.70, 1.11),
        handlelength=2.5,
        columnspacing=2.0,
    )

    # Highlight only the HARP method name; no values are written in the plot.
    harp_tick = ax_mae.get_xticklabels()[-1]
    harp_tick.set_color(COLORS["validation"])
    harp_tick.set_fontweight("bold")

    # (b) Validation loss only: continuous line, without markers or annotations.
    ax_loss.plot(
        epochs,
        validation_loss,
        color=COLORS["validation"],
        linewidth=2.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Validation loss",
    )
    ax_loss.set_title("(b) PEMS04 validation-loss curve", loc="left", pad=14,
                      fontweight="bold")
    ax_loss.set_xlabel("Training epochs", labelpad=10)
    ax_loss.set_ylabel("Validation loss", rotation=90, labelpad=18)
    ax_loss.set_xlim(int(epochs.min()), int(epochs.max()))
    ax_loss.set_xticks([1, 20, 40, 60, 80, 100])

    loss_min = math.floor(float(validation_loss.min())) - 1
    loss_max = math.ceil(float(validation_loss.max())) + 1
    ax_loss.set_ylim(loss_min, loss_max)
    ax_loss.set_yticks(np.linspace(loss_min, loss_max, 6))
    ax_loss.grid(axis="y", color="#D9E1EA", linestyle="--", linewidth=0.7)
    ax_loss.set_axisbelow(True)
    ax_loss.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.72, 1.11),
        handlelength=2.8,
    )

    fig.suptitle(
        "MAE comparison and PEMS04 validation-loss convergence",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.22,
        top=0.78,
        wspace=0.20,
    )

    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUTPUT_STEM.with_suffix(f".{extension}"),
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    draw_figure()

