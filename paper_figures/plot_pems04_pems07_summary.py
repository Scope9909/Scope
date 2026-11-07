"""Generate a publication-ready 2x2 PEMS04/PEMS07 summary figure.

Top row: test-MAE bar charts for all methods in the paper table, including HA.
Bottom row: real validation-loss curves parsed from the two backbone logs.

The script uses the repository's direct SVG/ReportLab renderer, avoiding a
runtime dependency on Matplotlib. It writes vector SVG/PDF plus a 600-dpi PNG.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from generate_harp_curve_components import Scene, save_scene


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LOG_FILES = {
    "PEMS04": (
        PROJECT_ROOT
        / "EXP_experiments/TLAST_improved/PEMS04"
        / "TLAST_improved_PEMS04_12to12_exp07170128_s1"
        / "TLAST_improved_PEMS04_12to12_exp07170128_s1.log"
    ),
    "PEMS07": (
        PROJECT_ROOT
        / "EXP_experiments/PEMS07/PEMS07"
        / "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_backbone"
        / "TLAST_improved_PEMS07_12to12_exppaper_pems07_s1_backbone.log"
    ),
}

METHODS = [
    "HA", "DCRNN", "AGCRN", "STGCN", "MTGNN", "STNorm", "PDFormer",
    "GMAN", "STAEformer", "TLAST", "HyperD", "Hist2Res",
]

MAE = {
    "PEMS04": [
        42.35, 19.63, 19.38, 19.57, 19.17, 18.96, 18.36,
        19.14, 18.22, 17.90, 18.20, 17.63,
    ],
    "PEMS07": [
        49.03, 21.16, 20.57, 21.74, 20.89, 20.50, 19.97,
        20.97, 19.14, 18.97, 19.37, 18.57,
    ],
}

WHITE = "#FFFFFF"
INK = "#263746"
AXIS = "#52657A"
GRID = "#D9E1EA"
TLAST_BLUE = "#0072B2"
HIST2RES_ORANGE = "#D55E00"
PEMS04_BLUE = "#0072B2"
PEMS07_GREEN = "#009E73"

# Font sizes are deliberately larger than the original values because this
# 2x2 canvas is finally scaled down to a single ICASSP column.  Keeping the
# values centralized also makes it easy to tune the exported figure later.
TICK_FONT_SIZE = 22
METHOD_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 28
PANEL_TITLE_FONT_SIZE = 30

# A restrained, color-blind-conscious palette. The same method keeps the same
# color in both datasets; Hist2Res retains the strongest visual emphasis.
METHOD_COLORS = {
    "HA": "#7F8C8D",
    "DCRNN": "#4E79A7",
    "AGCRN": "#76A5AF",
    "STGCN": "#8FAF88",
    "MTGNN": "#8064A2",
    "STNorm": "#C9A66B",
    "PDFormer": "#D39B6A",
    "GMAN": "#9B7E76",
    "STAEformer": "#B07AA1",
    "TLAST": TLAST_BLUE,
    "HyperD": "#5F9EA0",
    "Hist2Res": HIST2RES_ORANGE,
}


def read_validation_loss(dataset: str) -> tuple[list[int], list[float]]:
    path = LOG_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Missing {dataset} training log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = re.findall(
        r"Epoch:\s*(\d+),\s*Train Loss:.*?Valid Loss:\s*([0-9.]+)",
        text,
    )
    if not rows:
        raise RuntimeError(f"No validation-loss records found in {path}")
    by_epoch = {int(epoch): float(loss) for epoch, loss in rows}
    epochs = sorted(by_epoch)
    return epochs, [by_epoch[epoch] for epoch in epochs]


def bar_color(method: str) -> tuple[str, str]:
    fill = METHOD_COLORS[method]
    return fill, "#273746" if method == "Hist2Res" else fill


def draw_panel_frame(scene: Scene, x: float, y: float, w: float, h: float) -> None:
    scene.rect(x, y, w, h, fill=WHITE, stroke="#BCC7D1", sw=0.9, radius=3)


def draw_mae_panel(
    scene: Scene,
    x: float,
    y: float,
    w: float,
    h: float,
    dataset: str,
    letter: str,
) -> None:
    draw_panel_frame(scene, x, y, w, h)
    left, right, top, bottom = 68, 18, 30, 160
    px, py = x + left, y + top
    pw, ph = w - left - right, h - top - bottom
    # Use the same focused scale in both panels so differences among the
    # competitive methods remain visible. HA exceeds this range and is
    # intentionally clipped at the top boundary (documented in the caption).
    ymax = 25.0

    for tick in range(0, int(ymax) + 1, 5):
        gy = py + ph - (tick / ymax) * ph
        scene.line(px, gy, px + pw, gy, color=GRID, sw=0.8, dash="4,4")
        scene.text(
            px - 12,
            gy,
            str(tick),
            size=TICK_FONT_SIZE,
            color=AXIS,
            anchor="right",
        )

    scene.line(px, py, px, py + ph, color=AXIS, sw=1.0)
    scene.line(px, py + ph, px + pw, py + ph, color=AXIS, sw=1.0)
    scene.text(
        x + 18,
        py + ph / 2,
        "MAE",
        size=AXIS_LABEL_FONT_SIZE,
        bold=True,
        color=INK,
        angle=-90,
    )

    slot = pw / len(METHODS)
    bar_w = slot * 0.63
    for index, (method, value) in enumerate(zip(METHODS, MAE[dataset])):
        cx = px + (index + 0.5) * slot
        displayed_value = min(value, ymax)
        bh = (displayed_value / ymax) * ph
        fill, edge = bar_color(method)
        scene.rect(
            cx - bar_w / 2,
            py + ph - bh,
            bar_w,
            bh,
            fill=fill,
            stroke=edge,
            sw=0.7,
            radius=0,
        )
        scene.text(
            cx + 5,
            py + ph + 18,
            method,
            size=METHOD_FONT_SIZE,
            color=HIST2RES_ORANGE if method == "Hist2Res" else INK,
            bold=method == "Hist2Res",
            anchor="right",
            angle=-48,
        )

    scene.text(
        x + w / 2,
        y + h - 17,
        f"({letter}) {dataset} - MAE",
        size=PANEL_TITLE_FONT_SIZE,
        bold=True,
        color=INK,
    )


def nice_loss_bounds(values: list[float]) -> tuple[float, float, float]:
    lower = math.floor(min(values)) - 1
    upper = math.ceil(max(values)) + 1
    step = 5.0
    return float(lower), float(upper), step


def draw_loss_panel(
    scene: Scene,
    x: float,
    y: float,
    w: float,
    h: float,
    dataset: str,
    letter: str,
) -> None:
    draw_panel_frame(scene, x, y, w, h)
    epochs, losses = read_validation_loss(dataset)
    color = PEMS04_BLUE if dataset == "PEMS04" else PEMS07_GREEN
    left, right, top, bottom = 78, 22, 30, 125
    px, py = x + left, y + top
    pw, ph = w - left - right, h - top - bottom
    ymin, ymax, ystep = nice_loss_bounds(losses)

    tick = math.ceil(ymin / ystep) * ystep
    while tick <= ymax:
        gy = py + ph - ((tick - ymin) / (ymax - ymin)) * ph
        scene.line(px, gy, px + pw, gy, color=GRID, sw=0.8, dash="4,4")
        scene.text(
            px - 12,
            gy,
            f"{tick:.0f}",
            size=TICK_FONT_SIZE,
            color=AXIS,
            anchor="right",
        )
        tick += ystep

    for epoch in (1, 20, 40, 60, 80, 100):
        tx = px + ((epoch - epochs[0]) / (epochs[-1] - epochs[0])) * pw
        scene.line(tx, py + ph, tx, py + ph + 4, color=AXIS, sw=0.8)
        scene.text(
            tx,
            py + ph + 25,
            str(epoch),
            size=TICK_FONT_SIZE,
            color=AXIS,
        )

    scene.line(px, py, px, py + ph, color=AXIS, sw=1.0)
    scene.line(px, py + ph, px + pw, py + ph, color=AXIS, sw=1.0)
    scene.text(
        x + 20,
        py + ph / 2,
        "Validation loss",
        size=AXIS_LABEL_FONT_SIZE,
        bold=True,
        color=INK,
        angle=-90,
    )
    scene.text(
        px + pw / 2,
        y + h - 66,
        "Training epoch",
        size=AXIS_LABEL_FONT_SIZE,
        bold=True,
        color=INK,
    )

    points = []
    for epoch, loss in zip(epochs, losses):
        cx = px + ((epoch - epochs[0]) / (epochs[-1] - epochs[0])) * pw
        cy = py + ph - ((loss - ymin) / (ymax - ymin)) * ph
        points.append((cx, cy))
    scene.polyline(points, color=color, sw=3.0)

    best_index = min(range(len(losses)), key=losses.__getitem__)
    bx, by = points[best_index]
    scene.rect(bx - 3.5, by - 3.5, 7, 7, fill=WHITE, stroke=color, sw=1.5, radius=3.5)
    scene.text(
        x + w / 2,
        y + h - 17,
        f"({letter}) {dataset} - Validation loss",
        size=PANEL_TITLE_FONT_SIZE,
        bold=True,
        color=INK,
    )


def build_figure() -> Scene:
    scene = Scene(1400, 980)
    margin_x, gap_x = 30, 24
    panel_w = (scene.width - 2 * margin_x - gap_x) / 2
    top_y, top_h = 20, 430
    bottom_y, bottom_h = 474, 476
    right_x = margin_x + panel_w + gap_x

    draw_mae_panel(scene, margin_x, top_y, panel_w, top_h, "PEMS04", "a")
    draw_mae_panel(scene, right_x, top_y, panel_w, top_h, "PEMS07", "b")
    draw_loss_panel(scene, margin_x, bottom_y, panel_w, bottom_h, "PEMS04", "c")
    draw_loss_panel(scene, right_x, bottom_y, panel_w, bottom_h, "PEMS07", "d")
    return scene


def main() -> None:
    scene = build_figure()
    created = save_scene(
        scene,
        SCRIPT_DIR,
        "pems04_pems07_mae_validation_loss",
        dpi=600,
        transparent_svg=False,
    )
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
