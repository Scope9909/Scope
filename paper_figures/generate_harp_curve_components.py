"""Generate reusable Hist2Res time-series curve components.

This script intentionally avoids Matplotlib.  It writes publication-friendly
SVG directly, uses ReportLab for PDF, and uses Poppler's ``pdftoppm`` for PNG.

The default curves are deterministic schematic signals for an architecture
figure; they are not experimental results.  To use real signals, pass a CSV
whose numeric rows contain at least ``history_length + future_length`` values.

Examples
--------
python paper_figures/generate_harp_curve_components.py
python paper_figures/generate_harp_curve_components.py --seed 11 --dpi 360
python paper_figures/generate_harp_curve_components.py --csv my_curves.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas as pdfcanvas
except ModuleNotFoundError:  # SVG output still works with the standard library.
    pdfmetrics = None
    TTFont = None
    pdfcanvas = None


# Restrained palette shared with the Hist2Res framework draft.
INK = "#172B4D"
MUTED = "#617184"
GRID = "#D9E1EA"
TEAL = "#168C82"
TEAL_LIGHT = "#E8F7F4"
TEAL_PALE = "#B9DED8"
BLUE = "#1F5DA8"
BLUE_LIGHT = "#E8F1FB"
ORANGE = "#D97706"
WHITE = "#FFFFFF"
FONT_REGULAR = Path(r"C:\Windows\Fonts\times.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")


def rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))


@dataclass
class Scene:
    width: float
    height: float

    def __post_init__(self) -> None:
        self.ops: list[tuple] = []

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = WHITE,
        stroke: str = GRID,
        sw: float = 1.0,
        radius: float = 7.0,
        alpha: float = 1.0,
    ) -> None:
        self.ops.append(("rect", x, y, w, h, fill, stroke, sw, radius, alpha))

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = INK,
        sw: float = 1.2,
        dash: str | None = None,
        alpha: float = 1.0,
    ) -> None:
        self.ops.append(("line", x1, y1, x2, y2, color, sw, dash, alpha))

    def polyline(
        self,
        points: Iterable[tuple[float, float]],
        *,
        color: str = INK,
        sw: float = 1.5,
        dash: str | None = None,
        alpha: float = 1.0,
    ) -> None:
        self.ops.append(("polyline", list(points), color, sw, dash, alpha))

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 13,
        color: str = INK,
        bold: bool = False,
        anchor: str = "middle",
        angle: float = 0.0,
    ) -> None:
        self.ops.append(("text", x, y, value, size, color, bold, anchor, angle))


def write_svg(scene: Scene, path: Path, *, transparent: bool) -> None:
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{scene.width}" '
            f'height="{scene.height}" viewBox="0 0 {scene.width} {scene.height}">'
        )
    ]
    if not transparent:
        parts.append('<rect width="100%" height="100%" fill="#FFFFFF"/>')

    for op in scene.ops:
        kind = op[0]
        if kind == "rect":
            _, x, y, w, h, fill, stroke, sw, radius, alpha = op
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'rx="{radius:.2f}" fill="{fill}" fill-opacity="{alpha:.3f}" '
                f'stroke="{stroke}" stroke-width="{sw:.2f}"/>'
            )
        elif kind == "line":
            _, x1, y1, x2, y2, color, sw, dash, alpha = op
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="{color}" stroke-width="{sw:.2f}" stroke-opacity="{alpha:.3f}" '
                f'stroke-linecap="round"{dash_attr}/>'
            )
        elif kind == "polyline":
            _, points, color, sw, dash, alpha = op
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" '
                f'stroke-width="{sw:.2f}" stroke-opacity="{alpha:.3f}" '
                f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
            )
        elif kind == "text":
            _, x, y, value, size, color, bold, anchor, angle = op
            svg_anchor = {"left": "start", "middle": "middle", "right": "end"}[anchor]
            weight = "700" if bold else "400"
            transform = (
                f' transform="rotate({angle:.2f} {x:.2f} {y:.2f})"'
                if angle else ""
            )
            parts.append(
                f'<text x="{x:.2f}" y="{y:.2f}" font-family="Times New Roman, Times, serif" '
                f'font-size="{size:.2f}" font-weight="{weight}" fill="{color}" '
                f'text-anchor="{svg_anchor}" dominant-baseline="middle"{transform}>'
                f'{html.escape(value)}</text>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_pdf(scene: Scene, path: Path, *, transparent: bool) -> None:
    if pdfcanvas is None:
        raise RuntimeError("ReportLab is not installed; PDF export is unavailable.")

    # PDF pages themselves are white in most viewers; all vector elements remain editable.
    regular_font = "Times-Roman"
    bold_font = "Times-Bold"
    if pdfmetrics is not None and TTFont is not None and FONT_REGULAR.exists() and FONT_BOLD.exists():
        if "HARPTimes" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("HARPTimes", str(FONT_REGULAR)))
            pdfmetrics.registerFont(TTFont("HARPTimes-Bold", str(FONT_BOLD)))
        regular_font = "HARPTimes"
        bold_font = "HARPTimes-Bold"

    c = pdfcanvas.Canvas(
        str(path),
        pagesize=(scene.width, scene.height),
        initialFontName=regular_font,
    )

    def py(y: float) -> float:
        return scene.height - y

    if not transparent:
        c.setFillColorRGB(1, 1, 1)
        c.rect(0, 0, scene.width, scene.height, fill=1, stroke=0)

    for op in scene.ops:
        kind = op[0]
        if kind == "rect":
            _, x, y, w, h, fill, stroke, sw, radius, alpha = op
            c.saveState()
            if hasattr(c, "setFillAlpha"):
                c.setFillAlpha(alpha)
            c.setFillColorRGB(*rgb(fill))
            c.setStrokeColorRGB(*rgb(stroke))
            c.setLineWidth(sw)
            c.roundRect(x, py(y + h), w, h, radius, fill=1, stroke=1)
            c.restoreState()
        elif kind == "line":
            _, x1, y1, x2, y2, color, sw, dash, alpha = op
            c.saveState()
            if hasattr(c, "setStrokeAlpha"):
                c.setStrokeAlpha(alpha)
            c.setStrokeColorRGB(*rgb(color))
            c.setLineWidth(sw)
            c.setLineCap(1)
            if dash:
                c.setDash(*[float(v) for v in dash.split(",")])
            c.line(x1, py(y1), x2, py(y2))
            c.restoreState()
        elif kind == "polyline":
            _, points, color, sw, dash, alpha = op
            c.saveState()
            if hasattr(c, "setStrokeAlpha"):
                c.setStrokeAlpha(alpha)
            c.setStrokeColorRGB(*rgb(color))
            c.setLineWidth(sw)
            c.setLineJoin(1)
            c.setLineCap(1)
            if dash:
                c.setDash(*[float(v) for v in dash.split(",")])
            p = c.beginPath()
            p.moveTo(points[0][0], py(points[0][1]))
            for x, y in points[1:]:
                p.lineTo(x, py(y))
            c.drawPath(p, fill=0, stroke=1)
            c.restoreState()
        elif kind == "text":
            _, x, y, value, size, color, bold, anchor, angle = op
            c.saveState()
            c.setFillColorRGB(*rgb(color))
            c.setFont(bold_font if bold else regular_font, size)
            c.translate(x, py(y))
            if angle:
                c.rotate(-angle)
            baseline = -size * 0.34
            if anchor == "left":
                c.drawString(0, baseline, value)
            elif anchor == "right":
                c.drawRightString(0, baseline, value)
            else:
                c.drawCentredString(0, baseline, value)
            c.restoreState()

    c.showPage()
    c.save()


def save_scene(
    scene: Scene,
    out_dir: Path,
    stem: str,
    *,
    dpi: int,
    transparent_svg: bool = True,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{stem}.svg"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    write_svg(scene, svg_path, transparent=transparent_svg)
    created = [svg_path]
    if pdfcanvas is None:
        return created

    write_pdf(scene, pdf_path, transparent=False)
    created.append(pdf_path)
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "1",
                "-singlefile",
                "-png",
                "-r",
                str(dpi),
                str(pdf_path),
                str(png_path.with_suffix("")),
            ],
            check=True,
        )
    except FileNotFoundError:
        return created
    created.append(png_path)
    return created


def smooth(values: Sequence[float], radius: int = 2) -> list[float]:
    result: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def schematic_series(count: int, length: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    series: list[list[float]] = []
    for i in range(count):
        phase = rng.uniform(-0.8, 0.8)
        amplitude = rng.uniform(0.78, 1.18)
        slope = rng.uniform(-0.018, 0.024)
        raw_noise = [rng.gauss(0.0, 0.18) for _ in range(length)]
        noise = smooth(raw_noise, radius=1)
        values = []
        for t in range(length):
            periodic = math.sin(2 * math.pi * t / 8.0 + phase)
            local = 0.45 * math.sin(2 * math.pi * t / 3.7 + 0.4 * i)
            values.append(amplitude * periodic + local + slope * t + noise[t])
        series.append(values)
    return series


def read_numeric_rows(path: Path, required_length: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            values: list[float] = []
            for cell in row:
                try:
                    values.append(float(cell.strip()))
                except ValueError:
                    continue
            if len(values) >= required_length:
                rows.append(values[:required_length])
    if not rows:
        raise ValueError(
            f"No CSV row in {path} contains at least {required_length} numeric values."
        )
    return rows


def fit_points(
    values: Sequence[float], x: float, y: float, w: float, h: float
) -> list[tuple[float, float]]:
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 1e-9)
    pad = 0.14 * h
    usable_h = h - 2 * pad
    points = []
    for i, value in enumerate(values):
        px = x + (i / max(len(values) - 1, 1)) * w
        py = y + h - pad - ((value - lo) / span) * usable_h
        points.append((px, py))
    return points


def curve_card(
    scene: Scene,
    x: float,
    y: float,
    w: float,
    h: float,
    values: Sequence[float],
    history_length: int,
    *,
    solid_color: str = TEAL,
    future_color: str = TEAL,
    fill: str = WHITE,
    border: str = TEAL_PALE,
    show_boundary: bool = True,
    label: str | None = None,
) -> None:
    scene.rect(x, y, w, h, fill=fill, stroke=border, sw=1.0, radius=5)
    inner_x, inner_y = x + 10, y + 7
    inner_w, inner_h = w - 20, h - 14
    points = fit_points(values, inner_x, inner_y, inner_w, inner_h)
    split = max(2, min(history_length, len(points) - 1))
    scene.polyline(points[:split], color=solid_color, sw=1.9)
    scene.polyline(
        points[split - 1 :],
        color=future_color,
        sw=1.7,
        dash="5,4",
        alpha=0.58,
    )
    if show_boundary:
        bx = points[split - 1][0]
        scene.line(
            bx,
            y + 5,
            bx,
            y + h - 5,
            color=GRID,
            sw=0.9,
            dash="2,3",
        )
    if label:
        scene.text(x - 10, y + h / 2, label, size=12, color=MUTED, anchor="right")


def build_single_candidate(values: Sequence[float], history_length: int) -> Scene:
    scene = Scene(540, 185)
    scene.text(270, 20, "Historical analog and candidate continuation", size=16, bold=True)
    curve_card(scene, 35, 48, 470, 95, values, history_length)
    split_x = 35 + 10 + (history_length - 1) / (len(values) - 1) * (470 - 20)
    scene.text((35 + split_x) / 2, 162, "matched history", size=11, color=MUTED)
    scene.text((split_x + 505) / 2, 162, "candidate future", size=11, color=MUTED)
    return scene


def build_input_history_tensor(
    curves: Sequence[Sequence[float]], history_length: int
) -> Scene:
    """Draw the multi-sensor query tensor used at the left of Hist2Res."""
    scene = Scene(460, 370)
    scene.text(230, 26, "Input History  H_t", size=19, bold=True)

    x, y, w, h = 88, 62, 300, 235
    scene.rect(x, y, w, h, fill=WHITE, stroke=GRID, sw=1.1, radius=3)

    row_centers = [105, 166, 254]
    row_labels = ["1", "2", "N"]
    selected = [curves[0], curves[1 % len(curves)], curves[-1]]
    for row_index, (cy, label, values) in enumerate(zip(row_centers, row_labels, selected)):
        if row_index < 2:
            scene.line(x + 12, cy + 29, x + w - 12, cy + 29, color=GRID, sw=0.8)
        history = list(values[:history_length])
        points = fit_points(history, x + 22, cy - 23, w - 44, 46)
        scene.polyline(points, color=TEAL, sw=1.65)
        scene.text(x - 18, cy, label, size=13, color=MUTED, anchor="right")

    for dot_y in (197, 211, 225):
        scene.text(x - 18, dot_y, ".", size=15, color=MUTED, anchor="right")
    scene.text(x + 10, y + h + 27, "1", size=13, color=MUTED, anchor="left")
    scene.text(x + w - 10, y + h + 27, "T", size=13, color=MUTED, anchor="right")
    scene.text(230, 348, "N sensors × T observed steps", size=11.5, color=MUTED)
    return scene


def build_topk_panel(
    curves: Sequence[Sequence[float]], history_length: int, topk: int
) -> Scene:
    scene = Scene(580, 720)
    scene.text(290, 25, f"Top-K candidate futures  (K={topk})", size=18, bold=True)
    scene.text(300, 51, "solid: matched history     dashed: future continuation", size=11, color=MUTED)
    row_h = 68
    start_y = 72
    for i in range(topk):
        values = curves[i % len(curves)]
        curve_card(
            scene,
            60,
            start_y + i * (row_h + 8),
            485,
            row_h,
            values,
            history_length,
            label=str(i + 1),
        )
    return scene


def build_memory_stack(curves: Sequence[Sequence[float]]) -> Scene:
    scene = Scene(520, 450)
    scene.text(260, 24, "Training-only historical memory", size=18, bold=True)
    cards = min(6, len(curves))
    base_x, base_y = 65, 82
    for depth in reversed(range(cards)):
        x = base_x + depth * 16
        y = base_y - depth * 10
        scene.rect(x, y, 360, 270, fill=TEAL_LIGHT, stroke=TEAL_PALE, sw=1.1, radius=7)
        card_values = curves[depth]
        for row in range(3):
            shifted = card_values[row:] + card_values[:row]
            points = fit_points(shifted, x + 25, y + 36 + row * 70, 310, 48)
            scene.polyline(points, color=TEAL, sw=1.35, alpha=0.92)
            if row < 2:
                scene.line(x + 18, y + 99 + row * 70, x + 342, y + 99 + row * 70, color=GRID, sw=0.8)
    scene.text(260, 410, "schematic training windows", size=11, color=MUTED)
    return scene


def build_final_forecast(curves: Sequence[Sequence[float]]) -> Scene:
    scene = Scene(500, 420)
    scene.text(250, 24, "Multi-sensor forecast", size=18, bold=True)
    selected = curves[:4]
    for i, values in enumerate(selected):
        y = 70 + i * 76
        scene.text(38, y + 30, str(i + 1), size=12, color=MUTED)
        scene.rect(58, y, 405, 60, fill=WHITE, stroke=GRID, sw=0.8, radius=3)
        points = fit_points(values[-12:], 72, y + 8, 375, 44)
        scene.polyline(points, color=BLUE, sw=2.0)
        # A nearby dashed parent curve makes the refined forecast visually legible.
        parent = [v + 0.10 * math.sin(j * 1.3 + i) for j, v in enumerate(values[-12:])]
        parent_points = fit_points(parent, 72, y + 8, 375, 44)
        scene.polyline(parent_points, color=BLUE, sw=1.35, dash="5,4", alpha=0.45)
    scene.text(260, 390, "H = 12", size=13, bold=True)
    return scene


def build_overview(
    curves: Sequence[Sequence[float]], history_length: int, topk: int
) -> Scene:
    scene = Scene(1500, 760)
    scene.text(750, 30, "Reusable Hist2Res curve components", size=23, bold=True)

    # Query card.
    scene.text(160, 84, "Query", size=16, bold=True)
    curve_card(
        scene,
        35,
        110,
        250,
        105,
        curves[0],
        history_length,
        solid_color=TEAL,
        future_color=TEAL,
    )

    # Training memory stack.
    scene.text(165, 285, "Training-only Memory", size=16, bold=True, color=TEAL)
    for depth in reversed(range(5)):
        x = 40 + depth * 12
        y = 325 - depth * 8
        scene.rect(x, y, 255, 270, fill=TEAL_LIGHT, stroke=TEAL_PALE, sw=1.0, radius=6)
        card_values = curves[(depth + 1) % len(curves)]
        for row in range(3):
            shifted = card_values[row:] + card_values[:row]
            points = fit_points(shifted, x + 18, y + 34 + row * 72, 218, 47)
            scene.polyline(points, color=TEAL, sw=1.2)
            if row < 2:
                scene.line(x + 14, y + 101 + row * 72, x + 241, y + 101 + row * 72, color=GRID, sw=0.7)

    # Top-K cards.
    scene.text(690, 84, f"Top-K analogs  (K={topk})", size=17, bold=True)
    row_h = 62
    for i in range(topk):
        curve_card(
            scene,
            405,
            112 + i * 74,
            560,
            row_h,
            curves[i % len(curves)],
            history_length,
            label=str(i + 1),
        )

    # Weighted candidate futures and final forecast.
    scene.text(1220, 84, "Reranked candidates", size=17, bold=True)
    for i in range(3):
        values = curves[(i + 2) % len(curves)]
        curve_card(
            scene,
            1080,
            120 + i * 108,
            355,
            78,
            values,
            history_length,
            label=f"pi_{i + 1}",
        )
    scene.text(1250, 485, "Final Forecast", size=17, bold=True, color=BLUE)
    for i in range(3):
        values = curves[(i + 5) % len(curves)][-12:]
        y = 520 + i * 66
        scene.rect(1085, y, 350, 50, fill=BLUE_LIGHT, stroke=GRID, sw=0.8, radius=4)
        points = fit_points(values, 1100, y + 6, 320, 38)
        scene.polyline(points, color=BLUE, sw=1.9)

    scene.text(750, 738, "Solid curves denote observed/matched history; dashed curves denote candidate continuation.", size=12, color=MUTED)
    return scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "hist2res_curve_components",
        help="Directory for SVG, PDF, and PNG outputs.",
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV with one numeric series per row.")
    parser.add_argument("--history-length", type=int, default=12)
    parser.add_argument("--future-length", type=int, default=12)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.history_length < 2 or args.future_length < 1:
        raise ValueError("history-length must be >= 2 and future-length must be >= 1")
    if args.topk < 1:
        raise ValueError("topk must be >= 1")

    total_length = args.history_length + args.future_length
    if args.csv:
        curves = read_numeric_rows(args.csv, total_length)
    else:
        curves = schematic_series(max(args.topk + 6, 16), total_length, args.seed)

    created: list[Path] = []
    created += save_scene(
        build_input_history_tensor(curves, args.history_length),
        args.out_dir,
        "input_history_tensor",
        dpi=args.dpi,
    )
    created += save_scene(
        build_single_candidate(curves[0], args.history_length),
        args.out_dir,
        "candidate_history_future",
        dpi=args.dpi,
    )
    created += save_scene(
        build_topk_panel(curves, args.history_length, args.topk),
        args.out_dir,
        "topk_candidate_curves",
        dpi=args.dpi,
    )
    created += save_scene(
        build_memory_stack(curves),
        args.out_dir,
        "training_memory_stack",
        dpi=args.dpi,
    )
    created += save_scene(
        build_final_forecast(curves),
        args.out_dir,
        "multi_sensor_forecast",
        dpi=args.dpi,
    )
    created += save_scene(
        build_overview(curves, args.history_length, args.topk),
        args.out_dir,
        "hist2res_curve_components_overview",
        dpi=args.dpi,
        transparent_svg=False,
    )

    source = str(args.csv) if args.csv else f"deterministic schematic curves (seed={args.seed})"
    print(f"Source: {source}")
    if pdfcanvas is None:
        print("Note: ReportLab is unavailable in this Python environment; generated SVG files only.")
    elif not any(path.suffix.lower() == ".png" for path in created):
        print("Note: pdftoppm is unavailable; generated SVG and PDF files only.")
    print("Generated:")
    for path in created:
        print(path.resolve())


if __name__ == "__main__":
    main()
