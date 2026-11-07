from __future__ import annotations

import csv
import html
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"
FONT_REG = Path(r"C:\Windows\Fonts\times.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")

INK = "#172B4D"
MUTED = "#52657A"
GRID = "#D9E1EA"
BLUE = "#2F6FB0"
BLUE_FILL = "#E8F1FB"
ORANGE = "#D97706"
ORANGE_FILL = "#FFF2DF"
PURPLE = "#7B4FA3"
PURPLE_FILL = "#F2EAF9"
GREEN = "#23864A"
GREEN_FILL = "#E9F6ED"
RED = "#B8403A"
WHITE = "#FFFFFF"
BAR_BLUE = "#6587AC"
BAR_SAND = "#C49A70"
BAR_SAGE = "#819E82"


def _rgb(hex_color: str):
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))


class Scene:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.ops = []

    def rect(self, x, y, w, h, fill=WHITE, stroke=INK, sw=1.2, radius=8, alpha=1.0):
        self.ops.append(("rect", x, y, w, h, fill, stroke, sw, radius, alpha))

    def line(self, x1, y1, x2, y2, color=INK, sw=1.5, dash=None):
        self.ops.append(("line", x1, y1, x2, y2, color, sw, dash))

    def polyline(self, points, color=INK, sw=1.5, dash=None, fill="none", alpha=1.0):
        self.ops.append(("polyline", list(points), color, sw, dash, fill, alpha))

    def polygon(self, points, fill, stroke="none", sw=0, alpha=1.0):
        self.ops.append(("polygon", list(points), fill, stroke, sw, alpha))

    def circle(self, x, y, r, fill=WHITE, stroke=INK, sw=1.2):
        self.ops.append(("circle", x, y, r, fill, stroke, sw))

    def text(self, x, y, value, size=14, color=INK, bold=False, anchor="middle", angle=0):
        self.ops.append(("text", x, y, str(value), size, color, bold, anchor, angle))

    def arrow(self, x1, y1, x2, y2, color=INK, sw=1.8, dash=None, head=8):
        self.line(x1, y1, x2, y2, color, sw, dash)
        angle = math.atan2(y2 - y1, x2 - x1)
        p1 = (x2, y2)
        p2 = (
            x2 - head * math.cos(angle) + head * 0.55 * math.sin(angle),
            y2 - head * math.sin(angle) - head * 0.55 * math.cos(angle),
        )
        p3 = (
            x2 - head * math.cos(angle) - head * 0.55 * math.sin(angle),
            y2 - head * math.sin(angle) + head * 0.55 * math.cos(angle),
        )
        self.polygon([p1, p2, p3], color)

    def route_arrow(self, points, color=INK, sw=1.8, dash=None, head=8):
        for a, b in zip(points[:-2], points[1:-1]):
            self.line(a[0], a[1], b[0], b[1], color, sw, dash)
        self.arrow(points[-2][0], points[-2][1], points[-1][0], points[-1][1], color, sw, dash, head)


def _svg(scene: Scene, path: Path):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{scene.width}" height="{scene.height}" '
        f'viewBox="0 0 {scene.width} {scene.height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
    ]
    for op in scene.ops:
        kind = op[0]
        if kind == "rect":
            _, x, y, w, h, fill, stroke, sw, radius, alpha = op
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'rx="{radius}" fill="{fill}" fill-opacity="{alpha}" stroke="{stroke}" stroke-width="{sw}"/>'
            )
        elif kind == "line":
            _, x1, y1, x2, y2, color, sw, dash = op
            d = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="{color}" stroke-width="{sw}"{d}/>'
            )
        elif kind == "polyline":
            _, pts, color, sw, dash, fill, alpha = op
            d = f' stroke-dasharray="{dash}"' if dash else ""
            pp = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            parts.append(
                f'<polyline points="{pp}" fill="{fill}" fill-opacity="{alpha}" '
                f'stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" '
                f'stroke-linecap="round"{d}/>'
            )
        elif kind == "polygon":
            _, pts, fill, stroke, sw, alpha = op
            pp = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            parts.append(
                f'<polygon points="{pp}" fill="{fill}" fill-opacity="{alpha}" '
                f'stroke="{stroke}" stroke-width="{sw}"/>'
            )
        elif kind == "circle":
            _, x, y, r, fill, stroke, sw = op
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="{sw}"/>'
            )
        elif kind == "text":
            _, x, y, value, size, color, bold, anchor, angle = op
            weight = "700" if bold else "400"
            svg_anchor = {"left": "start", "middle": "middle", "right": "end"}[anchor]
            transform = f' transform="rotate({angle} {x:.2f} {y:.2f})"' if angle else ""
            parts.append(
                f'<text x="{x:.2f}" y="{y:.2f}" font-family="Times New Roman, serif" '
                f'font-size="{size}" font-weight="{weight}" fill="{color}" '
                f'text-anchor="{svg_anchor}" dominant-baseline="middle"{transform}>{html.escape(value)}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _pdf(scene: Scene, path: Path):
    pdfmetrics.registerFont(TTFont("TimesWin", str(FONT_REG)))
    pdfmetrics.registerFont(TTFont("TimesWin-Bold", str(FONT_BOLD)))
    c = pdfcanvas.Canvas(
        str(path),
        pagesize=(scene.width, scene.height),
        initialFontName="TimesWin",
    )

    def py(y):
        return scene.height - y

    for op in scene.ops:
        kind = op[0]
        if kind == "rect":
            _, x, y, w, h, fill, stroke, sw, radius, alpha = op
            c.saveState()
            if hasattr(c, "setFillAlpha"):
                c.setFillAlpha(alpha)
            c.setFillColorRGB(*_rgb(fill))
            c.setStrokeColorRGB(*_rgb(stroke))
            c.setLineWidth(sw)
            c.roundRect(x, py(y + h), w, h, radius, fill=1, stroke=1)
            c.restoreState()
        elif kind == "line":
            _, x1, y1, x2, y2, color, sw, dash = op
            c.saveState()
            c.setStrokeColorRGB(*_rgb(color))
            c.setLineWidth(sw)
            if dash:
                c.setDash(*[float(v) for v in dash.split(",")])
            c.line(x1, py(y1), x2, py(y2))
            c.restoreState()
        elif kind == "polyline":
            _, pts, color, sw, dash, fill, alpha = op
            c.saveState()
            c.setStrokeColorRGB(*_rgb(color))
            c.setLineWidth(sw)
            if dash:
                c.setDash(*[float(v) for v in dash.split(",")])
            p = c.beginPath()
            p.moveTo(pts[0][0], py(pts[0][1]))
            for x, y in pts[1:]:
                p.lineTo(x, py(y))
            if fill != "none":
                if hasattr(c, "setFillAlpha"):
                    c.setFillAlpha(alpha)
                c.setFillColorRGB(*_rgb(fill))
                c.drawPath(p, fill=1, stroke=1)
            else:
                c.drawPath(p, fill=0, stroke=1)
            c.restoreState()
        elif kind == "polygon":
            _, pts, fill, stroke, sw, alpha = op
            c.saveState()
            if hasattr(c, "setFillAlpha"):
                c.setFillAlpha(alpha)
            c.setFillColorRGB(*_rgb(fill))
            if stroke != "none":
                c.setStrokeColorRGB(*_rgb(stroke))
            c.setLineWidth(sw)
            p = c.beginPath()
            p.moveTo(pts[0][0], py(pts[0][1]))
            for x, y in pts[1:]:
                p.lineTo(x, py(y))
            p.close()
            c.drawPath(p, fill=1, stroke=1 if stroke != "none" else 0)
            c.restoreState()
        elif kind == "circle":
            _, x, y, r, fill, stroke, sw = op
            c.saveState()
            c.setFillColorRGB(*_rgb(fill))
            c.setStrokeColorRGB(*_rgb(stroke))
            c.setLineWidth(sw)
            c.circle(x, py(y), r, fill=1, stroke=1)
            c.restoreState()
        elif kind == "text":
            _, x, y, value, size, color, bold, anchor, angle = op
            c.saveState()
            c.setFillColorRGB(*_rgb(color))
            c.setFont("TimesWin-Bold" if bold else "TimesWin", size)
            if angle:
                c.translate(x, py(y))
                c.rotate(-angle)
                draw_x = 0
                baseline = -size * 0.33
            else:
                draw_x = x
                baseline = py(y) - size * 0.33
            if anchor == "left":
                c.drawString(draw_x, baseline, value)
            elif anchor == "right":
                c.drawRightString(draw_x, baseline, value)
            else:
                c.drawCentredString(draw_x, baseline, value)
            c.restoreState()
    c.showPage()
    c.save()


def save_scene(scene: Scene, stem: str):
    svg_path = OUT / f"{stem}.svg"
    pdf_path = OUT / f"{stem}.pdf"
    _svg(scene, svg_path)
    _pdf(scene, pdf_path)
    subprocess.run(
        ["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", "240", str(pdf_path), str(OUT / stem)],
        check=True,
    )


def box(scene, x, y, w, h, title, subtitle="", fill=BLUE_FILL, stroke=BLUE):
    scene.rect(x, y, w, h, fill, stroke, 1.6, 9)
    if subtitle:
        scene.text(x + w / 2, y + h / 2 - 10, title, 15, INK, True)
        scene.text(x + w / 2, y + h / 2 + 13, subtitle, 11.5, MUTED)
    else:
        scene.text(x + w / 2, y + h / 2, title, 14, INK, True)


def framework_figure():
    s = Scene(1120, 530)
    s.text(560, 28, "Hist2Res: Historical Pattern-to-Residual Adaptation", 24, INK, True)

    box(s, 25, 190, 125, 95, "Input", "history X + time T", BLUE_FILL, BLUE)

    s.rect(185, 65, 525, 155, "#F7FAFD", BLUE, 1.4, 12)
    s.text(205, 84, "Parametric spatio-temporal backbone", 16, BLUE, True, "left")
    names = ["Data\nencoding", "Time-lag\nencoding", "MS-TCN", "MAPA", "12-step\npredictor"]
    x0 = 205
    widths = [85, 92, 82, 76, 92]
    for i, (name, width) in enumerate(zip(names, widths)):
        parts = name.split("\n")
        s.rect(x0, 112, width, 72, BLUE_FILL, BLUE, 1.1, 7)
        for j, part in enumerate(parts):
            s.text(x0 + width / 2, 145 + (j - (len(parts) - 1) / 2) * 17, part, 12.2, INK, j == 0)
        if i < len(names) - 1:
            s.arrow(x0 + width, 148, x0 + width + 18, 148, BLUE, 1.5, head=7)
        x0 += width + 22

    box(s, 745, 105, 120, 85, "Backbone forecast", "Y0", GREEN_FILL, GREEN)

    s.rect(185, 265, 770, 205, "#FFFCF7", ORANGE, 1.4, 12)
    s.text(205, 286, "Leakage-safe historical analog branch", 16, ORANGE, True, "left")
    box(s, 205, 320, 135, 95, "Training memory", "histories + futures", PURPLE_FILL, PURPLE)
    box(s, 370, 320, 135, 95, "Node-wise Top-K", "same time-of-day", ORANGE_FILL, ORANGE)
    box(s, 535, 320, 135, 95, "Candidate reranker", "match + disagreement", ORANGE_FILL, ORANGE)
    box(s, 700, 305, 130, 60, "Rank-2 adapter", "node x horizon", ORANGE_FILL, ORANGE)
    box(s, 700, 385, 130, 60, "Exact-history", "sparse correction", ORANGE_FILL, ORANGE)
    box(s, 855, 320, 80, 95, "Confidence", "calibration", ORANGE_FILL, ORANGE)

    s.arrow(150, 220, 185, 148, BLUE, 1.8, head=8)
    s.route_arrow([(150, 255), (165, 255), (165, 445), (437, 445), (437, 415)], PURPLE, 1.7, "5,4", 8)
    s.text(287, 456, "query history", 10.5, PURPLE)
    s.arrow(710, 148, 745, 148, BLUE, 1.8, head=8)
    s.arrow(340, 368, 370, 368, PURPLE, 1.8, head=8)
    s.arrow(505, 368, 535, 368, ORANGE, 1.8, head=8)
    s.arrow(670, 368, 700, 335, ORANGE, 1.8, head=8)
    s.arrow(765, 365, 765, 385, ORANGE, 1.8, head=8)
    s.arrow(830, 415, 855, 368, ORANGE, 1.8, head=8)

    s.text(600, 303, "candidate futures Ck + observable quality features q", 11.5, MUTED)
    s.route_arrow([(805, 190), (805, 250), (600, 250), (600, 320)], GREEN, 1.4, "6,4", 7)
    s.text(690, 241, "parent-forecast disagreement", 11, GREEN)

    s.circle(1000, 250, 29, WHITE, GREEN, 2)
    s.text(1000, 250, "+", 26, GREEN, True)
    s.route_arrow([(865, 148), (970, 148), (970, 232)], GREEN, 2.0, None, 9)
    s.route_arrow([(935, 368), (970, 368), (970, 268)], ORANGE, 2.0, None, 9)
    box(s, 1040, 205, 65, 90, "Final", "forecast", GREEN_FILL, GREEN)
    s.arrow(1029, 250, 1040, 250, GREEN, 2.0, head=8)

    s.text(560, 500, "All retrieval candidates and their continuations are drawn exclusively from the training split.", 12.5, MUTED)
    save_scene(s, "hist2res_framework")


def nice_ceiling(value):
    magnitude = 10 ** math.floor(math.log10(max(value, 1)))
    scaled = value / magnitude
    step = 1 if scaled <= 4 else 2
    return math.ceil(value / (step * magnitude)) * step * magnitude


def map_point(x, y, x0, y0, w, h, xmin, xmax, ymin, ymax):
    px = x0 + (x - xmin) / (xmax - xmin) * w
    py = y0 + h - (y - ymin) / (ymax - ymin) * h
    return px, py


def axes(scene, x0, y0, w, h, xmin, xmax, ymin, ymax, xticks, yticks, xlabel="", ylabel=""):
    for v in yticks:
        _, py = map_point(xmin, v, x0, y0, w, h, xmin, xmax, ymin, ymax)
        scene.line(x0, py, x0 + w, py, GRID, 0.8)
        scene.text(x0 - 10, py, f"{v:g}", 10.5, MUTED, anchor="right")
    for v, label in xticks:
        px, _ = map_point(v, ymin, x0, y0, w, h, xmin, xmax, ymin, ymax)
        scene.line(px, y0 + h, px, y0 + h + 5, MUTED, 0.8)
        scene.text(px, y0 + h + 18, label, 10.5, MUTED)
    scene.rect(x0, y0, w, h, "#FFFFFF", MUTED, 0.9, 0)
    if xlabel:
        scene.text(x0 + w / 2, y0 + h + 38, xlabel, 11.5, INK)
    if ylabel:
        scene.text(x0 - 56, y0 + h / 2, ylabel, 11.5, INK)


def dataset_figure():
    s = Scene(1050, 760)
    s.text(525, 27, "Traffic-flow patterns across three PeMS datasets", 22, INK, True)
    s.line(715, 55, 765, 55, BLUE, 2.2)
    s.text(775, 55, "Median across sensors", 11.5, INK, anchor="left")
    s.rect(890, 47, 32, 16, BLUE_FILL, BLUE_FILL, 0, 2, 1)
    s.text(930, 55, "Interquartile range", 11.5, INK, anchor="left")

    configs = [("PEMS03", 358), ("PEMS04", 307), ("PEMS07", 883)]
    panel_y = [80, 305, 530]
    for (name, sensors), y0 in zip(configs, panel_y):
        arr = np.load(ROOT / "datasets" / name / "data.npz", allow_pickle=False)["data"][..., 0]
        start = 7 * 288
        week = arr[start : start + 7 * 288]
        med = np.median(week, axis=1).reshape(-1, 3).mean(axis=1)
        q1 = np.percentile(week, 25, axis=1).reshape(-1, 3).mean(axis=1)
        q3 = np.percentile(week, 75, axis=1).reshape(-1, 3).mean(axis=1)
        t = np.arange(len(med)) / 96.0
        ymax = nice_ceiling(float(np.percentile(q3, 99.5)) * 1.08)
        x, y, w, h = 100, y0 + 35, 910, 140
        yticks = np.linspace(0, ymax, 5)
        xticks = [(i, str(i + 1)) for i in range(7)] + [(7, "")]
        axes(s, x, y, w, h, 0, 7, 0, ymax, xticks, yticks, "Day of representative week", "Traffic flow")
        lower = [map_point(xx, yy, x, y, w, h, 0, 7, 0, ymax) for xx, yy in zip(t, q1)]
        upper = [map_point(xx, yy, x, y, w, h, 0, 7, 0, ymax) for xx, yy in zip(t, q3)]
        s.polygon(upper + list(reversed(lower)), BLUE, alpha=0.15)
        line = [map_point(xx, yy, x, y, w, h, 0, 7, 0, ymax) for xx, yy in zip(t, med)]
        s.polyline(line, BLUE, 1.8)
        s.text(100, y0 + 15, f"({chr(97 + panel_y.index(y0))}) {name} ({sensors} sensors)", 14, INK, True, "left")
    save_scene(s, "pems_datasets_visualization")


def parse_loss_log():
    log = ROOT / "EXP_experiments" / "TLAST_improved" / "PEMS04" / \
        "TLAST_improved_PEMS04_12to12_exp07170128_s1" / \
        "TLAST_improved_PEMS04_12to12_exp07170128_s1.log"
    text = log.read_text(encoding="utf-8")
    rows = re.findall(
        r"Epoch:\s*(\d+), Train Loss:\s*([0-9.]+).*?Valid Loss:\s*([0-9.]+)",
        text,
    )
    return np.array([(int(e), float(t), float(v)) for e, t, v in rows], dtype=float)


def export_validation_loss_csv(data):
    output = OUT / "pems04_validation_loss.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "validation_loss"])
        writer.writerows((int(epoch), f"{valid_loss:.4f}")
                         for epoch, _, valid_loss in data)
    return output


def plot_loss_panel(scene, x, y, w, h, data, xmin, xmax, ymin, ymax, title, annotate=False):
    subset = data[(data[:, 0] >= xmin) & (data[:, 0] <= xmax)]
    xtick_values = np.linspace(xmin, xmax, 6)
    xticks = [(float(v), f"{int(round(v))}") for v in xtick_values]
    yticks = np.linspace(ymin, ymax, 6)
    axes(scene, x, y, w, h, xmin, xmax, ymin, ymax, xticks, yticks, "Epoch", "Loss (MAE)")
    train_pts = [map_point(e, t, x, y, w, h, xmin, xmax, ymin, ymax) for e, t, _ in subset]
    valid_pts = [map_point(e, v, x, y, w, h, xmin, xmax, ymin, ymax) for e, _, v in subset]
    scene.polyline(train_pts, BLUE, 2.0)
    scene.polyline(valid_pts, ORANGE, 2.0)
    scene.text(x, y - 18, title, 14, INK, True, "left")
    if annotate:
        best = data[np.argmin(data[:, 2])]
        px, py = map_point(best[0], best[2], x, y, w, h, xmin, xmax, ymin, ymax)
        scene.circle(px, py, 4.5, WHITE, RED, 2)
        scene.route_arrow([(px - 115, py - 28), (px - 22, py - 28), (px - 5, py - 5)], RED, 1.1, None, 6)
        scene.text(px - 120, py - 31, f"Best validation: {best[2]:.4f} (epoch {int(best[0])})", 10.5, RED, True, "left")


def loss_figure():
    data = parse_loss_log()
    s = Scene(1050, 520)
    s.text(525, 27, "PEMS04 training and validation loss", 22, INK, True)
    s.line(390, 58, 435, 58, BLUE, 2.2)
    s.text(445, 58, "Training loss", 11.5, INK, anchor="left")
    s.line(555, 58, 600, 58, ORANGE, 2.2)
    s.text(610, 58, "Validation loss", 11.5, INK, anchor="left")
    plot_loss_panel(s, 110, 110, 395, 330, data, 1, 100, 15, 62, "(a) Full training trajectory")
    zoom = data[data[:, 0] >= 10]
    ymin = math.floor(min(zoom[:, 1].min(), zoom[:, 2].min()) * 2) / 2 - 0.25
    ymax = math.ceil(max(zoom[:, 1].max(), zoom[:, 2].max()) * 2) / 2 + 0.25
    plot_loss_panel(s, 620, 110, 355, 330, data, 10, 100, ymin, ymax, "(b) Convergence detail", True)
    s.text(525, 490, "The checkpoint is selected by the minimum validation MAE.", 12, MUTED)
    save_scene(s, "pems04_loss_curve")


def _series_marker(scene, x, y, shape, color, size=4.0, emphasized=False):
    radius = size * (1.45 if emphasized else 1.0)
    if shape == "circle":
        scene.circle(x, y, radius, WHITE if not emphasized else color, color, 1.4)
    elif shape == "square":
        scene.rect(x - radius, y - radius, 2 * radius, 2 * radius,
                   WHITE if not emphasized else color, color, 1.3, 0)
    else:
        scene.polygon(
            [(x, y - radius * 1.2), (x + radius * 1.05, y + radius),
             (x - radius * 1.05, y + radius)],
            WHITE if not emphasized else color, color, 1.3,
        )


def mae_comparison_figure():
    methods = [
        "HA", "DCRNN", "AGCRN", "STGCN", "MTGNN", "STNorm",
        "PDFormer", "GMAN", "STAEformer", "HyperD", "HARP",
    ]
    results = {
        "PEMS03": [32.62, 15.54, 15.24, 15.83, 14.85, 15.32, 14.94, 16.87, 15.35, 14.69, 14.19],
        "PEMS04": [42.35, 19.63, 19.38, 19.57, 19.17, 18.96, 18.36, 19.14, 18.22, 18.20, 17.63],
        "PEMS07": [49.03, 21.16, 20.57, 21.74, 20.89, 20.50, 19.97, 20.97, 19.14, 19.37, 18.57],
    }
    styles = {
        "PEMS03": ("#2F5A8A", "circle", None),
        "PEMS04": ("#8C5E2A", "square", "7,4"),
        "PEMS07": ("#3E6B45", "triangle", "2,3"),
    }

    s = Scene(1050, 600)
    s.text(525, 30, "MAE comparison on three datasets", 28, INK, True)

    lx, ly, lw, lh = 120, 120, 820, 320
    xmin, xmax, ymin, ymax = -0.25, 10.25, 10, 52
    yticks = [10, 20, 30, 40, 50]
    s.rect(lx, ly, lw, lh, WHITE, "#8C9AA8", 1.3, 0)
    for value in yticks:
        _, py = map_point(xmin, value, lx, ly, lw, lh, xmin, xmax, ymin, ymax)
        s.line(lx, py, lx + lw, py, "#BFCBD8", 1.0, "3,3")
        s.text(lx - 12, py, value, 15, MUTED, True, anchor="right")
    for idx, method in enumerate(methods):
        px, _ = map_point(idx, ymin, lx, ly, lw, lh, xmin, xmax, ymin, ymax)
        s.line(px, ly + lh, px, ly + lh + 5, "#3A4C61", 1.0)
        s.text(px - 2, ly + lh + 20, method, 15,
               "#9B2B25" if method == "HARP" else INK,
               True, "right", -48)
    s.text(lx + lw / 2, ly + lh + 100, "Methods", 18, INK, True)
    s.text(42, ly + lh / 2, "MAE", 18, INK, True, angle=-90)

    legend_x = lx + 230
    for i, name in enumerate(results):
        color, marker, dash = styles[name]
        xx = legend_x + i * 130
        s.line(xx, 90, xx + 30, 90, color, 2.4, dash)
        _series_marker(s, xx + 15, 90, marker, color, 4.4)
        s.text(xx + 40, 90, name, 15, INK, anchor="left")

    for name, values in results.items():
        color, marker, dash = styles[name]
        pts = [map_point(i, value, lx, ly, lw, lh, xmin, xmax, ymin, ymax)
               for i, value in enumerate(values)]
        s.polyline(pts, color, 2.5, dash)
        for px, py in pts:
            _series_marker(s, px, py, marker, color, 4.2)

    save_scene(s, "mae_comparison")


def pems04_validation_loss_figure():
    loss = parse_loss_log()

    s = Scene(1050, 560)
    s.text(525, 30, "PEMS04 validation-loss curve", 28, INK, True)

    rx, ry, rw, rh = 120, 120, 820, 320
    xvals, yvals = loss[:, 0], loss[:, 2]
    vxmin, vxmax = 1, 100
    vymin = math.floor(float(yvals.min())) - 1
    vymax = math.ceil(float(yvals.max())) + 1
    vyticks = np.linspace(vymin, vymax, 6)
    for value in vyticks:
        _, py = map_point(vxmin, value, rx, ry, rw, rh, vxmin, vxmax, vymin, vymax)
        s.line(rx, py, rx + rw, py, "#BFCBD8", 1.0, "3,3")
        s.text(rx - 12, py, f"{value:.0f}", 15, MUTED, True, anchor="right")
    xticks = [1, 20, 40, 60, 80, 100]
    for value in xticks:
        px, _ = map_point(value, vymin, rx, ry, rw, rh, vxmin, vxmax, vymin, vymax)
        s.line(px, ry + rh, px, ry + rh + 5, "#3A4C61", 1.0)
        s.text(px, ry + rh + 20, value, 15, MUTED, True)
    s.rect(rx, ry, rw, rh, WHITE, "#8C9AA8", 1.3, 0)
    s.text(rx + rw / 2, ry + rh + 58, "Training epochs", 18, INK, True)
    s.text(42, ry + rh / 2, "Validation loss", 18, INK, True, angle=-90)

    loss_pts = [map_point(e, v, rx, ry, rw, rh, vxmin, vxmax, vymin, vymax)
                for e, v in zip(xvals, yvals)]
    s.polyline(loss_pts, "#9B2B25", 2.6)
    s.line(rx + 338, 90, rx + 373, 90, "#9B2B25", 2.3)
    s.text(rx + 381, 90, "Validation loss", 15, INK, anchor="left")

    save_scene(s, "pems04_validation_loss_curve")


def main():
    OUT.mkdir(exist_ok=True)
    validation_csv = export_validation_loss_csv(parse_loss_log())
    framework_figure()
    dataset_figure()
    loss_figure()
    mae_comparison_figure()
    pems04_validation_loss_figure()
    print("Generated:")
    for path in sorted(OUT.glob("hist2res_framework.*")) + sorted(OUT.glob("pems_datasets_visualization.*")) + sorted(OUT.glob("pems04_loss_curve.*")):
        print(path.relative_to(ROOT))
    for path in sorted(OUT.glob("mae_comparison.*")) + sorted(OUT.glob("pems04_validation_loss_curve.*")):
        print(path.relative_to(ROOT))
    print(validation_csv.relative_to(ROOT))


if __name__ == "__main__":
    main()
