"""Create publication-ready vector figures from archived ICDE results.

No training is performed. Convergence smoothing is display-only and uses
the trailing three recorded epochs; raw observations remain visible.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PERFORMANCE_CSV = HERE / "overall_complete_methods.csv"
CONVERGENCE_CSV = ROOT / "paper_results/icde_convergence/convergence_epochs.csv"
DATASETS = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
METRICS = [("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE (%)")]
INK = colors.HexColor("#183047")
MUTED = colors.HexColor("#64758a")
BLUE = colors.HexColor("#2468a0")
ORANGE = colors.HexColor("#c46a27")
GRID = colors.HexColor("#e0e6ec")


def load_performance():
    with PERFORMANCE_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 17
    for row in rows:
        for dataset in DATASETS:
            for metric, _ in METRICS:
                value = float(row[f"{dataset.lower()}_{metric}"])
                assert math.isfinite(value)
    return rows


def to_png(pdf_path: Path):
    pdf = pdfium.PdfDocument(str(pdf_path))
    image = pdf[0].render(scale=2.0).to_pil()
    image.save(pdf_path.with_suffix(".png"))
    pdf.close()


def draw_fit_text(c, value, x, y, max_width, size=10, font="Helvetica"):
    while stringWidth(value, font, size) > max_width and size > 6:
        size -= 0.25
    c.setFont(font, size)
    c.drawString(x, y, value)


def performance_chart(rows, metric, title):
    width, height = 810, 645
    pdf_path = HERE / f"overall_{metric}_heatmap.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(width, height))
    c.setTitle(f"Complete-method comparison: {title}")
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 19)
    c.drawString(42, 598, f"Forecasting performance | {title}")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(42, 578, "All methods have results on all four datasets; lower values are better.")
    c.drawString(42, 564, "Cell shading shows within-dataset rank, not cross-dataset magnitude.")

    left, top, row_h, label_w, col_w = 42, 516, 26, 255, 119
    for j, dataset in enumerate(DATASETS):
        x = left + label_w + j * col_w
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(x + col_w / 2, 533, dataset)
        values = [float(r[f"{dataset.lower()}_{metric}"]) for r in rows]
        ranked = sorted(values)
        for i, row in enumerate(rows):
            y = top - (i + 1) * row_h
            value = float(row[f"{dataset.lower()}_{metric}"])
            rank = ranked.index(value)
            strength = 1 - rank / (len(rows) - 1)
            shade = colors.Color(0.93 - 0.63 * strength, 0.96 - 0.27 * strength, 0.98 - 0.18 * strength)
            c.setFillColor(shade)
            c.rect(x + 2, y + 1, col_w - 4, row_h - 2, stroke=0, fill=1)
            if row["method"] == "DCLD Full":
                c.setStrokeColor(BLUE)
                c.setLineWidth(1.6)
                c.rect(x + 2, y + 1, col_w - 4, row_h - 2, stroke=1, fill=0)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold" if row["method"] == "DCLD Full" else "Helvetica", 10)
            c.drawCentredString(x + col_w / 2, y + 9, f"{value:.2f}")

    for i, row in enumerate(rows):
        y = top - (i + 1) * row_h
        method = row["method"]
        if i == 14:
            c.setStrokeColor(colors.HexColor("#aab9c7"))
            c.setLineWidth(1)
            c.line(left, y + row_h, left + label_w + 4 * col_w, y + row_h)
        c.setFillColor(BLUE if method == "DCLD Full" else INK)
        draw_fit_text(c, method, left + 5, y + 9, label_w - 12, 10,
                      "Helvetica-Bold" if method == "DCLD Full" else "Helvetica")

    c.setStrokeColor(GRID)
    c.line(left, 56, left + label_w + 4 * col_w, 56)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(left, 40, "Reported baselines above; fixed teacher and DCLD variants below the divider.")
    c.showPage()
    c.save()
    to_png(pdf_path)
    print(pdf_path)


def load_convergence():
    grouped = defaultdict(list)
    with CONVERGENCE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] not in ("PEMS04", "PEMS07"):
                continue
            if not row["val_mae"]:
                continue
            grouped[(row["dataset"], row["method"], int(row["seed"]))].append(
                (int(row["epoch"]), float(row["val_mae"]))
            )
    # Supersede the archived PEMS04 student curves with the final
    # strict-cache-alignment runs, keeping the same fixed teacher trace.
    strict_logs = sorted((ROOT / "EXP_experiments/TLAST_improved/PEMS04").glob(
        "TLAST_improved_PEMS04_12to12_exppems04_joint3_lite_r256_strictalign_s*/*.log"
    ))
    assert len(strict_logs) == 3, strict_logs
    for path in strict_logs:
        seed = int(re.search(r"strictalign_s([123])", str(path)).group(1))
        values = {}
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                anchor = re.search(r"Epoch 000 anchor, Valid Loss: [^,]+, Valid MAE: ([0-9.]+)", line)
                measured = re.search(r"Epoch: (\d+), Train Loss: .*?Valid MAE: ([0-9.]+)", line)
                if anchor:
                    values[0] = float(anchor.group(1))
                if measured:
                    values[int(measured.group(1))] = float(measured.group(2))
        assert 0 in values and len(values) > 10, path
        grouped[("PEMS04", "Lite-R256", seed)] = sorted(values.items())
    for series in grouped.values():
        series.sort()
    return grouped


def moving_average(values, window=3):
    return [sum(values[max(0, i - window + 1):i + 1]) /
            len(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


def convergence_chart(grouped):
    width, height = 1000, 665
    pdf_path = HERE / "convergence_all_recorded_epochs.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(width, height))
    c.setTitle("Training convergence: all recorded epochs")
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 19)
    c.drawString(54, 623, "Network training convergence")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(54, 605, "Raw observations remain visible; bold curves are trailing 3-epoch averages.")
    c.drawString(54, 591, "Each panel uses its own labeled MAE scale; no epochs are extrapolated.")

    panel_w, panel_h = 425, 225
    x_positions = [58, 548]
    y_positions = [327, 70]
    for col, dataset in enumerate(("PEMS04", "PEMS07")):
        for row_idx, method in enumerate(("Teacher", "Lite-R256")):
            x0, y0 = x_positions[col], y_positions[row_idx]
            series = {seed: points for (ds, name, seed), points in grouped.items()
                      if ds == dataset and name == method}
            assert series
            xmax = max(max(epoch for epoch, _ in points) for points in series.values())
            # End each axis at the last observed epoch; do not imply that
            # the teacher was trained for the student's full duration.
            x_cap = xmax
            all_values = [value for points in series.values() for _, value in points]
            vmin, vmax = min(all_values), max(all_values)
            pad = max((vmax - vmin) * 0.18, 0.002)
            vmin -= pad
            vmax += pad
            plot_left, plot_bottom = x0 + 48, y0 + 33
            plot_w, plot_h = panel_w - 63, panel_h - 67

            def xp(epoch):
                return plot_left + epoch / x_cap * plot_w

            def yp(value):
                return plot_bottom + (value - vmin) / (vmax - vmin) * plot_h

            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x0, y0 + panel_h - 16, f"{dataset}  |  {method}")
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 8)
            c.drawRightString(x0 + panel_w, y0 + panel_h - 15,
                              f"observed through epoch {xmax}")
            for t in range(5):
                value = vmin + t * (vmax - vmin) / 4
                y = yp(value)
                c.setStrokeColor(GRID)
                c.setLineWidth(0.5)
                c.line(plot_left, y, plot_left + plot_w, y)
                c.setFillColor(MUTED)
                c.setFont("Helvetica", 8)
                c.drawRightString(plot_left - 6, y - 3, f"{value:.3f}")
            tick_step = 2 if x_cap <= 10 else 5
            for epoch in range(0, x_cap + 1, tick_step):
                x = xp(epoch)
                c.setFillColor(MUTED)
                c.setFont("Helvetica", 8)
                c.drawCentredString(x, plot_bottom - 16, str(epoch))
            c.setStrokeColor(MUTED)
            c.setLineWidth(0.8)
            c.line(plot_left, plot_bottom, plot_left + plot_w, plot_bottom)
            c.line(plot_left, plot_bottom, plot_left, plot_bottom + plot_h)

            base = ORANGE if method == "Teacher" else BLUE
            for seed, points in sorted(series.items()):
                c.setStrokeColor(colors.Color(base.red, base.green, base.blue, alpha=0.30))
                c.setLineWidth(0.9)
                path = c.beginPath()
                for i, (epoch, value) in enumerate(points):
                    (path.moveTo if i == 0 else path.lineTo)(xp(epoch), yp(value))
                c.drawPath(path)
                c.setFillColor(base)
                c.circle(xp(points[-1][0]), yp(points[-1][1]), 1.7, fill=1, stroke=0)

            if len(series) == 1:
                common = next(iter(series.values()))
                smooth = moving_average([v for _, v in common])
                epochs = [e for e, _ in common]
            else:
                common_end = min(points[-1][0] for points in series.values())
                epochs = list(range(common_end + 1))
                lookup = {seed: dict(points) for seed, points in series.items()}
                mean_values = [sum(lookup[seed][epoch] for seed in series) / len(series)
                               for epoch in epochs]
                smooth = moving_average(mean_values)
                c.setFillColor(MUTED)
                c.setFont("Helvetica", 8)
                c.drawString(plot_left + 3, plot_bottom + plot_h - 12,
                             f"3-seed mean through epoch {common_end}")

            c.setStrokeColor(base)
            c.setLineWidth(2.5)
            path = c.beginPath()
            for i, (epoch, value) in enumerate(zip(epochs, smooth)):
                (path.moveTo if i == 0 else path.lineTo)(xp(epoch), yp(value))
            c.drawPath(path)
            c.setFillColor(base)
            c.circle(xp(epochs[-1]), yp(smooth[-1]), 2.5, fill=1, stroke=0)

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(55, 36, "Each panel ends at its last recorded epoch; lighter lines show the raw seed traces.")
    c.showPage()
    c.save()
    to_png(pdf_path)
    print(pdf_path)


def main():
    rows = load_performance()
    for metric, title in METRICS:
        performance_chart(rows, metric, title)
    convergence_chart(load_convergence())


if __name__ == "__main__":
    main()
