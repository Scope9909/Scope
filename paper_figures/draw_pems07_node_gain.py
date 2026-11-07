"""Draw a paper-ready PEMS07 per-node retrieval-gain survival chart.

The input is the frozen, held-out validation result. Positive gain means
base-prediction MAE minus retrieval-prediction MAE is positive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "pems07_exact_node_gain_validation.json"
OUTPUT = (Path(sys.argv[1]) if len(sys.argv) > 1 else
          HERE.parent / "output" / "pdf" / "pems07-node-gain-single-column.pdf")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("TimesNewRoman", str(FONT_DIR / "times.ttf")))
pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(FONT_DIR / "timesbd.ttf")))

GREEN = colors.HexColor("#147D66")
GRAY = colors.HexColor("#75828D")
INK = colors.HexColor("#27313B")
GRID = colors.HexColor("#E2E7EB")


def survival(values: list[float], threshold: float) -> float:
    return 100.0 * sum(value > threshold for value in values) / len(values)


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert data["test_evaluated"] is False
    gains = data["heldout_gain_by_node"]
    selected_ids = set(data["selected_node_ids"])
    assert len(gains) == 883 and len(selected_ids) == 121
    groups = [
        ("SRM-selected nodes (121)", [gains[i] for i in sorted(selected_ids)], GREEN, []),
        ("Unselected nodes (762)", [g for i, g in enumerate(gains) if i not in selected_ids], GRAY, [7, 4]),
    ]

    page_w, page_h = 250, 165  # Final artwork width: 3.47 in (one column).
    left, right, bottom, top = 39, 230, 39, 123
    width, height = right - left, top - bottom
    x_value = lambda threshold: left + width * threshold / 8.0
    y_value = lambda percent: bottom + height * percent / 100.0

    pdf = canvas.Canvas(str(OUTPUT), pagesize=(page_w, page_h), pageCompression=1)
    pdf.setTitle("PEMS07 node-wise retrieval gain")
    pdf.setAuthor("Experiment visualization")

    # Keep the same 8 pt Times New Roman scale as the storage figure.
    legend_font = 8.0
    pdf.setFont("TimesNewRoman", legend_font)
    legend_widths = [pdfmetrics.stringWidth(group[0], "TimesNewRoman", legend_font)
                     for group in groups]
    total_legend_width = sum(legend_widths) + 2 * 16 + 12
    legend_x = (page_w - total_legend_width) / 2
    for (label, _values, color, dash), label_width in zip(groups, legend_widths):
        pdf.setStrokeColor(color)
        pdf.setLineWidth(1.5)
        pdf.setDash(dash)
        pdf.line(legend_x, 149, legend_x + 12, 149)
        pdf.setDash()
        pdf.setFillColor(INK)
        pdf.drawString(legend_x + 16, 146.5, label)
        legend_x += 16 + label_width + 12

    pdf.setFont("TimesNewRoman", 7.8)
    for tick in range(0, 101, 20):
        y = y_value(tick)
        if tick > 0:
            pdf.setStrokeColor(GRID)
            pdf.setLineWidth(0.45)
            pdf.line(left, y, right, y)
        pdf.setFillColor(INK)
        pdf.drawRightString(left - 5, y - 2.6, str(tick))

    for tick in (0, 2, 4, 6, 8):
        x = x_value(tick)
        pdf.setStrokeColor(INK)
        pdf.setLineWidth(0.55)
        pdf.line(x, bottom, x, bottom - 3)
        pdf.setFillColor(INK)
        pdf.drawCentredString(x, bottom - 13, str(tick))

    pdf.setStrokeColor(INK)
    pdf.setLineWidth(0.7)
    pdf.line(left, bottom, right + 3, bottom)
    pdf.line(left, bottom, left, top + 3)
    pdf.setFillColor(INK)
    x_arrow = pdf.beginPath()
    x_arrow.moveTo(right + 8, bottom)
    x_arrow.lineTo(right + 2, bottom + 2.5)
    x_arrow.lineTo(right + 2, bottom - 2.5)
    x_arrow.close()
    pdf.drawPath(x_arrow, stroke=0, fill=1)
    y_arrow = pdf.beginPath()
    y_arrow.moveTo(left, top + 8)
    y_arrow.lineTo(left - 2.5, top + 2)
    y_arrow.lineTo(left + 2.5, top + 2)
    y_arrow.close()
    pdf.drawPath(y_arrow, stroke=0, fill=1)

    for _label, values, color, dash in groups:
        path = pdf.beginPath()
        for j in range(481):
            threshold = 8.0 * j / 480.0
            x, y = x_value(threshold), y_value(survival(values, threshold))
            if j == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        pdf.setStrokeColor(color)
        pdf.setLineWidth(1.5)
        pdf.setDash(dash)
        pdf.drawPath(path, stroke=1, fill=0)
        pdf.setDash()

    pdf.setFillColor(INK)
    pdf.setFont("TimesNewRoman-Bold", 8.2)
    pdf.drawCentredString((left + right) / 2, 5.5,
                          "Minimum per-node MAE reduction from retrieval")
    pdf.saveState()
    pdf.translate(9, (bottom + top) / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Nodes exceeding threshold (%)")
    pdf.restoreState()

    pdf.showPage()
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
