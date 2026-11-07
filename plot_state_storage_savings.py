"""Draw the retrieval-state savings chart at its final single-column size."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output" / "pdf" / "state-storage-savings-single-column.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Saved vs. NPZ percentages from Table~\ref{tab:encoding_analysis}.
DATA = [("PEMS03", 39.66), ("PEMS04", 51.08),
        ("PEMS07", 77.27), ("PEMS08", 54.46)]

FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("TimesNewRoman", str(FONT_DIR / "times.ttf")))
pdfmetrics.registerFont(TTFont("TimesNewRoman-Bold", str(FONT_DIR / "timesbd.ttf")))

GREEN = colors.HexColor("#718C79")
BLUE = colors.HexColor("#69849B")
INK = colors.HexColor("#3D4850")
CONNECTOR = colors.HexColor("#C9D3D9")
GRID = colors.HexColor("#DFE4E7")


def main() -> None:
    page_w, page_h = 250, 165  # 3.47 in: true IEEE single-column artwork.
    x0, x1 = 62, 235
    ys = [114, 91, 68, 45]
    scale_x = lambda value: x0 + (x1 - x0) * value / 86.0

    pdf = canvas.Canvas(str(OUTPUT), pagesize=(page_w, page_h), pageCompression=1)
    pdf.setTitle("Retrieval-state storage savings")
    pdf.setAuthor("Experiment visualization")

    # At final physical size, legend and axes share an 8 pt base type scale.
    legend_font = 8.0
    legend = [("Original NPZ (0% saved)", BLUE, True),
              ("Final state", GREEN, False)]
    widths = [pdfmetrics.stringWidth(label, "TimesNewRoman", legend_font)
              for label, _color, _open in legend]
    gap, swatch_space = 13, 12
    total = sum(widths) + 2 * swatch_space + gap
    start = (page_w - total) / 2
    for (label, color, is_open), label_w in zip(legend, widths):
        pdf.setStrokeColor(color)
        pdf.setFillColor(colors.white if is_open else color)
        pdf.setLineWidth(1.0)
        pdf.circle(start + 3.5, 148, 3.1, stroke=1, fill=1)
        pdf.setFillColor(INK)
        pdf.setFont("TimesNewRoman", legend_font)
        pdf.drawString(start + swatch_space, 145.3, label)
        start += swatch_space + label_w + gap

    for tick in (0, 20, 40, 60, 80):
        x = scale_x(tick)
        pdf.setStrokeColor(GRID)
        pdf.setLineWidth(0.45)
        pdf.setDash(1.5, 2)
        pdf.line(x, 37, x, 127)
        pdf.setDash()
        pdf.setStrokeColor(INK)
        pdf.line(x, 34, x, 31)
        pdf.setFillColor(INK)
        pdf.setFont("TimesNewRoman", 7.8)
        pdf.drawCentredString(x, 20, str(tick))

    pdf.setStrokeColor(INK)
    pdf.setLineWidth(0.65)
    pdf.line(53, 34, 239, 34)

    for (dataset, saved), y in zip(DATA, ys):
        final_x = scale_x(saved)
        pdf.setFont("TimesNewRoman-Bold", 8.0)
        pdf.setFillColor(colors.black)
        pdf.drawRightString(47, y - 2.8, dataset)
        pdf.setStrokeColor(CONNECTOR)
        pdf.setLineWidth(1.8)
        pdf.line(x0, y, final_x, y)
        pdf.setStrokeColor(BLUE)
        pdf.setFillColor(colors.white)
        pdf.setLineWidth(1.1)
        pdf.circle(x0, y, 2.9, stroke=1, fill=1)
        pdf.setFillColor(GREEN)
        pdf.circle(final_x, y, 3.2, stroke=0, fill=1)

    pdf.setFillColor(colors.black)
    pdf.setFont("TimesNewRoman-Bold", 8.2)
    pdf.drawCentredString(page_w / 2, 5.5,
                          "Storage saved vs. original NPZ (%)")
    pdf.showPage()
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
