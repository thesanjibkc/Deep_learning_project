"""Bring the 18-slide Team 5 deck in line with the final report and make it read well in print (2026-11-27).

The slides are printed one per page and pinned up as the poster, so every figure needs its own key.
  slide 1   subtitle: both histories (1 h matched 24 h), no '24 hours' claim
  slide 9   attention diagram: summary token centred, a key for beams, blocks and bars
  slide 11  chart: every model with 1-hour and 24-hour history (report Table 1), colours by history
  slide 13  ablation table: adds residual vs direct Transformer (24 h), as in report Table 2
Run after deck_add_interpretation.py. Idempotent.
"""
import copy
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
DECK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "poster" / "Team5_Sanjib_KC_slides.pptx"
prs = Presentation(DECK)
S = lambda n: prs.slides[n - 1]


def shape(slide, name):
    return next(sh for sh in slide.shapes if sh.name == name)


def set_runs(sh, *texts, para=0):
    for r, t in zip(sh.text_frame.paragraphs[para].runs, texts):
        r.text = t


# ---- slide 1: subtitle
set_runs(shape(S(1), "TextBox 5"),
         "Teaching neural networks what a physics model misses, from flight position and recent space weather")

# ---- slide 9: attention diagram
tok = shape(S(9), "TextBox 6")
tok.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
cap = shape(S(9), "TextBox 79")
cap.left, cap.top, cap.width, cap.height = Inches(0.75), Inches(6.44), Inches(8.40), Inches(0.52)
set_runs(cap, "Thicker beam = more attention.  Block = one hour-token (35 readings); red = storm hour (Kp ≥ 5).  "
              "Bars = Kp index in that hour.")

# ---- slide 11: all models, both histories (18-run means, report Table 1), worst at the top
GRAY, TEAL, LIGHT, ORANGE, RED = "9AA3AF", "24907F", "F2B880", "E8892B", "C8553D"
rows = [("NAIRAS V3 (physics)", 4.044, GRAY), ("Flat MLP · 24 h", 3.909, RED), ("Flat MLP · 1 h", 3.874, RED),
        ("MLP, direct · no history", 3.842, TEAL), ("MLP · no history", 3.740, TEAL),
        ("Transformer, direct · 24 h", 3.712, ORANGE), ("Transformer · 1 h", 3.688, LIGHT),
        ("CNN–MLP · 1 h", 3.686, LIGHT), ("Transformer · 24 h", 3.683, ORANGE),
        ("Transformer, direct · 1 h", 3.683, LIGHT), ("TCN · 24 h", 3.679, ORANGE), ("TCN · 1 h", 3.678, LIGHT),
        ("CNN–MLP · 24 h", 3.667, ORANGE)]
chart = next(sh for sh in S(11).shapes if sh.has_chart).chart
cd = CategoryChartData(number_format="0.000")
cd.categories = [r[0] for r in rows]
cd.add_series("Test RMSE, all flights", [r[1] for r in rows])
chart.replace_data(cd)
ser = chart.plots[0].series[0]
for pt, (_, _, col) in zip(ser.points, rows):
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = RGBColor.from_string(col)
chart.plots[0].gap_width = 35
dl = chart.plots[0].data_labels
dl.number_format, dl.number_format_is_linked = "0.000", False
set_runs(shape(S(11), "TextBox 6"),
         "Test RMSE (µSv/h), mean of 18 runs; shorter is better. Light orange: 1-hour history; orange: 24-hour; "
         "teal: no history; red: Flat MLP (history without time order).")
c6 = shape(S(11), "TextBox 6"); c6.top, c6.height = Inches(6.66), Inches(0.36)
next(sh for sh in S(11).shapes if sh.has_chart).height = Inches(5.05)   # chart ends at 6.60, above the caption

# ---- slide 13: residual vs direct Transformer row (not significant), after the MLP row
tbl = next(sh for sh in S(13).shapes if sh.has_table)
t = tbl.table
label = "Learn the residual instead of the dose (Transformer, 24 h)"
if not any(r.cells[0].text == label for r in t.rows):
    src = next(r for r in t.rows if r.cells[0].text.startswith("Longer window"))   # a non-significant row's style
    new = copy.deepcopy(src._tr)
    t._tbl.insert(t._tbl.index(t.rows[1]._tr) + 1, new)
    for cell, text in zip(t.rows[2].cells, (label, "−0.03", "−0.09", "Not significant")):
        cell.text_frame.paragraphs[0].runs[0].text = text
for r in t.rows:
    r.height = Inches(0.55)
tbl.height = Inches(0.55 * len(t.rows))

prs.save(DECK)
print("polished", DECK)
