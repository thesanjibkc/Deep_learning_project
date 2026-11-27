"""Patch claim wording in the Team 5 deck to match the final report (2026-11-27).

Text and layout, idempotent, keeps each run's formatting. Run after deck_update_results.py and deck_update_retrain.py.
  slide 1   title: temporal networks (the best overall model is the CNN-MLP, not the Transformer)
  slide 9   layout: move the caption and Kp label off the Kp bars
  layout    slides 8, 11, 16, 17, 18 (18-slide numbering): remove text/figure overlaps (see 'overlap scan' below); slide 16 'overfits'
  slide 12  weighted Transformer: matches the MLP overall (ensembles, +0.01 n.s.) but +0.08 vs unweighted Transformer
  slide 18  residual vs direct: clear for the MLP, narrow for the Transformer; Flat MLP overfit
"""
import sys
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
DECK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "poster" / "Team5_Sanjib_KC_slides.pptx"
prs = Presentation(DECK)
S = lambda n: prs.slides[n - 1]


def shape(slide, name):
    return next(sh for sh in slide.shapes if sh.name == name)


def set_runs(sh, *texts, para=0):
    for r, t in zip(sh.text_frame.paragraphs[para].runs, texts):
        r.text = t


set_runs(shape(S(1), "Title"), "Physics-Guided Temporal Networks for Aviation Dose")

set_runs(shape(S(12), "TextBox 15"), "change in overall error vs the MLP (not significant)")
S(12).notes_slide.notes_text_frame.text = (
    "Left three panels: NAIRAS, the best overall model (CNN–MLP, 24 h) and the weighted Transformer against the measurement, "
    "all on unseen flights. Right: the error grows with the true dose for every model. Weighting high-dose errors four times "
    "(weighted Huber) cuts the high-dose error 19% vs NAIRAS (−0.82 vs the MLP, 95% CI −0.92 to −0.71) while matching the "
    "MLP overall (+0.01, not significant); against the unweighted Transformer it costs +0.08 overall. All models still "
    "underpredict the highest doses. XGBoost 2025 did not report high-dose error.")

set_runs(shape(S(18), "TextBox 7"),
         "With the same inputs, learning NAIRAS's correction beat learning the dose: clearly for the MLP, narrowly for the Transformer.")
set_runs(shape(S(18), "TextBox 10"), "Small, regularized temporal networks beat a no-history MLP; a flat, larger network overfit.")
set_runs(shape(S(18), "TextBox 13"), "Weighting high doses cut their error by 19% vs NAIRAS while matching the MLP overall.")

# slide 9 layout: caption and Kp label overlapped the Kp bars (bars span y 5.32-6.15 in)
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN
cap = shape(S(9), "TextBox 79"); cap.left, cap.top = Inches(0.75), Inches(6.52)
kp = shape(S(9), "TextBox 82"); kp.left, kp.top, kp.width = Inches(2.30), Inches(6.20), Inches(1.20)
set_runs(kp, "▲ Kp 8.3"); kp.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

# layout fixes found by an overlap scan (positions in inches; idempotent)
def place(slide, name, x, y, w=None, h=None):
    sh = shape(S(slide), name); sh.left, sh.top = Inches(x), Inches(y)
    if w is not None: sh.width = Inches(w)
    if h is not None: sh.height = Inches(h)
place(8, "TextBox 10", 0.72, 6.16, h=0.28)             # NAIRAS card: "physics model" collided with "no training"
place(8, "TextBox 11", 0.72, 6.40, h=0.28)
place(11, "Chart 5", 0.50, 1.55, h=5.15)              # chart bottom ran under its caption
place(16, "Picture 5", 0.79, 1.50, 7.42, 5.18)        # opaque figure covered its caption (aspect kept)
place(16, "TextBox 9", 0.60, 6.70)
place(17, "Picture 5", 2.65, 1.50, 8.04, 3.78)        # opaque figure covered the tops of the three cards (aspect kept)
for card, text in (("Rounded Rectangle 11", "TextBox 13"), ("Rounded Rectangle 14", "TextBox 16")):
    place(18, card, shape(S(18), card).left / 914400, 3.60, h=1.38)   # cards end above the "Code on GitHub" label
    place(18, text, shape(S(18), text).left / 914400, 4.32, h=0.62)
place(16, "TextBox 6", 8.85, 1.80, h=3.40)            # side text box frame ended under the two QR images
set_runs(shape(S(16), "TextBox 6"), "Flat MLP: ", "even weights; overfits after epoch 4.", para=2)

prs.save(DECK)
print("patched", DECK)
