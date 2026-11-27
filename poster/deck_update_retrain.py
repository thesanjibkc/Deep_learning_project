"""Patch the Team 5 deck after retraining without NAIRAS as a network input (2026-11-27).

Patches the existing, possibly hand-edited deck in place (run after poster/deck_update_results.py):
  slide 8   equation without NAIRAS as input; parameters and training time per model
  slide 9   real attention beams of the new tracking run, best epoch, no NAIRAS into the head
  slide 11  chart values and cards (best model CNN-MLP, 24 h)
  slide 12  cards and notes (figure is replaced by deck_update_results.py)
  slide 13  ablation table
  slide 16  training-dynamics figure and best epoch
  slide 18  conclusion wording
Numbers come from results/ (18-run means) and results/final/significance.csv; idempotent.
"""
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DECK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "poster" / "Team5_Sanjib_KC_slides.pptx"
prs = Presentation(DECK)
S = lambda n: prs.slides[n - 1]


def shape(slide, name):
    return next(sh for sh in slide.shapes if sh.name == name)


def set_runs(sh, *texts, para=0):
    """Set the runs of one paragraph in order, keeping each run's formatting."""
    runs = sh.text_frame.paragraphs[para].runs
    for r, t in zip(runs, texts):
        r.text = t


# ---- slide 8: equation and model cards
set_runs(shape(S(8), "TextBox 6"), "prediction  =  ", "NAIRAS", "  +  ", "network(history, static inputs)")
for name, txt in (("TextBox 15", "14 k params · 2 s"), ("TextBox 19", "56 k params · 2 s"), ("TextBox 23", "7 k params · 3 s"),
                  ("TextBox 27", "10 k params · 8 s"), ("TextBox 31", "9 k params · 29 s")):
    set_runs(shape(S(8), name), txt)
S(8).notes_slide.notes_text_frame.text = (
    "NAIRAS is not a network input: every residual network sees only position, shielding and space weather, and its output is "
    "added to NAIRAS. Direct variants get the same inputs and predict the dose. Final setting chosen on validation only: width 16, "
    "dropout 0.3, weight decay 0.01. Parameters and mean training time per run on a laptop GPU (Apple M5).")

# ---- slide 9: attention beams from the new tracking run
trk = json.load(open(ROOT / "poster/animation/tracking.json"))
best = trk["best_epoch_transformer"]; att = trk["transformer"]["attention_storm_flight"][best]
lo, hi = min(att), max(att)
beams = [sh for sh in S(9).shapes if sh.shape_type == 9]          # 24 connectors, hour 23 h before ... flight
assert len(beams) == 24, len(beams)
beams.sort(key=lambda sh: sh.begin_x if sh.begin_x != sh.end_x else sh.end_x)   # order is creation order anyway
for b, a in zip([sh for sh in S(9).shapes if sh.shape_type == 9], att):
    b.line.width = Pt(0.5 + 7 * max(0.0, (a - lo) / (hi - lo)))
set_runs(shape(S(9), "TextBox 94"), "weighted summary + altitude + shield → correction, added to NAIRAS")
set_runs(shape(S(9), "TextBox 95"), f"Real learned attention, 9 Sept 2017 flight (epoch {best}). No storm labels given.")
S(9).notes_slide.notes_text_frame.text = (
    f"Beam width is the real learned attention at the best epoch ({best}). It peaks 16–18 h before the flight, where Kp reached "
    "8.3. Attention = softmax(QKᵀ/√d)V. NAIRAS is not an input; it is only added to the output.")

# ---- slide 11: overall results chart and cards
R = {"NAIRAS V3": 4.044, "Flat MLP, 24 h": 3.909, "MLP, direct": 3.842, "MLP (no history)": 3.740,
     "Transformer, direct, 24 h": 3.712, "Transformer, 1 h": 3.688, "Transformer, 24 h": 3.683,
     "TCN, 24 h": 3.679, "CNN–MLP, 24 h": 3.667}
chart = next(sh for sh in S(11).shapes if sh.has_chart).chart
cd = CategoryChartData(); cd.categories = list(R); cd.add_series("Test RMSE, all flights", list(R.values()))
chart.replace_data(cd)
set_runs(shape(S(11), "TextBox 8"), "−9.3%")
set_runs(shape(S(11), "TextBox 9"), "error vs NAIRAS for the best model, CNN–MLP with 24 h history")
set_runs(shape(S(11), "TextBox 11"), "−0.06")
set_runs(shape(S(11), "TextBox 12"), "µSv/h vs the no-history MLP (CNN–MLP, 24 h): better in all 6 splits (95% CI −0.09 to −0.03)")
set_runs(shape(S(11), "TextBox 14"), "3.67")
S(11).notes_slide.notes_text_frame.text = (
    "Flat MLP is worst among networks: same history, no time order, more parameters. CNN–MLP, TCN and Transformer with 24 h "
    "are within 0.02 of each other. 1-hour and 24-hour windows give the same average error (difference 0.00, not significant).")

# ---- slide 12: high-dose cards
set_runs(shape(S(12), "TextBox 8"), "5.55")
set_runs(shape(S(12), "TextBox 9"), "µSv/h high-dose error, down from 6.85 for NAIRAS (−19%)")
set_runs(shape(S(12), "TextBox 14"), "+0.01")
set_runs(shape(S(12), "TextBox 15"), "change in overall error vs the MLP (not significant)")
S(12).notes_slide.notes_text_frame.text = (
    "Left three panels: NAIRAS, the best overall model (CNN–MLP, 24 h) and the weighted Transformer against the measurement, "
    "all on unseen flights. Right: the error grows with the true dose for every model. Weighting high-dose errors four times "
    "(weighted Huber) cuts the high-dose error 19% vs NAIRAS (−0.82 vs the MLP, 95% CI −0.92 to −0.71) while matching the MLP overall "
    "(+0.01, n.s.; +0.08 vs the unweighted Transformer), yet all models still underpredict the highest doses. XGBoost 2025 did not report high-dose error.")

# ---- slide 13: ablation table (text only; formatting kept)
tbl = next(sh for sh in S(13).shapes if sh.has_table).table
rows = [("Learn the residual instead of the dose (MLP, same inputs)", "−0.07", "−0.29", "Helps"),
        ("Add 24 h history (Transformer vs MLP)", "−0.06", "0.00", "Helps overall"),
        ("Add 24 h history (CNN–MLP vs MLP)", "−0.06", "−0.18", "Helps"),
        ("Read history with structure (Transformer vs Flat MLP)", "−0.25", "−0.40", "Helps most"),
        ("Longer window (24 h vs 1 h Transformer)", "0.00", "+0.04", "No difference"),
        ("Weight high doses ×4 (vs unweighted)", "+0.08", "−0.64", "Trade-off")]
for i, r in enumerate(rows, start=1):
    for j, v in enumerate(r):
        tbl.cell(i, j).text_frame.paragraphs[0].runs[0].text = v

# ---- slide 16 (was 14): training dynamics
s14 = S(16); old = shape(s14, "Picture 5")
x, y, w, h = old.left, old.top, old.width, old.height
old._element.getparent().remove(old._element)
pic = s14.shapes.add_picture(str(ROOT / "figures/results/T1_training_evolution.png"), x, y, w, h); pic.name = "Picture 5"
set_runs(shape(s14, "TextBox 6"), f"By epoch {best}: ", "it peaks 16–18 h before the flight, exactly where Kp hit 8.3.", para=1)

# ---- slide 18 (was 16): conclusion wording
set_runs(shape(S(18), "TextBox 7"), "With the same inputs, learning NAIRAS's correction beat learning the dose: clearly for the MLP, narrowly for the Transformer.")
set_runs(shape(S(18), "TextBox 13"), "Weighting high doses cut their error by 19% vs NAIRAS while matching the MLP overall.")

prs.save(DECK)
print("patched", DECK, "best epoch", best)
