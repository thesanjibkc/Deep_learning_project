"""Bring slides 12, 15 and 16 in line with the final report (patches the existing, possibly
hand-edited deck instead of rebuilding it with build_deck.py).

  slide 12  replace the high-dose bar chart with the NAIRAS-vs-best-models figure
            (figures/results/R4_best_vs_nairas.png); stat cards move to a bottom row,
            middle card now shows the underprediction rate above 15 uSv/h
  slide 15  refreshed 'why some error remains' figure (six-split averages) and cards
  slide 16  conclusion card 4 now states what drives the correction and whose history matters

Usage: python3 poster/deck_update_results.py [path/to/deck.pptx]
Idempotent: the slide-12 picture is named UPD_best; re-running replaces it.
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
DECK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "poster" / "Team5_Sanjib_KC_slides.pptx"


def shape(slide, name):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    return None


def set_text(sh, text):
    """Replace the text of the first run, keep its formatting, drop the other runs."""
    p = sh.text_frame.paragraphs[0]
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    for extra in sh.text_frame.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def move(sh, x=None, y=None):
    if x is not None:
        sh.left = Inches(x)
    if y is not None:
        sh.top = Inches(y)


def replace_picture(slide, old, img, name, x, y, w, h=None):
    if old is not None:
        old._element.getparent().remove(old._element)
    pic = slide.shapes.add_picture(str(img), Inches(x), Inches(y), Inches(w), Inches(h) if h else None)
    pic.name = name
    return pic


prs = Presentation(DECK)
s12, s15, s16 = prs.slides[11], prs.slides[14], prs.slides[15]

# ---- slide 12: figure on top, three cards in a bottom row
for nm in ("Chart 5", "TextBox 6"):                        # old bar chart and its caption
    sh = shape(s12, nm)
    if sh is not None:
        sh._element.getparent().remove(sh._element)
replace_picture(s12, shape(s12, "UPD_best"), ROOT / "figures/results/R4_best_vs_nairas.png",
                "UPD_best", 0.6, 1.45, 12.1)
for k, (rect, val, lab) in enumerate((("Rounded Rectangle 7", "TextBox 8", "TextBox 9"),
                                      ("Rounded Rectangle 10", "TextBox 11", "TextBox 12"),
                                      ("Rounded Rectangle 13", "TextBox 14", "TextBox 15"))):
    x = 0.6 + 4.1 * k
    move(shape(s12, rect), x, 5.3)
    move(shape(s12, val), x + 0.25, 5.42)
    move(shape(s12, lab), x + 0.25, 6.16)
set_text(shape(s12, "TextBox 11"), "89% → 73%")
set_text(shape(s12, "TextBox 12"), "of high-dose readings underpredicted: NAIRAS → weighted Transformer")
s12.notes_slide.notes_text_frame.text = (
    "Left three panels: NAIRAS, the best overall model (CNN–MLP, 24 h) and the weighted Transformer against the "
    "measurement, all on unseen flights. Right: the error grows with the true dose for every model. "
    "Weighting high-dose errors four times (weighted Huber) cuts the high-dose error 19% vs NAIRAS "
    "(-0.82 vs the MLP, 95% CI -0.92 to -0.71) at no overall cost, yet all models still underpredict the "
    "highest doses. XGBoost 2025 did not report high-dose error.")

# ---- slide 15: six-split figure and cards
pic = shape(s15, "Picture 5")
replace_picture(s15, pic, ROOT / "figures/results/E1_why_error.png", "Picture 5", 2.07, 1.5, 9.15, 4.3) \
    if pic is not None else None
set_text(shape(s15, "TextBox 7"), "Solar wind")
set_text(shape(s15, "TextBox 8"), "only history adding beyond its current value; NM counts act as a level")
set_text(shape(s15, "TextBox 10"), "83%")
set_text(shape(s15, "TextBox 11"), "of remaining error varies minute to minute within a flight: noise")
set_text(shape(s15, "TextBox 13"), "15% → 5%")
set_text(shape(s15, "TextBox 14"), "of error tied to the aircraft (NAIRAS → model): an instrument effect")
s15.notes_slide.notes_text_frame.text = (
    "Averages over the six train/test splits. The correction relies on altitude, shielding and the "
    "neutron-monitor level. Flattening each history to its current value shows that only the solar-wind "
    "history adds information (+0.03 RMSE). NAIRAS's error is tied to the aircraft (15%) and the year (9%); "
    "after the model these drop to 5% and 3%, and 83% of what remains varies within a flight.")

# ---- slide 16: conclusion card 4
set_text(shape(s16, "TextBox 15"), "What drives the correction")
set_text(shape(s16, "TextBox 16"), "Altitude, shielding, neutron-monitor level; of all histories, only solar wind adds.")
tb = shape(s16, "TextBox 17")
p0 = tb.text_frame.paragraphs[0]
p0.runs[-1].text = ("galactic cosmic rays only; all models still underpredict the highest doses; "
                    "noise and instrument differences bound accuracy.")

prs.save(DECK)
print("patched", DECK)
