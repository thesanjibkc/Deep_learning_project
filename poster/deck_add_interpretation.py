"""Add two interpretation slides to the Team 5 deck (2026-11-27): which inputs the correction uses, and occlusion.

The deck is printed slide by slide and pinned up as the poster, so each slide must explain itself.
New slides are copies of the 'Why some error remains' slide (figure + three cards) inserted after the ablation
slide (13). Old slides 14-16 become 16-18, and every footer page number becomes 'n / 18'. Idempotent.

Figures are cut from the report's paper figures (figures/paper/*.pdf) at print resolution with scripts/pdf2png.js
(macOS PDFKit via osascript); crops are written to figures/poster/.
"""
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
DECK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "poster" / "Team5_Sanjib_KC_slides.pptx"
OUT = ROOT / "figures" / "poster"
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 8                      # PDF points -> pixels; crops below were measured at scale 5
K = SCALE / 5

A_TITLE = "What the correction relies on"
B_TITLE = "Every network leans on the same few inputs"


def render(pdf):
    d = tempfile.mkdtemp()
    subprocess.run(["osascript", "-l", "JavaScript", str(ROOT / "scripts" / "pdf2png.js"), str(pdf), d, str(SCALE)],
                   check=True, capture_output=True)
    return Image.open(os.path.join(d, "page-01.png")).convert("RGB")


def crops():
    f = render(ROOT / "figures/paper/features.pdf")
    f.crop((0, 0, int(904 * K), f.height)).save(OUT / "inputs_shuffle.png")          # panel (a) + legend
    c = f.crop((int(1536 * K), 0, f.width, f.height))                              # panel (c) + legend
    c.paste((255, 255, 255), (0, int(0.70 * c.height), int(14 * K), int(0.82 * c.height)))  # tail of (b)'s x label
    c.save(OUT / "inputs_history.png")
    o = render(ROOT / "figures/paper/occlusion_models.pdf")
    o.crop((0, 0, int(1660 * K), o.height)).save(OUT / "occlusion_models_ab.png")    # panels (a, b)


prs = Presentation(DECK)
titles = [next((sh.text_frame.text for sh in s.shapes if sh.name == "Title"), "") for s in prs.slides]


def shape(slide, name):
    return next(sh for sh in slide.shapes if sh.name == name)


def set_runs(sh, *texts, para=0):
    for r, t in zip(sh.text_frame.paragraphs[para].runs, texts):
        r.text = t


def clone(src):
    """Copy a slide (shapes except pictures, background) to the end of the deck."""
    new = prs.slides.add_slide(src.slide_layout)
    for ph in list(new.placeholders):
        ph._element.getparent().remove(ph._element)
    if src._element.cSld.bg is not None:
        new._element.cSld.insert(0, copy.deepcopy(src._element.cSld.bg))
    for sh in src.shapes:
        if sh.shape_type != 13:
            new.shapes._spTree.append(copy.deepcopy(sh._element))
    return new


def move(slide, index):
    ids = prs.slides._sldIdLst
    el = next(e for e in ids if e.rId == prs.part.relate_to(slide.part, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"))
    ids.remove(el); ids.insert(index, el)


def picture(slide, path, x, y, max_w, max_h, name):
    w, h = Image.open(path).size
    s = min(max_w / w, max_h / h)
    pw, ph = w * s, h * s
    pic = slide.shapes.add_picture(str(path), Inches(x + (max_w - pw) / 2), Inches(y), Inches(pw), Inches(ph))
    pic.name = name
    return pic


if A_TITLE not in titles:
    crops()
    src = next(s for s in prs.slides if any(sh.name == "Title" and sh.text_frame.text == "Why some error remains" for sh in s.shapes))

    # ---- slide A: which inputs the correction uses (24 h Transformer, permutation and history tests)
    a = clone(src)
    set_runs(shape(a, "Title"), A_TITLE)
    picture(a, OUT / "inputs_shuffle.png", 0.75, 1.55, 5.6, 3.65, "Picture A1")
    picture(a, OUT / "inputs_history.png", 6.75, 1.55, 5.6, 3.65, "Picture A2")
    set_runs(shape(a, "TextBox 7"), "+0.40")
    set_runs(shape(a, "TextBox 8"), "µSv/h error when altitude is shuffled; shielding +0.38 (+0.50 on high-dose flights)")
    set_runs(shape(a, "TextBox 10"), "+0.21")
    set_runs(shape(a, "TextBox 11"), "µSv/h when the neutron-monitor past is swapped, ~0 when flattened: it reads their level")
    set_runs(shape(a, "TextBox 13"), "Solar wind")
    set_runs(shape(a, "TextBox 14"), "the one history that adds beyond its current value (+0.03 of the +0.035 from all histories)")
    a.notes_slide.notes_text_frame.text = (
        "24-hour residual Transformer, seed 42, all six assignments. Left: rise in test RMSE when one input is shuffled "
        "across test readings. Right: blue, a group's past replaced by its current value (flattened); orange, its past "
        "swapped with another reading's while the current value is kept.")

    # ---- slide B: occlusion of every residual network
    b = clone(src)
    set_runs(shape(b, "Title"), B_TITLE)
    picture(b, OUT / "occlusion_models_ab.png", 0.75, 1.55, 11.6, 3.65, "Picture B1")
    set_runs(shape(b, "TextBox 7"), "Altitude")
    set_runs(shape(b, "TextBox 8"), "hiding GPS altitude costs +0.60 to +0.86 µSv/h in every network except the Flat MLP")
    set_runs(shape(b, "TextBox 10"), "≤ 0.04")
    set_runs(shape(b, "TextBox 11"), "µSv/h: the most any single input matters to the Flat MLP, which spreads itself over 840 values")
    set_runs(shape(b, "TextBox 13"), "TCN")
    set_runs(shape(b, "TextBox 14"), "the only network tied to the last time step (+0.09 µSv/h when it is hidden)")
    b.notes_slide.notes_text_frame.text = (
        "Occlusion: one input is replaced by its training mean and the change in test RMSE is measured (seed 42, six "
        "assignments, 24-hour models; the MLP sees current values only). Right: one time step occluded at a time.")

    move(a, 13); move(b, 14)

    # ---- the remaining-error slide: its solar-wind card moves to slide A
    set_runs(shape(src, "TextBox 7"), "< 1%")
    set_runs(shape(src, "TextBox 8"), "of the remaining error is explained by altitude, shielding or storm level")

# ---- footers: 'n / 18'
for n, s in enumerate(prs.slides, start=1):
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().endswith(("/ 16", "/ 17", "/ 18")) and len(sh.text_frame.text) < 9:
            set_runs(sh, f"{n} / {len(prs.slides)}")

prs.save(DECK)
print("slides:", len(prs.slides), "->", DECK)
