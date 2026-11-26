"""Add 'run the models live' QR codes to the Team 5 deck (slides 8, 9 and 14).

Used by build_deck.py (fresh builds) and run directly to patch an existing, possibly hand-edited deck:
    python3 poster/deck_qr.py [path/to/deck.pptx]
Idempotent: shapes named QR_zoo_* are replaced, not duplicated.
"""
import sys
from pathlib import Path

import qrcode
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ZOO_URL = "https://claude.ai/artifact/XTzA1TvvKFi7pJ45pAKZ9B"   # Flight Model Zoo (share it from the page before the session)
NAVY, MUTED = "14213D", "5B6474"


def _label(slide, x, y, w, text, size=12, bold=False, color=MUTED, align=PP_ALIGN.RIGHT, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.5))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.name = "Calibri"
    r.font.color.rgb = RGBColor.from_string(color)
    if name:
        tb.name = name
    return tb


def add_zoo_qr(prs):
    img = HERE / "figures" / "qr_zoo.png"
    qrcode.make(ZOO_URL, border=1).save(img)
    for i, s in enumerate(prs.slides):                      # remove an earlier copy first
        for sh in list(s.shapes):
            if sh.name.startswith("QR_zoo"):
                sh._element.getparent().remove(sh._element)
    slides = prs.slides
    # slides 8 and 9: top-right corner, label to the left of the code
    for idx, line in ((7, "Run these models live: scan the code"), (8, "Try the Transformer yourself: scan the code")):
        s = slides[idx]
        pic = s.shapes.add_picture(str(img), Inches(11.55), Inches(0.3), width=Inches(1.2)); pic.name = "QR_zoo_img"
        _label(s, 8.9, 0.62, 2.5, line, size=12, bold=True, color=NAVY, name="QR_zoo_label")
    # slide 14: beside the GitHub code
    s = slides[13]
    for sh in list(s.shapes):                                # replace the old caption that mentioned only GitHub
        if sh.has_text_frame and sh.text_frame.text.startswith("Live animation on the laptop"):
            sh._element.getparent().remove(sh._element)
    pic = s.shapes.add_picture(str(img), Inches(10.5), Inches(5.3), width=Inches(1.4)); pic.name = "QR_zoo_img"
    _label(s, 8.7, 6.72, 1.8, "Code on GitHub", size=11, align=PP_ALIGN.CENTER, name="QR_zoo_label_code")
    _label(s, 10.3, 6.72, 1.8, "Run the models live", size=11, bold=True, color=NAVY, align=PP_ALIGN.CENTER, name="QR_zoo_label")
    _label(s, 12.0, 5.4, 0.85, "scan to open", size=10, align=PP_ALIGN.LEFT, name="QR_zoo_label_hint")


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "Team5_Sanjib_KC_slides.pptx")
    prs = Presentation(path)
    add_zoo_qr(prs)
    prs.save(path)
    print("patched", path)
