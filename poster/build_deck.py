"""Build the Team 5 slide deck: poster/Team5_Sanjib_KC_slides.pptx (16:9, 16 slides).

Native text, shapes, tables and charts (editable in PowerPoint) plus the project figures.
Numbers come from results/ (final 18-run results) and poster/animation/tracking.json.
Run:  python3 poster/build_deck.py
"""
import json
from pathlib import Path

import qrcode
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; PFIG = ROOT / "poster" / "figures"
OUT = ROOT / "poster" / "Team5_Sanjib_KC_slides.pptx"
REPO = "https://github.com/thesanjibkc/Deep_learning_project/tree/final-project"

NAVY, AMBER, TEAL, RED = "14213D", "E8892B", "24907F", "C8553D"
GRAY, INK, MUTED, TINT, WHITE, AMBER_SOFT, TEAL_SOFT = "9AA3AF", "1B1F2A", "5B6474", "F2F4F7", "FFFFFF", "FCE9D3", "D7F0EB"
HEAD, BODY = "Calibri", "Calibri"
rgb = lambda h: RGBColor.from_string(h)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = 13.333, 7.5
TOTAL = 16


def box(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(1.25)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if name:
        s.name = name
    return s


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         font=BODY, italic=False, space_after=0, name=None):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(space_after)
        for t, o in ([(p, {})] if isinstance(p, str) else p):
            r = para.add_run(); r.text = t
            f = r.font; f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic); f.color.rgb = rgb(o.get("color", color))
    if name:
        tb.name = name
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=INK, gap=8):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.25)))); pPr.set("indent", str(-Emu(Inches(0.25))))
        from pptx.oxml.ns import qn
        bu = pPr.makeelement(qn("a:buChar"), {"char": "•"}); pPr.append(bu)
        for t, o in ([(it, {})] if isinstance(it, str) else it):
            r = p.add_run(); r.text = t
            r.font.name = BODY; r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", False)
            r.font.color.rgb = rgb(o.get("color", color))
    return tb


def picture(slide, path, x, y, w=None, h=None):
    from PIL import Image
    iw, ih = Image.open(path).size
    if w and h:                                   # fit inside box, centered
        s = min(w / iw, h / ih); pw, ph = iw * s, ih * s
        x, y, w, h = x + (w - pw) / 2, y + (h - ph) / 2, pw, ph
    elif w:
        h = w * ih / iw
    else:
        w = h * iw / ih
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def frame(n, title, kicker=None, dark=False):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = rgb(NAVY if dark else WHITE)
    if title:
        if kicker:
            text(s, 0.6, 0.38, 12, 0.3, kicker.upper(), size=12, color=AMBER if dark else MUTED, bold=True)
        text(s, 0.6, 0.62, 12.1, 0.9, title, size=32, bold=True, color=WHITE if dark else NAVY, font=HEAD,
             name="Title")
    text(s, 0.6, 7.02, 8, 0.3, "Team 5  ·  Sanjib K C  ·  CSC 8851 Deep Learning, Fall 2026", size=10,
         color="B8C0CC" if dark else MUTED)
    text(s, 12.1, 7.02, 0.65, 0.3, f"{n} / {TOTAL}", size=10, color="B8C0CC" if dark else MUTED, align=PP_ALIGN.RIGHT)
    return s


def stat(slide, x, y, w, big, label, color=NAVY, big_size=44, fill=TINT):
    box(slide, x, y, w, 1.6, fill=fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    text(slide, x + 0.25, y + 0.12, w - 0.5, 0.72, big, size=big_size, bold=True, color=color, font=HEAD)
    text(slide, x + 0.25, y + 0.86, w - 0.5, 0.7, label, size=12, color=INK)


def qr(path, url):
    img = qrcode.make(url, border=1); img.save(path); return path


# ------------------------------------------------------------------ data
R = {  # 18-run test RMSE (mean ± sd), from results/temporal_final*, results/basic
    "NAIRAS V3": (4.044, 6.845), "MLP, direct": (3.842, 6.627), "MLP (no history)": (3.737, 6.311),
    "Flat MLP, 24 h": (3.906, 6.731), "CNN–MLP, 24 h": (3.661, 6.212), "TCN, 24 h": (3.659, 6.173),
    "Transformer, direct, 24 h": (3.712, 6.417), "Transformer, 24 h": (3.667, 6.309),
    "Transformer, 1 h": (3.663, 6.261), "Transformer + high-dose weight, 1 h": (3.762, 5.563)}
TRK = json.load(open(ROOT / "poster" / "animation" / "tracking.json"))
BEST = TRK["best_epoch_transformer"]; ATT = TRK["transformer"]["attention_storm_flight"][BEST]
KP = [v / 10 for v in TRK["storm_flight"]["raw_24h"][1]]

# ------------------------------------------------------------------ 1 title
s = frame(1, None, dark=True)
text(s, 0.6, 0.7, 7.2, 0.4, "CSC 8851 DEEP LEARNING  ·  FALL 2026  ·  TEAM 5", size=13, bold=True, color=AMBER)
text(s, 0.6, 1.25, 6.9, 2.6, "Physics-Guided Temporal Transformer for Aviation Dose", size=44, bold=True,
     color=WHITE, font=HEAD, name="Title")
text(s, 0.6, 3.95, 6.6, 1.0, "Teaching a neural network what a physics model misses, using 24 hours of space weather",
     size=20, color="D6DCE6")
text(s, 0.6, 5.35, 6.6, 0.8, [[("Sanjib K C", {"bold": True, "color": WHITE, "size": 18})],
                               [("skc3@gsu.edu  ·  Department of Physics and Astronomy, Georgia State University",
                                 {"color": "B8C0CC", "size": 13})]])
box(s, 7.75, 0.75, 5.0, 5.9, fill="FFFFFF", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04)
picture(s, PFIG / "slide_map.png", 7.9, 0.95, w=4.7, h=2.7)
text(s, 7.95, 3.75, 4.6, 2.7, [[("92,415", {"size": 40, "bold": True, "color": NAVY, "font": HEAD})],
                              [("in-flight radiation measurements, 2013–2023, from about 589 flights", {"size": 14})],
                              [("−19%", {"size": 40, "bold": True, "color": AMBER, "font": HEAD})],
                              [("error on high-dose flights compared with the physics model", {"size": 14})]],
     space_after=4)
s.notes_slide.notes_text_frame.text = ("Hook: at cruising altitude, cosmic rays deliver a radiation dose that the physics "
    "model underestimates exactly when it matters most. We learn the correction with neural networks.")

# ------------------------------------------------------------------ 2 why
s = frame(2, "Aircrew are among the most exposed workers", "The problem")
bullets(s, 0.6, 1.75, 6.2, 4.6, [
    [("Cosmic rays hit the atmosphere and make particle showers. ", {"bold": True}),
     ("At 10–13 km the dose is many times the ground level.", {})],
    [("Measured on few flights. ", {"bold": True}), ("The ARMAS instrument records dose on board, but only on some aircraft.", {})],
    [("Everyone else relies on physics. ", {"bold": True}),
     ("The NAIRAS model estimates dose anywhere, for any flight.", {})],
    [("It misses the dangerous cases. ", {"bold": True}),
     ("Above 15 µSv/h, NAIRAS reads too low 89% of the time.", {})],
    [("Our question: ", {"bold": True, "color": AMBER}),
     ("can a neural network learn NAIRAS's mistakes from where the plane is and what space weather did in the past 24 hours?", {})],
], size=17, gap=12)
stat(s, 7.4, 1.8, 2.6, "89%", "of high-dose readings underestimated by NAIRAS", color=RED)
stat(s, 10.2, 1.8, 2.55, "6.85", "µSv/h NAIRAS error above 15 µSv/h", color=NAVY)
stat(s, 7.4, 3.6, 2.6, "16.5%", "of readings are high dose: rare and imbalanced", color=AMBER)
stat(s, 10.2, 3.6, 2.55, "35 × 24", "variables × hours of history per reading", color=TEAL, big_size=38)
text(s, 7.4, 5.45, 5.35, 1.2, "Sources: ARMAS (Tobiska et al. 2016), NAIRAS (Mertens et al. 2013), "
     "ML-ready data set (Sadykov et al., Space Weather 2026).", size=11, color=MUTED)
s.notes_slide.notes_text_frame.text = "Why it matters and the one question the project answers."

# ------------------------------------------------------------------ 3 knobs
s = frame(3, "Three knobs set the dose", "Physics in one picture")
picture(s, PFIG / "slide03_three_knobs.png", 0.6, 1.55, w=12.13, h=5.35)
s.notes_slide.notes_text_frame.text = ("Altitude roughly doubles the dose; the magnetic shield is the biggest knob "
    "(polar flights get about five times the equatorial dose); a quiet Sun lets in about 10% more cosmic rays. "
    "No flight in the data was exposed to solar particle events.")

# ------------------------------------------------------------------ 4 data
s = frame(4, "The data: one reading, two kinds of input", "Data")
for x, title, col, soft, lines in (
        (0.6, "8 static inputs", TEAL, TEAL_SOFT,
         ["Position: latitude, longitude", "Altitude: barometric and GPS",
          "Magnetic shield: cutoff rigidity Rc, L-shell, geomagnetic latitude and longitude",
          "One value per reading"]),
        (4.75, "35 temporal inputs", AMBER, AMBER_SOFT,
         ["Neutron monitors (4), solar wind (11)", "Kp, ap, Dst indices (3), particles (7)",
          "Solar X-rays (2), solar activity (8)",
          "History: 12 × 5 min (1 h) or 24 × 1 h (24 h)"])):
    box(s, x, 1.75, 3.9, 3.35, fill=soft, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    text(s, x + 0.3, 1.95, 3.4, 0.5, title, size=20, bold=True, color=NAVY, font=HEAD)
    bullets(s, x + 0.3, 2.55, 3.35, 2.5, lines, size=14, gap=6)
box(s, 8.9, 1.75, 3.85, 3.35, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
text(s, 9.2, 1.95, 3.3, 0.5, "Target and baseline", size=20, bold=True, color=NAVY, font=HEAD)
bullets(s, 9.2, 2.55, 3.3, 2.5, ["Target: measured ARMAS dose A (µSv/h)",
                                 "Baseline: NAIRAS estimate N",
                                 "Networks learn the correction r = A − N",
                                 "Aircraft ID is never an input"], size=14, gap=6)
text(s, 0.6, 5.35, 12, 0.4, "Three partitions, split by flight so no flight appears in two of them",
     size=16, bold=True, color=NAVY)
for i, (p, n, hi) in enumerate((("P1", "32,369", "16.4%"), ("P2", "30,752", "16.2%"), ("P3", "29,294", "16.9%"))):
    x = 0.6 + i * 4.1
    box(s, x, 5.85, 3.9, 0.95, fill=WHITE, line="DCE1E8", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
    text(s, x + 0.25, 5.97, 3.4, 0.75, [[(p + "  ", {"bold": True, "size": 22, "color": NAVY}),
                                          (f"{n} readings  ·  {hi} high dose", {"size": 15})]],
         anchor=MSO_ANCHOR.MIDDLE)
s.notes_slide.notes_text_frame.text = ("Same readings in the 5-minute and hourly files; only the history length differs. "
    "Splitting by flight prevents leakage: 56% of consecutive rows repeat the same ARMAS value.")

# ------------------------------------------------------------------ 5 where NAIRAS fails
s = frame(5, "NAIRAS cannot tell which flights will be high", "Exploration")
picture(s, FIG / "explore" / "e01_where_nairas_fails.png", 0.5, 1.5, w=6.6, h=5.4)
bullets(s, 7.5, 1.8, 5.2, 4.8, [
    [("Panel a: ", {"bold": True}), ("above 15 µSv/h the measured dose climbs to 40, but NAIRAS stays near 10–15.", {})],
    [("Panel c: ", {"bold": True}), ("grouped by the measured dose, the error grows to +17 µSv/h.", {})],
    [("Panel d: ", {"bold": True}), ("grouped by NAIRAS's own estimate, the error is centered near zero but spread wide in both directions.", {})],
    [("So: ", {"bold": True, "color": AMBER}),
     ("NAIRAS is right on average but cannot pick out the high flights. The extra information has to come from other inputs.", {})],
], size=16, gap=12)
s.notes_slide.notes_text_frame.text = ("The gap between panels c and d is regression to the mean: selecting by the measured "
    "value also selects lucky-high measurements. Both views together give the real story.")

# ------------------------------------------------------------------ 6 map
s = frame(6, "High doses live on polar routes", "Exploration")
picture(s, PFIG / "slide_map.png", 0.6, 1.55, w=12.13, h=3.65)
for i, (big, lab, col) in enumerate((("36%", "of polar readings (Rc < 1 GV) exceed 15 µSv/h", RED),
                                      ("0%", "near the equator (Rc > 10 GV), even though those flights fly highest", TEAL),
                                      ("Mixed", "red and blue errors share regions: location alone does not explain them", NAVY))):
    stat(s, 0.6 + i * 4.1, 5.3, 3.9, big, lab, color=col, big_size=34)
s.notes_slide.notes_text_frame.text = "Hawaii flights cruise at 13.6 km yet get the lowest dose: the strong shield beats altitude."

# ------------------------------------------------------------------ 7 history information
s = frame(7, "Does the past 24 hours carry information?", "Exploration")
picture(s, FIG / "explore" / "e04_temporal_information.png", 0.5, 1.5, w=7.4, h=5.4)
bullets(s, 8.2, 1.8, 4.55, 4.9, [
    [("Within 1 hour, 16 of 35 variables never change ", {"bold": True}),
     ("(Kp and ap are 3-hourly; solar indices daily).", {})],
    [("Over 24 hours, ", {"bold": True}), ("solar wind, Kp and Dst move by 1–2 standard deviations.", {})],
    [("A delayed signal: ", {"bold": True, "color": AMBER}),
     ("Kp about 15 h before the flight relates to the NAIRAS error more (ρ ≈ 0.12) than Kp at flight time (ρ ≈ 0.04).", {})],
    [("Correlations are weak, ", {"bold": True}), ("so the model has to combine many small signals.", {})],
], size=16, gap=12)
s.notes_slide.notes_text_frame.text = "This is the lead–lag idea the instructor suggested; attention later tests it independently."

# ------------------------------------------------------------------ 8 method
s = frame(8, "Learn the correction, not the dose", "Method")
box(s, 0.6, 1.7, 12.13, 1.15, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
text(s, 0.9, 1.78, 11.6, 1.0, [[("prediction  =  ", {"size": 24, "color": MUTED}),
                                ("NAIRAS", {"size": 24, "bold": True, "color": TEAL}),
                                ("  +  ", {"size": 24, "color": MUTED}),
                                ("network(history, static inputs, NAIRAS)", {"size": 24, "bold": True, "color": AMBER})],
                               [("The physics model supplies the baseline; the network learns only what it misses.",
                                 {"size": 14, "color": MUTED})]], anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
text(s, 0.6, 3.15, 12, 0.45, "A ladder of models: each step adds one idea, so every gain can be traced",
     size=18, bold=True, color=NAVY)
ladder = [("NAIRAS V3", "physics model", "—", GRAY), ("MLP", "current values only", "14 k", TEAL),
          ("Flat MLP", "history as one flat row", "57 k", RED), ("CNN–MLP", "convolutions over time", "7 k", AMBER),
          ("TCN", "dilated causal convolutions", "10 k", AMBER), ("Transformer", "attention over hour-tokens", "9 k", AMBER)]
for i, (n, d, p, c) in enumerate(ladder):
    x = 0.6 + i * 2.05; y = 5.75 - i * 0.36
    box(s, x, y, 1.9, 6.75 - y, fill=c, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    text(s, x + 0.12, y + 0.1, 1.66, 0.4, n, size=15, bold=True, color=WHITE)
    text(s, x + 0.12, y + 0.48, 1.66, 0.55, d, size=11, color=WHITE)
    text(s, x + 0.12, 6.38, 1.66, 0.3, f"{p} params" if p != "—" else "no training", size=11, color=WHITE, bold=True)
s.notes_slide.notes_text_frame.text = ("All networks except the direct variants learn the residual. Final setting chosen "
    "on validation only: width 16, dropout 0.3, weight decay 0.01. A high-dose variant weights errors above 15 µSv/h four times "
    "(weighted Huber loss).")

# ------------------------------------------------------------------ 9 attention
s = frame(9, "How the Transformer reads 24 hours", "Method")
x0, tw, ty = 0.75, 0.33, 4.65
sx, sy = x0 + 12 * tw - 0.75, 2.05
box(s, sx, sy, 1.5, 0.62, fill=AMBER_SOFT, line=AMBER, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.2)
text(s, sx, sy, 1.5, 0.62, "summary token", size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
mx = max(ATT)
for i in range(24):
    cx = x0 + i * tw + tw / 2
    k = max(0.0, (ATT[i] - min(ATT)) / (mx - min(ATT)))
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(sx + 0.75), Inches(sy + 0.62), Inches(cx), Inches(ty))
    ln.line.color.rgb = rgb(AMBER); ln.line.width = Pt(0.5 + 7 * k)
    kpk = KP[i] / 9
    box(s, x0 + i * tw + 0.02, ty, tw - 0.04, 0.55, fill=RED if KP[i] >= 5 else ("C9D0DA" if kpk < 0.4 else GRAY))
    bh = 0.9 * KP[i] / 9
    box(s, x0 + i * tw + 0.05, 6.15 - bh, tw - 0.1, bh, fill=RED if KP[i] >= 5 else GRAY)
text(s, x0, 5.3, 8, 0.3, "Each block is one hour-token: 35 readings at that hour. Red = storm hours (Kp ≥ 5).", size=12, color=MUTED)
text(s, x0, 6.2, 1.5, 0.3, "23 h before", size=11, color=MUTED)
text(s, x0 + 24 * tw - 1.0, 6.2, 1.0, 0.3, "flight", size=11, color=MUTED, align=PP_ALIGN.RIGHT)
text(s, x0 + 5 * tw - 0.4, 5.85, 1.4, 0.3, "Kp 8.3", size=11, bold=True, color=RED)
steps = [("1  Tokens", "each hour's 35 values become one vector"),
         ("2  Scores", "the summary token asks every hour: how relevant are you?  q · k / √d"),
         ("3  Weights", "softmax turns scores into weights that add to 1 (beam width)"),
         ("4  Correction", "weighted summary + altitude + shield + NAIRAS → correction")]
for i, (h_, d_) in enumerate(steps):
    y = 1.75 + i * 1.12
    box(s, 9.3, y, 3.45, 0.98, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
    text(s, 9.5, y + 0.1, 3.1, 0.35, h_, size=16, bold=True, color=NAVY)
    text(s, 9.5, y + 0.45, 3.1, 0.5, d_, size=12, color=INK)
text(s, 0.75, 1.6, 8.2, 0.35, f"Real learned attention, 9 Sept 2017 flight (epoch {BEST}). No storm labels given.",
     size=14, bold=True, color=AMBER)
s.notes_slide.notes_text_frame.text = ("Beam width is the real learned attention at the best epoch. It peaks 16–18 h before the "
    "flight, where Kp reached 8.3. Attention = softmax(QKᵀ/√d)V.")

# ------------------------------------------------------------------ 10 protocol
s = frame(10, "A fair test: unseen flights, 18 runs per model", "Evaluation")
rots = ["123", "132", "213", "231", "312", "321"]
role = {0: ("train", TEAL), 1: ("validate", AMBER), 2: ("test", NAVY)}
text(s, 0.6, 1.65, 6, 0.4, "Every split of P1, P2, P3 into train / validate / test", size=16, bold=True, color=NAVY)
for j, p in enumerate(("P1", "P2", "P3")):
    text(s, 1.55 + j * 1.55, 2.1, 1.4, 0.3, p, size=14, bold=True, color=MUTED, align=PP_ALIGN.CENTER)
for i, r in enumerate(rots):
    y = 2.45 + i * 0.6
    text(s, 0.6, y + 0.08, 0.9, 0.35, f"split {i + 1}", size=12, color=MUTED)
    for pos_, part in enumerate(r):
        lab, col = role[pos_]
        j = int(part) - 1
        b = box(s, 1.55 + j * 1.55, y, 1.4, 0.5, fill=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.15)
        text(s, 1.55 + j * 1.55, y, 1.4, 0.5, lab, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 6.15, 6.3, 0.5, "× 3 random seeds = 18 runs per model", size=16, bold=True, color=AMBER)
bullets(s, 7.4, 1.75, 5.35, 5, [
    [("No peeking: ", {"bold": True}), ("scaling statistics come from the training flights only.", {})],
    [("Settings chosen on validation only ", {"bold": True}),
     ("(one split). A first pass with 4–8× larger models overfit within 2–7 epochs.", {})],
    [("Test once: ", {"bold": True}), ("each run is scored on flights it never saw.", {})],
    [("Significance: ", {"bold": True}),
     ("a paired bootstrap that resamples whole flights (1,000 draws) compares two models on the same test flights.", {})],
    [("Error reported overall and above 15 µSv/h, ", {"bold": True}), ("always next to NAIRAS on the same readings.", {})],
], size=16, gap=12)
s.notes_slide.notes_text_frame.text = "Rotation makes sure the result is not luck of one split; seeds make sure it is not luck of initialization."

# ------------------------------------------------------------------ 11 results overall


def bar_chart(slide, x, y, w, h, names, values, colors, title, lo, hi, fmt='0.00'):
    cd = CategoryChartData(); cd.categories = names; cd.add_series(title, values)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart; ch.has_legend = False; ch.has_title = False
    pl = ch.plots[0]; pl.gap_width = 45; pl.has_data_labels = True
    dl = pl.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(12); dl.font.color.rgb = rgb(INK)
    for i, c in enumerate(colors):
        pt = pl.series[0].points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
    va = ch.value_axis; va.minimum_scale = lo; va.maximum_scale = hi
    va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb("E3E7ED")
    va.tick_labels.font.size = Pt(11); va.tick_labels.font.color.rgb = rgb(MUTED); va.format.line.fill.background()
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(13); ca.tick_labels.font.color.rgb = rgb(INK)
    ca.format.line.color.rgb = rgb("C9D0DA"); ca.reverse_order = True
    return gf


col_of = lambda n: GRAY if n.startswith("NAIRAS") else TEAL if "MLP" in n and "Flat" not in n else RED if "Flat" in n else AMBER
s = frame(11, "History helps when it is read with structure", "Results")
order = ["NAIRAS V3", "Flat MLP, 24 h", "MLP, direct", "MLP (no history)", "Transformer, direct, 24 h",
         "Transformer, 24 h", "Transformer, 1 h", "CNN–MLP, 24 h", "TCN, 24 h"]
bar_chart(s, 0.5, 1.55, 8.0, 5.35, order, [R[n][0] for n in order], [col_of(n) for n in order],
          "Test RMSE, all flights", 3.4, 4.15)
text(s, 0.6, 6.72, 7.8, 0.3, "Test RMSE (µSv/h), mean of 18 runs; shorter is better", size=11, color=MUTED)
stat(s, 8.85, 1.7, 3.9, "−9.5%", "error vs NAIRAS for TCN, CNN–MLP and Transformer with history", color=AMBER)
stat(s, 8.85, 3.45, 3.9, "−0.07", "µSv/h vs the no-history MLP: better in all 6 splits (95% CI −0.10 to −0.04)", color=TEAL, big_size=40)
stat(s, 8.85, 5.2, 3.9, "3.67", "ties the 2025 static XGBoost (3.671); different data, so no claim of a win", color=NAVY, big_size=40)
s.notes_slide.notes_text_frame.text = ("Flat MLP is worst among networks: same history, no structure, more parameters. "
    "1-hour and 24-hour windows give the same average error (difference +0.01, not significant).")

# ------------------------------------------------------------------ 12 high dose
s = frame(12, "Weighting high doses cuts their error by 19%", "Results")
order2 = ["NAIRAS V3", "MLP (no history)", "Transformer, 24 h", "TCN, 24 h", "Transformer + high-dose weight, 1 h"]
bar_chart(s, 0.5, 1.55, 8.0, 5.0, ["NAIRAS V3", "MLP (no history)", "Transformer, 24 h", "TCN, 24 h",
                                   "Transformer + high-dose weight"],
          [R[n][1] for n in order2], [GRAY, TEAL, AMBER, AMBER, RED], "High-dose RMSE", 4.5, 7.2)
text(s, 0.6, 6.6, 7.8, 0.35, "Test RMSE (µSv/h) on readings above 15 µSv/h, mean of 18 runs", size=11, color=MUTED)
stat(s, 8.85, 1.7, 3.9, "5.56", "µSv/h high-dose error, down from 6.85 for NAIRAS (−19%)", color=RED)
stat(s, 8.85, 3.45, 3.9, "−0.78", "µSv/h vs the MLP on high doses, all 6 splits (95% CI −0.89 to −0.67)", color=AMBER, big_size=40)
stat(s, 8.85, 5.2, 3.9, "+0.00", "change in overall error vs the MLP: no cost", color=TEAL, big_size=40)
s.notes_slide.notes_text_frame.text = ("One knob controls the trade-off: high-dose errors count four times in the loss "
    "(weighted Huber). XGBoost 2025 did not report high-dose error.")

# ------------------------------------------------------------------ 13 ablations
s = frame(13, "What made the difference", "Ablations")
rows = [("Change tested", "All flights", "High dose", "Verdict"),
        ("Learn the residual instead of the dose (MLP)", "−0.07", "−0.30", "Helps"),
        ("Add 24 h history (Transformer vs MLP)", "−0.07", "0.00", "Helps overall"),
        ("Add 24 h history (TCN vs MLP)", "−0.07", "−0.14", "Helps"),
        ("Read history with structure (Transformer vs Flat MLP)", "−0.24", "−0.43", "Helps most"),
        ("Longer window (24 h vs 1 h Transformer)", "+0.01", "+0.05", "No difference"),
        ("Weight high doses ×4 (vs unweighted)", "+0.07", "−0.67", "Trade-off")]
tb = s.shapes.add_table(len(rows), 4, Inches(0.6), Inches(1.65), Inches(12.13), Inches(4.4)).table
for j, w_ in enumerate((6.2, 1.9, 1.9, 2.13)):
    tb.columns[j].width = Inches(w_)
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        c = tb.cell(i, j); c.text = v
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if j in (0, 3) else PP_ALIGN.CENTER
        f = p.runs[0].font; f.size = Pt(15 if i else 14); f.name = BODY; f.bold = i == 0 or j == 3
        f.color.rgb = rgb(WHITE if i == 0 else INK)
        c.fill.solid(); c.fill.fore_color.rgb = rgb(NAVY if i == 0 else (TINT if i % 2 else WHITE))
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        if i and j in (1, 2) and v.startswith("−") and v not in ("−0.00",) and not (i == 2 and j == 2):
            f.color.rgb = rgb(TEAL); f.bold = True
        if i and j in (1, 2) and v.startswith("+") and i == 6 and j == 1:
            f.color.rgb = rgb(RED); f.bold = True
text(s, 0.6, 6.2, 12.1, 0.65, "Paired difference in test RMSE (µSv/h); negative = the change helps. Bold teal = 95% interval "
     "excludes zero (flight bootstrap, seed-ensembled predictions, 6 splits).", size=12, color=MUTED)
s.notes_slide.notes_text_frame.text = "Each row changes exactly one thing, so each gain can be attributed."

# ------------------------------------------------------------------ 14 attention learning
s = frame(14, "Watching attention find the storm", "Interpretation")
picture(s, FIG / "results" / "T1_training_evolution.png", 0.5, 1.5, w=8.0, h=5.45)
bullets(s, 8.85, 1.8, 3.9, 3.6, [
    [("Epoch 0: ", {"bold": True}), ("attention is even.", {})],
    [("By epoch 11: ", {"bold": True}), ("it peaks 16–18 h before the flight, exactly where Kp hit 8.3.", {})],
    [("Flat MLP: ", {"bold": True}), ("even weights; memorizes after epoch 4.", {})],
    [("Agrees ", {"bold": True, "color": AMBER}), ("with the 15 h lead–lag found in exploration.", {})],
], size=15, gap=10)
qr(PFIG / "qr_repo.png", REPO)
picture(s, PFIG / "qr_repo.png", 8.9, 5.3, w=1.4)
text(s, 10.45, 5.4, 2.3, 1.3, "Live animation on the laptop. Code and data pipeline: scan for GitHub.", size=12, color=MUTED)
text(s, 0.6, 6.75, 8, 0.3, "One flight, one run: attention shows what the model used, not physical cause.", size=11,
     color=MUTED, italic=True)
s.notes_slide.notes_text_frame.text = "Play the 'Watching Attention Learn' animation here if a laptop is available."

# ------------------------------------------------------------------ 15 why error remains
s = frame(15, "Why some error remains", "Interpretation")
picture(s, FIG / "results" / "E1_why_error.png", 0.5, 1.5, w=12.3, h=4.3)
for i, (big, lab, col) in enumerate((("Altitude + shield", "are what the correction relies on most (permutation importance)", TEAL),
                                      ("81%", "of remaining error varies minute to minute within a flight: noise", NAVY),
                                      ("4–7%", "follows the aircraft, above the 3% chance level: an instrument effect", RED))):
    stat(s, 0.6 + i * 4.1, 5.35, 3.9, big, lab, color=col, big_size=26 if i == 0 else 34)
s.notes_slide.notes_text_frame.text = ("After correction, shielding, altitude and storm level each explain under 1% of the "
    "remaining error. What is left cannot be predicted from space-weather inputs.")

# ------------------------------------------------------------------ 16 conclusions
s = frame(16, "What we learned", "Conclusions", dark=True)
lessons = [("Start from physics", "Learning NAIRAS's correction beat learning the dose: more accurate and three times more stable."),
           ("Structure beats size", "Small, regularized temporal networks beat a no-history MLP; a flat, larger network memorized."),
           ("Train for what matters", "Weighting high doses cut their error by 19% vs NAIRAS at no overall cost."),
           ("Attention found the storm", "On its own, it focused on the hours of a severe storm 16–18 h before a flight.")]
for i, (h_, d_) in enumerate(lessons):
    x = 0.6 + (i % 2) * 6.15; y = 1.75 + (i // 2) * 1.85
    box(s, x, y, 5.95, 1.65, fill="1F2E52", shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    text(s, x + 0.3, y + 0.2, 5.4, 0.45, h_, size=21, bold=True, color=AMBER, font=HEAD)
    text(s, x + 0.3, y + 0.72, 5.4, 0.85, d_, size=15, color=WHITE)
text(s, 0.6, 5.55, 9.6, 1.35, [[("Limits: ", {"bold": True, "color": AMBER}),
                                ("one instrument family; galactic cosmic rays only (no solar-particle flights); "
                                 "measurement noise bounds accuracy.", {"color": "D6DCE6"})],
                               [("Next: ", {"bold": True, "color": AMBER}),
                                ("uncertainty bands from seed ensembles, per-instrument calibration, pretraining on long "
                                 "space-weather archives.", {"color": "D6DCE6"})]], size=15, space_after=6)
box(s, 10.75, 5.35, 1.6, 1.6, fill=WHITE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
picture(s, PFIG / "qr_repo.png", 10.85, 5.45, w=1.4)
text(s, 10.4, 5.0, 2.3, 0.3, "Code on GitHub", size=12, color="D6DCE6", align=PP_ALIGN.CENTER)
s.notes_slide.notes_text_frame.text = "Close with the one-sentence answer to the reviewer: deep learning wins on high doses and shows storm timing."

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
