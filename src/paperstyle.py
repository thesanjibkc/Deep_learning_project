"""Matplotlib style for figures in the CVPR-format report (not the poster/slides).

Course guideline: the report must have the look and feel of an IEEE CVPR paper; figures must be
labeled, adequately captioned, and referred to in the text. So, for report figures:
  - no titles or narrative text inside the image: the LaTeX caption explains the figure;
    panels carry only a short label such as "(a) NAIRAS V3"
  - Times serif text matching the paper body, 7-8 pt at the printed size
  - drawn at the exact printed width (one column = 3.25 in, two columns = 6.875 in) so
    fonts are not rescaled by LaTeX; saved as vector PDF
  - colorblind-safe (Okabe-Ito) colors that remain distinguishable in grayscale
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL, FULL = 3.25, 6.875                     # CVPR column and text widths (inches)
OUT = Path(__file__).resolve().parents[1] / "figures" / "paper"

# Okabe-Ito palette
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMILION, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
GRAY, LIGHT = "#7F7F7F", "#D9D9D9"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlepad": 3,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)
    return path
