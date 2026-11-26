"""Slide 3: the three knobs of aviation dose, with numbers from the ARMAS data.

Altitude : median ARMAS at Rc < 1 GV, 8-10 km vs 12-14 km
Shield   : median ARMAS at 10.5-12 km, Rc > 10 GV vs Rc < 1 GV
Sun      : median Oulu neutron-monitor rate, active-Sun years (2014, 2023) vs solar minimum (2019-20)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

NAVY, AMBER, TEAL, INK, MUTED = "#14213D", "#F2A33A", "#2A9D8F", "#1B1B1B", "#6B7280"
CMAP = LinearSegmentedColormap.from_list("dose", [TEAL, "#E9C46A", AMBER, "#E76F51"])
plt.rcParams.update({"font.family": "DejaVu Sans"})

knobs = [
    dict(title="ALTITUDE", sub="less air above the aircraft", lo="8–10 km", hi="12–14 km",
         lo_v="9.7 µSv/h", hi_v="16.8 µSv/h", big="×1.7", note="median dose, polar routes", reverse=False),
    dict(title="MAGNETIC SHIELD", sub="cutoff rigidity Rc", lo="Rc > 10 GV\n(equator)", hi="Rc < 1 GV\n(poles)",
         lo_v="3.2 µSv/h", hi_v="15.2 µSv/h", big="×4.8", note="median dose, 10.5–12 km", reverse=False),
    dict(title="THE SUN", sub="two opposite effects", lo="active Sun\n(2014, 2023)", hi="quiet Sun\n(2019–20)",
         lo_v="101–104 /s", hi_v="112–113 /s", big="+10%", note="galactic cosmic rays (Oulu neutron monitor)", reverse=False),
]

fig = plt.figure(figsize=(13.33, 7.5), dpi=200)
fig.patch.set_facecolor("white")
fig.text(0.05, 0.9, "Three knobs set the radiation dose at flight altitude", fontsize=30,
         fontweight="bold", color=NAVY)
fig.text(0.05, 0.845, "Each dial runs from low dose (teal) to high dose (amber). Numbers are from the 92,415 ARMAS observations.",
         fontsize=14, color=MUTED)

for k, kn in enumerate(knobs):
    ax = fig.add_axes([0.04 + k * 0.315, 0.2, 0.29, 0.58])
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.25, 1.35); ax.set_aspect("equal"); ax.axis("off")
    # gradient arc
    n = 90
    for i in range(n):
        a0 = 180 - i * 180 / n
        ax.add_patch(Wedge((0, 0), 1.0, a0 - 180 / n - 0.3, a0, width=0.22,
                           color=CMAP(i / (n - 1)), lw=0))
    # ticks
    for t in np.linspace(0, 1, 11):
        a = np.pi * (1 - t)
        ax.plot([0.74 * np.cos(a), 0.70 * np.cos(a)], [0.74 * np.sin(a), 0.70 * np.sin(a)],
                color=NAVY, lw=1.2, alpha=.5)
    # needle pointing to the high-dose end
    a = np.pi * 0.12
    ax.annotate("", xy=(0.86 * np.cos(a), 0.86 * np.sin(a)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=3.5, mutation_scale=22))
    ax.add_patch(Circle((0, 0), 0.08, color=NAVY, zorder=5))
    # labels
    ax.text(0, 1.27, kn["title"], ha="center", fontsize=19, fontweight="bold", color=NAVY)
    ax.text(0, 1.13, kn["sub"], ha="center", fontsize=12.5, color=MUTED, style="italic")
    ax.text(-1.0, -0.13, kn["lo"], ha="center", va="top", fontsize=11.5, color=TEAL, fontweight="bold")
    ax.text(1.0, -0.13, kn["hi"], ha="center", va="top", fontsize=11.5, color="#C8553D", fontweight="bold")
    ax.text(-1.0, -0.47, kn["lo_v"], ha="center", va="top", fontsize=13, color=INK)
    ax.text(1.0, -0.47, kn["hi_v"], ha="center", va="top", fontsize=13, color=INK, fontweight="bold")
    ax.text(0, 0.36, kn["big"], ha="center", fontsize=30, fontweight="bold", color="#C8553D")
    ax.text(0, -0.8, kn["note"], ha="center", fontsize=10.5, color=MUTED)
    if k == 2:
        ax.text(0, -1.02, "eruptions can add solar particles (SEP/GLE),", ha="center",
                fontsize=11, color="#C8553D", fontweight="bold")
        ax.text(0, -1.17, "but no flight in this dataset was exposed to one",
                ha="center", fontsize=10.5, color="#C8553D")

box = FancyBboxPatch((0.05, 0.05), 0.9, 0.11, boxstyle="round,pad=0.008,rounding_size=0.02",
                     transform=fig.transFigure, facecolor="#FFF4E0", edgecolor=AMBER, lw=1.5)
fig.patches.append(box)
fig.text(0.5, 0.122, "Highest dose  =  high altitude  +  weak shield (polar route)  +  quiet Sun",
         ha="center", va="center", fontsize=17, color=NAVY, fontweight="bold")
fig.text(0.5, 0.082, "The knobs interact: near the equator the shield is so strong that flying higher barely matters.",
         ha="center", va="center", fontsize=13, color=NAVY)

out = Path(__file__).parent / "figures" / "slide03_three_knobs.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=200, facecolor="white")
print("saved", out)
