"""NAIRAS vs the best models, each against the ARMAS measurement, on unseen flights.

Every observation is predicted once, by the 3-seed ensemble of the assignment whose TEST
partition contains it (rotations 123 -> P3, 231 -> P1, 312 -> P2), so no prediction comes
from a model that saw that flight.

Panels: (a) NAIRAS V3, (b) best overall model (TCN, 24 h), (c) best high-dose model
(Transformer with high-dose weight, 1 h), each versus ARMAS; (d) median error A - prediction
by bins of the measured dose, with 25th-75th percentile bands.

Writes figures/results/R4_best_vs_nairas.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).parent))
from data import load, split

NAVY, AMBER, TEAL, GRAY, RED = "#14213D", "#F2A33A", "#2A9D8F", "#9CA3AF", "#C8553D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#6B7280"})
OUT = Path("figures/results"); OUT.mkdir(parents=True, exist_ok=True)
ROTS = ["123", "231", "312"]; SEEDS = [42, 1, 2]

d = load("1H"); A, N = d["A"], d["N"]
MODELS = {"TCN, 24 h": ("results/temporal_final/1H", "TCN_residual"),
          "Transformer, 1 h, w = 4": ("results/temporal_final_weighted/5M", "Transformer_residual")}
pred = {}
for lab, (root, stem) in MODELS.items():
    p = np.full_like(A, np.nan)
    for r in ROTS:
        te = split(d, *(f"P{c}" for c in r))["test"]
        p[te] = np.mean([np.load(Path(root) / f"rot{r}_seed{s}" / f"pred_{stem}.npy")[te] for s in SEEDS], 0)
    assert not np.isnan(p).any()
    pred[lab] = p

hi = A > 15
rm = lambda p, k=slice(None): float(np.sqrt(np.mean((p[k] - A[k]) ** 2)))
series = [("NAIRAS V3", N, GRAY)] + [(k, v, c) for (k, v), c in zip(pred.items(), (TEAL, AMBER))]

fig, axs = plt.subplots(1, 6, figsize=(17, 4.3),
                        gridspec_kw={"width_ratios": [1, 1, 1, 0.05, 0.22, 1.3], "wspace": 0.28})
axs[4].axis("off")                                  # spacer between colorbar and panel (d)
ax = [axs[0], axs[1], axs[2], axs[5]]
lim = 45
for i, (lab, p, _) in enumerate(series):
    a = ax[i]
    hb = a.hexbin(A, p, gridsize=60, extent=(0, lim, 0, lim), norm=LogNorm(), cmap="viridis", mincnt=1)
    a.plot([0, lim], [0, lim], "--", color=RED, lw=1.2)
    a.axvline(15, color="#6B7280", lw=0.8, ls=":")
    a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
    a.set_xlabel("ARMAS measured dose (µSv/h)")
    a.set_ylabel("predicted dose (µSv/h)" if i == 0 else "")
    a.set_title(f"({'abc'[i]}) {lab}", fontsize=12, loc="left")
    a.text(0.03, 0.97, f"RMSE {rm(p):.2f}\nA>15: {rm(p, hi):.2f}", transform=a.transAxes, va="top",
           fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.85))
cb = fig.colorbar(hb, cax=axs[3]); cb.set_label("observations per cell")

a = ax[3]
edges = np.arange(0, 42, 2); mid = edges[:-1] + 1
for lab, p, c in series:
    e = A - p; q = np.full((len(mid), 3), np.nan)
    for j in range(len(mid)):
        k = (A >= edges[j]) & (A < edges[j + 1])
        if k.sum() >= 30:
            q[j] = np.percentile(e[k], [25, 50, 75])
    a.fill_between(mid, q[:, 0], q[:, 2], color=c, alpha=0.18, lw=0)
    a.plot(mid, q[:, 1], color=c if c != GRAY else "#4B5563", lw=2.2, label=lab)
a.axhline(0, color="black", lw=0.8); a.axvline(15, color="#6B7280", lw=0.8, ls=":")
a.set_xlabel("ARMAS measured dose (µSv/h)"); a.set_ylabel("error A − prediction (µSv/h)")
a.set_title("(d) Error by measured dose", fontsize=12, loc="left")
a.legend(frameon=False, fontsize=9.5, loc="upper left")
fig.savefig(OUT / "R4_best_vs_nairas.png", dpi=200, bbox_inches="tight")

for lab, p, _ in series:
    print(f"{lab:28s} RMSE {rm(p):.3f}  A>15 {rm(p, hi):.3f}  "
          f"under-predicted above 15: {np.mean(p[hi] < A[hi]) * 100:.0f}%  "
          f"median error A>15: {np.median((A - p)[hi]):+.2f}")
