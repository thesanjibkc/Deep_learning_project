"""Training-evolution figure: how the Flat MLP and the Transformer learn (train P1, validate P2, 24 h data).

T1_training_evolution.png
  (a) learning curves for both models
  (b) Transformer attention over the 24 hours for the Sept 9 2017 storm flight, every epoch
  (c) attention at the best epoch vs the storm's Kp, for the same flight
  (d) Flat MLP: how strongly its first layer uses each hour, every epoch
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from data import load

NAVY, AMBER, TEAL, RED, MUTED = "#14213D", "#E8892B", "#24907F", "#C8553D", "#6B7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})
z = np.load("results/tracking/tracking.npz")
tc, fc = z["trf_curve"], z["flat_curve"]
best = int(tc[:, 1].argmin())
hours_before = np.arange(23, -1, -1)                      # column 0 = 23 h before ... 23 = now

d = load("1H")
import pandas as pd
t = pd.to_datetime(pd.Series(d["time"]))
i = int(np.where(t.dt.strftime("%Y-%m-%d %H:%M") == "2017-09-09 06:24")[0][0])
kp = d["temporal"][i, :, list(d["env"]).index("Index_Kp")] / 10

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)

a = fig.add_subplot(gs[0, 0])
ep = np.arange(len(tc))
a.plot(ep, fc[:, 0], color=RED, ls="--", lw=1.5, label="Flat MLP, train")
a.plot(ep, fc[:, 1], color=RED, lw=2.5, label="Flat MLP, validation")
a.plot(ep, tc[:, 0], color=AMBER, ls="--", lw=1.5, label="Transformer, train")
a.plot(ep, tc[:, 1], color=AMBER, lw=2.5, label="Transformer, validation")
a.axvline(best, color=MUTED, ls=":"); a.text(best + 0.4, 3.9, f"Transformer best\nepoch {best}", fontsize=10, color=MUTED)
a.set(xlabel="epoch", ylabel="RMSE (µSv/h)", ylim=(2.6, 4.05))
a.set_title("(a) Learning curves: Flat MLP memorises",
            loc="left", fontsize=13, color=NAVY, fontweight="bold")
a.legend(frameon=False, fontsize=10, loc="lower left")

b = fig.add_subplot(gs[0, 1])
im = b.imshow(z["trf_att_storm"], aspect="auto", cmap="YlOrBr", origin="lower",
              extent=(-0.5, 23.5, -0.5, len(tc) - 0.5))
b.axhline(best, color=NAVY, lw=1, ls=":")
b.set_xticks(range(0, 24, 3)); b.set_xticklabels(hours_before[::3])
b.set(xlabel="hours before the flight", ylabel="epoch")
b.set_title("(b) Attention forming, epoch by epoch", loc="left", fontsize=13,
            color=NAVY, fontweight="bold")
fig.colorbar(im, ax=b, label="attention weight", shrink=.85)

c = fig.add_subplot(gs[1, 0])
w = z["trf_att_storm"][best]
c.bar(range(24), w, color=AMBER, label=f"attention (epoch {best})")
c.axhline(1 / 24, color=MUTED, ls="--", lw=1); c.text(0, 1 / 24 + 0.002, "equal weights", fontsize=10, color=MUTED)
c2 = c.twinx(); c2.step(range(24), kp, where="mid", color=NAVY, lw=2, label="Kp")
c2.set_ylabel("Kp index", color=NAVY); c2.spines["top"].set_visible(False)
c.set_xticks(range(0, 24, 3)); c.set_xticklabels(hours_before[::3])
c.set(xlabel="hours before the flight", ylabel="attention weight")
c.set_title("(c) Learned attention vs the storm's Kp", loc="left", fontsize=13,
            color=NAVY, fontweight="bold")

e = fig.add_subplot(gs[1, 1])
wm = z["flat_wmap"].sum(2)
wm = wm / wm.sum(1, keepdims=True)
im2 = e.imshow(wm, aspect="auto", cmap="YlOrBr", origin="lower", extent=(-0.5, 23.5, -0.5, len(fc) - 0.5),
               vmin=z["trf_att_storm"].min(), vmax=z["trf_att_storm"].max())
e.set_xticks(range(0, 24, 3)); e.set_xticklabels(hours_before[::3])
e.set(xlabel="hours before the flight", ylabel="epoch")
e.set_title("(d) Flat MLP: weight share per hour", loc="left", fontsize=13,
            color=NAVY, fontweight="bold")
fig.colorbar(im2, ax=e, label="share of weight", shrink=.85)

fig.suptitle("Watching two networks learn from 24 hours of space weather", x=0.02, ha="left",
             fontsize=20, color=NAVY, fontweight="bold")
fig.text(0.02, 0.935, "Sept 9 2017 storm flight (panels b, c). Train P1, validate P2, one seed. Same 24-hour input; same width, dropout and weight decay.",
         fontsize=12, color=MUTED)
out = Path("figures/results/T1_training_evolution.png")
fig.savefig(out, dpi=170, bbox_inches="tight"); print("saved", out)
