"""Exploration 1: where does the physics model (NAIRAS V3) fail?

Uses ALL data (P1+P2+P3) to describe the dataset. Descriptive only:
modelling choices (features, thresholds, hyperparameters) must still be made
on the training/validation partitions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from load import load, FIGS

df = pd.concat([load("5M", p)[0] for p in ("P1", "P2", "P3")], ignore_index=True)
A, N, r = df["ARMAS"].values, df["NAIRASV3"].values, df["residual"].values
hi = A > 15

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

# (a) NAIRAS vs ARMAS
a = ax[0, 0]
hb = a.hexbin(A, N, gridsize=70, bins="log", cmap="viridis", mincnt=1)
m = max(A.max(), N.max())
a.plot([0, m], [0, m], "k--", lw=1.2, label="perfect (1:1)")
a.axvline(15, color="tab:red", lw=1, ls=":", label="A = 15 µSv/h")
a.set(xlabel="ARMAS measured A (µSv/h)", ylabel="NAIRAS V3 model N (µSv/h)",
      title="(a) Physics model vs measurement")
a.legend(loc="upper left", fontsize=8)
fig.colorbar(hb, ax=a, label="log10(count)")

# (b) distribution of A
a = ax[0, 1]
bins = np.linspace(0, A.max(), 80)
a.hist(A[~hi], bins=bins, color="tab:blue", alpha=.8, label=f"A ≤ 15  ({(~hi).mean():.0%})")
a.hist(A[hi], bins=bins, color="tab:red", alpha=.8, label=f"A > 15  ({hi.mean():.0%})")
a.set(xlabel="ARMAS A (µSv/h)", ylabel="observations", title="(b) Target distribution (all data)")
a.legend()


def binned(ax_, x, y, edges, color, label):
    mids, med, q1, q3 = [], [], [], []
    for lo, up in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < up)
        if s.sum() < 30:
            continue
        mids.append((lo + up) / 2); med.append(np.median(y[s]))
        q1.append(np.percentile(y[s], 25)); q3.append(np.percentile(y[s], 75))
    ax_.fill_between(mids, q1, q3, color=color, alpha=.25)
    ax_.plot(mids, med, "o-", color=color, ms=3, label=label)


edges = np.arange(0, 42, 2)
# (c) residual vs MEASURED dose
a = ax[1, 0]
binned(a, A, r, edges, "tab:red", "median, IQR band")
a.axhline(0, color="k", lw=.8)
a.axvline(15, color="tab:red", lw=1, ls=":")
a.set(xlabel="ARMAS A (µSv/h)  — binned by the TRUTH",
      ylabel="residual r = A − N (µSv/h)", title="(c) Error vs measured dose")
a.legend()

# (d) residual vs NAIRAS estimate
a = ax[1, 1]
binned(a, N, r, edges, "tab:purple", "median, IQR band")
a.axhline(0, color="k", lw=.8)
a.set(xlabel="NAIRAS N (µSv/h)  — binned by the PREDICTION",
      ylabel="residual r = A − N (µSv/h)", title="(d) Error vs physics estimate")
a.legend()

fig.suptitle("Exploration 1 — Where does NAIRAS V3 fail?  (5-min data, all partitions P1+P2+P3, "
             f"n = {len(df):,})", fontweight="bold")
fig.tight_layout()
FIGS.mkdir(parents=True, exist_ok=True)
out = FIGS / "e01_where_nairas_fails.png"
fig.savefig(out, dpi=150)
print("saved", out)

# numbers we will discuss
rm = lambda e: np.sqrt(np.mean(e ** 2))
print(f"RMSE all={rm(r):.3f}  A<=15={rm(r[~hi]):.3f}  A>15={rm(r[hi]):.3f}")
print(f"mean residual all={r.mean():.3f}  A<=15={r[~hi].mean():.3f}  A>15={r[hi].mean():.3f}")
print(f"share of A>15 where N<A: {(N[hi] < A[hi]).mean():.1%}")
print(f"corr(A,N)={np.corrcoef(A, N)[0,1]:.3f}")
print(f"N>15 count={int((N>15).sum())}, mean residual when N>15: {r[N>15].mean():.3f}")

print("\nper partition:")
for p, g in df.groupby("partition"):
    e = g["residual"].values; h = g["ARMAS"].values > 15
    print(f"  {p}: n={len(g):,}  A>15={h.sum():,} ({h.mean():.1%})  RMSE all={rm(e):.3f}  A>15={rm(e[h]):.3f}  mean r(A>15)={e[h].mean():.3f}")
