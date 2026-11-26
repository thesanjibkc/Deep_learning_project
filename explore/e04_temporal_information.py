"""Exploration 4: does the history contain information?

(a) How much does each variable change inside its window?
    For every observation: (max - min over the window) / (overall std of the
    variable). 0 = flat history (nothing new to learn); large = the variable
    moves within the window. Median over observations, 5M (55 min) vs 1H (23 h).
(b,c) Lead-lag maps: Spearman correlation between the NAIRAS error r = A - N
    and each variable at each lag (lag 0 = current time), for 5M and 1H.
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

def load_all(cad):
    parts = [load(cad, p) for p in ("P1", "P2", "P3")]
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    X = np.concatenate([p[1] for p in parts])      # n, T, 35  (oldest -> current)
    return df, X, parts[0][2]

df5, X5, env = load_all("5M")
df1, X1, _ = load_all("1H")
assert (df5["ARMAS"].values == df1["ARMAS"].values).all()
r = df5["residual"].values
hi = df5["ARMAS"].values > 15

sd = X1.reshape(-1, 35).std(0) + 1e-12            # common scale for both cadences
move5 = np.median((X5.max(1) - X5.min(1)) / sd, 0)
move1 = np.median((X1.max(1) - X1.min(1)) / sd, 0)


def rank(a):
    return pd.DataFrame(a).rank().values


def lagcorr(X):
    n, T, V = X.shape
    rr = rank(r[:, None])[:, 0]; rr = (rr - rr.mean()) / rr.std()
    out = np.zeros((V, T))
    for t in range(T):
        z = rank(X[:, t, :]); z = (z - z.mean(0)) / (z.std(0) + 1e-12)
        out[:, t] = (z * rr[:, None]).mean(0)
    return out[:, ::-1]                             # column 0 = current time


C5, C1 = lagcorr(X5), lagcorr(X1)

order = np.argsort(-move1)
plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.6], wspace=.35)

a = fig.add_subplot(gs[0])
y = np.arange(35)
a.barh(y - .2, move5[order], .4, color="tab:orange", label="1 hour (5-min data)")
a.barh(y + .2, move1[order], .4, color="tab:blue", label="24 hours (hourly data)")
a.set_yticks(y); a.set_yticklabels([env[i] for i in order]); a.invert_yaxis()
a.set_xlabel("typical change inside the window\n(range / overall std)")
a.set_title("(a) Does the variable move\nwithin the history window?", loc="left")
a.legend(loc="lower right")

v = 0.35
for k, (C, T, title, xl) in enumerate([(C5, 12, "(b) 1-hour history (5-min steps)", "minutes before observation"),
                                       (C1, 24, "(c) 24-hour history (hourly steps)", "hours before observation")]):
    a = fig.add_subplot(gs[k + 1])
    im = a.imshow(C[order], aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    step = 5 if T == 12 else 1
    ticks = range(0, T, 2 if T == 12 else 4)
    a.set_xticks(list(ticks)); a.set_xticklabels([t * step for t in ticks])
    a.set_yticks(y); a.set_yticklabels([env[i] for i in order] if k == 0 else [])
    a.set_xlabel(xl); a.set_title(title + "\ncorrelation with NAIRAS error A − N", loc="left")
fig.colorbar(im, ax=a, label="Spearman correlation", shrink=.6)
fig.suptitle("Exploration 4 — Does the recent history carry information about the NAIRAS error?  "
             f"(n = {len(r):,})", fontweight="bold")
out = FIGS / "e04_temporal_information.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)

tab = pd.DataFrame({"move_1h": move5, "move_24h": move1,
                    "corr_now": C1[:, 0], "corr_best": np.abs(C1).max(1),
                    "best_lag_h": np.abs(C1).argmax(1)}, index=env)
print(tab.sort_values("move_24h", ascending=False).round(3).to_string())
print("\nflat in 1h (move<0.05):", [e for e, m in zip(env, move5) if m < .05])
gain = np.abs(C1).max(1) - np.abs(C1[:, 0])
print("vars where an earlier hour beats current by >0.03:",
      [(env[i], int(np.abs(C1[i]).argmax()), round(gain[i], 3)) for i in np.argsort(-gain) if gain[i] > .03])
