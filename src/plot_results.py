"""Easy-to-read result figures (print-friendly, same palette as the slides).

R1 leaderboard     : test RMSE for every model, overall and A > 15
R2 history window  : 1-hour vs 24-hour history for each temporal model
R3 robustness      : every train/validation/test assignment as one dot

Usage: python3 src/plot_results.py <temporal results root> [label]
       e.g. python3 src/plot_results.py results/temporal_final "final"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, AMBER, TEAL, GRAY, RED = "#14213D", "#F2A33A", "#2A9D8F", "#9CA3AF", "#C8553D"
XGB = 3.671                                         # K C et al. 2025, cited
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#6B7280"})

troot = Path(sys.argv[1] if len(sys.argv) > 1 else "results/temporal")
label = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = Path("figures/results"); OUT.mkdir(parents=True, exist_ok=True)


def runs(path):
    frames = []
    for f in sorted(Path(path).glob("rot*_seed*/metrics.csv")):
        rot, seed = f.parent.name.split("_")
        frames.append(pd.read_csv(f).assign(rotation=rot[3:], seed=int(seed[4:])))
    return pd.concat(frames)


B = runs("results/basic").assign(history="none")
T = {c: runs(troot / c).assign(history=c) for c in ("5M", "1H") if (troot / c).exists()}
nice = {"MLP (residual)": "MLP", "MLP (direct)": "MLP (direct)", "FlatMLP (residual)": "Flat MLP",
        "CNN-MLP (residual)": "CNN–MLP", "TCN (residual)": "TCN",
        "Transformer (residual)": "Transformer", "Transformer (direct)": "Transformer (direct)"}


def test(df):
    return df[df["set"] == "test"]


rows = []
for name, g in test(B).groupby("model"):
    if name == "NAIRAS V3":
        rows.append(("NAIRAS V3", "physics", g)); continue
    rows.append((nice[name], "no history", g))
for c, df in T.items():
    win = "1 h history" if c == "5M" else "24 h history"
    for name, g in test(df).groupby("model"):
        if name != "NAIRAS V3":
            rows.append((f"{nice[name]} · {win}", win, g))

W = {c: runs(Path(str(troot) + "_weighted") / c) for c in ("5M", "1H") if (Path(str(troot) + "_weighted") / c).exists()}
for c, df in W.items():
    win = "1 h history" if c == "5M" else "24 h history"
    g = test(df)[test(df).model == "Transformer (residual)"]
    rows.append((f"Transformer + high-dose weight · {win}", win, g))

S = pd.DataFrame([{"name": n, "kind": k, "all": g["rmse_all"].mean(), "all_sd": g["rmse_all"].std(),
                   "high": g["rmse_high"].mean(), "high_sd": g["rmse_high"].std(), "n": len(g)}
                  for n, k, g in rows])
color = {"physics": GRAY, "no history": TEAL, "1 h history": "#E9C46A", "24 h history": AMBER}
nairas = S.loc[S.name == "NAIRAS V3"].iloc[0]

# ---------- R1 leaderboard ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), sharey=True)
S1 = S.sort_values("all", ascending=False).reset_index(drop=True)
y = np.arange(len(S1))
for ax, col, title in ((axes[0], "all", "All flights"), (axes[1], "high", "High dose (ARMAS > 15 µSv/h)")):
    ax.barh(y, S1[col], xerr=S1[col + "_sd"], color=[color[k] for k in S1.kind],
            edgecolor="white", error_kw=dict(ecolor="#6B7280", lw=1, capsize=2))
    ax.axvline(nairas[col], color=NAVY, ls="--", lw=1.5)
    ax.text(nairas[col], len(S1) - 0.3, " NAIRAS", color=NAVY, fontsize=11, va="bottom")
    if col == "all":
        ax.axvline(XGB, color=RED, ls=":", lw=1.5)
        ax.text(XGB, -0.9, "XGBoost 2025 ", color=RED, fontsize=10, ha="right")
    for i, v in enumerate(S1[col]):
        ax.text(v + S1[col + "_sd"][i] + 0.02, i, f"{v:.2f}", va="center", fontsize=10)
    lo = S1[col].min() - 0.35
    ax.set_xlim(lo, S1[col].max() + 0.3)
    ax.set_title(title, fontsize=15, color=NAVY, fontweight="bold", loc="left")
    ax.set_xlabel("test RMSE (µSv/h)  —  shorter is better")
axes[0].set_yticks(y); axes[0].set_yticklabels(S1.name)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color.values()]
fig.legend(handles, list(color.keys()), loc="lower center", ncol=4, frameon=False, fontsize=12)
fig.suptitle(f"Which model predicts the dose best?  {label}", fontsize=20, color=NAVY,
             fontweight="bold", x=0.02, ha="left")
fig.text(0.02, 0.915, f"Mean ± std over {int(S.n.min())}–{int(S.n.max())} runs per model (6 train/validation/test "
         "assignments × seeds). Each run is tested on flights it never saw.",
         fontsize=12, color="#6B7280")
fig.tight_layout(rect=(0, 0.05, 1, 0.9))
fig.savefig(OUT / "R1_leaderboard.png", dpi=180); plt.close(fig)

# ---------- R2 1 h vs 24 h ----------
if len(T) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    models = ["Flat MLP", "CNN–MLP", "TCN", "Transformer", "Transformer (direct)"]
    for ax, col, title in ((axes[0], "all", "All flights"), (axes[1], "high", "High dose (ARMAS > 15)")):
        for i, mname in enumerate(models):
            a = S.loc[S.name == f"{mname} · 1 h history", col].values
            b = S.loc[S.name == f"{mname} · 24 h history", col].values
            if not len(a) or not len(b):
                continue
            ax.plot([a[0], b[0]], [i, i], color="#D1D5DB", lw=4, zorder=1)
            ax.scatter(a, i, s=140, color="#E9C46A", zorder=2, label="1 hour" if i == 0 else None)
            ax.scatter(b, i, s=140, color=AMBER, edgecolor=NAVY, zorder=3, label="24 hours" if i == 0 else None)
        ax.axvline(nairas[col], color=NAVY, ls="--", lw=1.2)
        mlp = S.loc[S.name == "MLP", col].values[0]
        ax.axvline(mlp, color=TEAL, ls="-", lw=1.5)
        ax.text(mlp, -0.7, "MLP, no history ", color=TEAL, fontsize=10, ha="right")
        ax.text(nairas[col], -0.7, " NAIRAS", color=NAVY, fontsize=10)
        ax.set_title(title, loc="left", fontsize=15, color=NAVY, fontweight="bold")
        ax.set_xlabel("test RMSE (µSv/h) — further left is better")
    axes[0].set_yticks(range(len(models))); axes[0].set_yticklabels(models); axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle(f"Does a longer look back help?  1-hour vs 24-hour history  {label}", fontsize=18,
                 color=NAVY, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "R2_history_window.png", dpi=180); plt.close(fig)

# ---------- R3 robustness ----------
pick = [("NAIRAS V3", test(B), "NAIRAS V3", GRAY), ("MLP (residual)", test(B), "MLP", TEAL)]
for c, df in T.items():
    pick.append(("Transformer (residual)", test(df),
                 f"Transformer · {'1 h' if c == '5M' else '24 h'}", "#E9C46A" if c == "5M" else AMBER))
fig, ax = plt.subplots(figsize=(13, 5.5))
rots = ["123", "132", "213", "231", "312", "321"]
for j, (m, df, lab, col) in enumerate(pick):
    g = df[df.model == m].groupby("rotation")["rmse_all"].mean().reindex(rots)
    x = np.arange(len(rots)) + (j - (len(pick) - 1) / 2) * 0.17
    ax.scatter(x, g.values, s=110, color=col, edgecolor=NAVY if "Transformer" in lab else "none",
               label=lab, zorder=3)
ax.set_xticks(range(len(rots)))
ax.set_xticklabels([f"train P{r[0]}\nval P{r[1]}\ntest P{r[2]}" for r in rots], fontsize=10)
ax.set_ylabel("test RMSE (µSv/h)")
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
ax.grid(axis="y", color="#E5E7EB")
fig.suptitle(f"Is the result robust?  Every train / validation / test assignment  {label}",
             fontsize=17, color=NAVY, fontweight="bold", x=0.02, ha="left", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "R3_robustness.png", dpi=180, bbox_inches="tight"); plt.close(fig)
print("saved R1, R2, R3 to", OUT)
print(S.sort_values("all")[["name", "all", "high", "n"]].round(3).to_string(index=False))
