"""E1: why is the error large? (24 h residual Transformer, train P1 / validate P2 / test P3)

(a) what the model's correction relies on: RMSE rise when one feature group is shuffled
(b) what explains the error: share of error variance explained by each grouping,
    NAIRAS error vs the model's remaining error
(c) where the remaining error lives: within one flight vs between flights vs aircraft
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, AMBER, TEAL, GRAY, RED, MUTED = "#14213D", "#E8892B", "#24907F", "#9AA3AF", "#C8553D", "#6B7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})
# averages over the six train/test assignments (seed 42); residual_shares_* comes from
# src/paper_figures.py (residual_shares), importance/remaining_error from src/explain_errors.py
EX = Path("results/explain")
imp = (pd.concat([pd.read_csv(f) for f in sorted(EX.glob("importance_groups_1H_rot*_seed42.csv"))])
       .groupby(["group", "kind"], as_index=False)["d_all"].mean().sort_values("d_all"))
left = (pd.concat([pd.read_csv(f) for f in sorted(EX.glob("remaining_error_1H_rot*_seed42.csv"))])
        .groupby("grouping", as_index=False, sort=False).mean(numeric_only=True))
sh = pd.read_csv(EX / "residual_shares_1H_seed42.csv").groupby("kind").mean(numeric_only=True)
FLIGHT_REM, FLIGHT_NAI = sh.loc["model", "Flight"], sh.loc["NAIRAS", "Flight"]
AIR_REM, AIR_NAI, AIR_NULL = sh.loc["model", "Aircraft"], sh.loc["NAIRAS", "Aircraft"], sh.loc["model", "Aircraft chance"]

fig = plt.figure(figsize=(17, 7.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 0.8], wspace=0.62, top=0.8, bottom=0.12)

a = fig.add_subplot(gs[0])
col = [TEAL if k == "flight (current)" else AMBER for k in imp["kind"]]
a.barh(imp["group"], imp["d_all"], color=col)
for i, v in enumerate(imp["d_all"]):
    a.text(v + 0.01, i, f"+{v:.2f}", va="center", fontsize=10)
a.set_xlabel("RMSE increase when shuffled (µSv/h)", labelpad=2)
a.set_title("(a) What the correction relies on", loc="left", fontsize=14, color=NAVY, fontweight="bold")
a.legend(bbox_to_anchor=(0.5, -0.13), ncol=2, handles=[plt.Rectangle((0, 0), 1, 1, color=TEAL), plt.Rectangle((0, 0), 1, 1, color=AMBER)],
         labels=["flight (current)", "space weather (24 h history)"], frameon=False, loc="upper center", fontsize=10)

b = fig.add_subplot(gs[1])
L = left.iloc[::-1]
y = range(len(L))
b.barh([i + 0.2 for i in y], L["NAIRAS error"] * 100, 0.4, color=GRAY, label="NAIRAS error")
b.barh([i - 0.2 for i in y], L["remaining error"] * 100, 0.4, color=AMBER, label="error after the model")
b.axvline(AIR_NULL * 100, color=MUTED, ls=":", lw=1)
b.text(AIR_NULL * 100 + 0.2, -0.75, "chance level for aircraft", fontsize=9, color=MUTED)
b.set_yticks(list(y)); b.set_yticklabels(L["grouping"])
b.set_xlabel("share of error variance explained (%)")
b.set_title("(b) What the error still depends on", loc="left", fontsize=14, color=NAVY, fontweight="bold")
b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=10)

c = fig.add_subplot(gs[2])
within_n, within_r = 1 - FLIGHT_NAI, 1 - FLIGHT_REM
for x, (w, f, ac, lab) in enumerate([(within_n, FLIGHT_NAI - AIR_NAI, AIR_NAI, "NAIRAS"),
                                     (within_r, FLIGHT_REM - AIR_REM, AIR_REM, "after model")]):
    c.bar(x, w * 100, color="#D1D5DB")
    c.bar(x, f * 100, bottom=w * 100, color=GRAY)
    c.bar(x, ac * 100, bottom=(w + f) * 100, color=RED)
    for v, bot, t in ((w, 0, "within one flight"), (f, w, "between flights"), (ac, w + f, "aircraft")):
        c.text(x, (bot + v / 2) * 100, f"{v*100:.0f}%", ha="center", va="center", fontsize=10,
               color="white" if t == "aircraft" else NAVY)
c.set_xticks([0, 1]); c.set_xticklabels(["NAIRAS\nerror", "error after\nthe model"])
c.set_ylabel("share of error variance (%)")
c.set_title("(c) Where the error lives", loc="left", fontsize=14, color=NAVY, fontweight="bold")
c.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=k) for k in ("#D1D5DB", GRAY, RED)],
         labels=["within one flight", "between flights", "aircraft (instrument)"], frameon=False,
         fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18))

fig.suptitle("Why is the error large?", x=0.02, y=0.98, ha="left", fontsize=22, color=NAVY, fontweight="bold")
fig.text(0.02, 0.905, "The model removes the error tied to altitude, shielding and storms. What remains is mostly minute-to-minute noise within a flight, plus an aircraft effect.",
         fontsize=13, color=NAVY)
fig.text(0.02, 0.87, "24-hour residual Transformer, averaged over the six train/test splits. Aircraft ID is never a model input.",
         fontsize=12, color=MUTED)
out = Path("figures/results/E1_why_error.png")
fig.savefig(out, dpi=170, bbox_inches="tight"); print("saved", out)
