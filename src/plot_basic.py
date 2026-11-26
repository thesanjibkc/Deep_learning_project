"""Plot learning curves and test RMSE for the basic MLP run."""
import json, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/basic/rot123_seed42")
H = json.load(open(run / "histories.json"))
R = pd.read_csv(run / "metrics.csv")
R = R[~R["model"].str.startswith("Linear")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, (name, h) in zip(ax[:2], H.items()):
    h = pd.DataFrame(h)
    a.plot(h.epoch, h.train_rmse, "o-", ms=3, label="train (P1)")
    a.plot(h.epoch, h.val_rmse, "o-", ms=3, label="validation (P2)")
    b = h.val_rmse.idxmin()
    a.axvline(h.epoch[b], color="gray", ls=":", label=f"best epoch {h.epoch[b]}")
    a.set(title=f"{name}: learning curve", xlabel="epoch", ylabel="RMSE (µSv/h)")
    a.legend()
t = R[R["set"] == "test"].set_index("model")
x = range(len(t))
ax[2].bar([i - .2 for i in x], t.rmse_all, .4, label="all", color="tab:blue")
ax[2].bar([i + .2 for i in x], t.rmse_high, .4, label="A > 15", color="tab:red")
for i, (a_, h_) in enumerate(zip(t.rmse_all, t.rmse_high)):
    ax[2].text(i - .2, a_ + .05, f"{a_:.2f}", ha="center", fontsize=8)
    ax[2].text(i + .2, h_ + .05, f"{h_:.2f}", ha="center", fontsize=8)
ax[2].set_xticks(list(x)); ax[2].set_xticklabels(t.index, rotation=15)
ax[2].set(title="Test RMSE on P3 (lower is better)", ylabel="RMSE (µSv/h)")
ax[2].legend()
fig.tight_layout()
out = Path("figures/results") / f"basic_{run.name}.png"
fig.savefig(out, dpi=140); print("saved", out)
