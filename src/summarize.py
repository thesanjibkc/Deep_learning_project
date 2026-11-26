"""Combine all rotations into one table: mean ± std of the TEST metrics.

Usage: python3 src/summarize.py results/basic
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/basic")
frames = []
for f in sorted(root.glob("rot*_seed*/metrics.csv")):
    df = pd.read_csv(f)
    rot, seed = f.parent.name.split("_")
    frames.append(df.assign(rotation=rot[3:], seed=int(seed[4:])))
R = pd.concat(frames)
R = R[~R["model"].str.startswith("Linear")]
R.to_csv(root / "all_runs.csv", index=False)

T = R[R["set"] == "test"]
cols = ["rmse_all", "rmse_high", "rmse_low", "mae_all", "r2_all", "bias_high"]
per_rot = T.pivot_table(index="rotation", columns="model", values="rmse_all").round(3)
summ = T.groupby("model")[cols].agg(["mean", "std"])
nairas = summ.loc["NAIRAS V3", ("rmse_all", "mean")]
nairas_hi = summ.loc["NAIRAS V3", ("rmse_high", "mean")]

lines = []
for model, row in summ.iterrows():
    f = lambda c: f"{row[(c, 'mean')]:.3f} ± {row[(c, 'std')]:.3f}"
    gain = 100 * (1 - row[("rmse_all", "mean")] / nairas)
    gain_hi = 100 * (1 - row[("rmse_high", "mean")] / nairas_hi)
    lines.append({"model": model, "RMSE all": f("rmse_all"), "RMSE A>15": f("rmse_high"),
                  "RMSE A<=15": f("rmse_low"), "R2": f("r2_all"),
                  "gain all %": round(gain, 1), "gain A>15 %": round(gain_hi, 1)})
out = pd.DataFrame(lines).sort_values("RMSE all")
out.to_csv(root / "summary.csv", index=False)
print(f"Test metrics over {T['rotation'].nunique()} rotations x {T['seed'].nunique()} seed(s)\n")
print(out.to_string(index=False))
print("\nTest RMSE (all) per rotation:\n", per_rot.to_string())
