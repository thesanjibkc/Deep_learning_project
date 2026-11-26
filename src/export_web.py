"""Export the small JSON the interactive page (poster/web/) needs.

Usage: python3 src/export_web.py [temporal results root]   (default results/temporal_final)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import load

troot = Path(sys.argv[1] if len(sys.argv) > 1 else "results/temporal_final")
OUT = Path("poster/web/data.json")
d = load("1H")
A, N = d["A"], d["N"]
alt, rc = d["static"][:, 3] / 1000, d["static"][:, 6]
env = list(d["env"])

# 1) dose landscape: median ARMAS / NAIRAS by altitude x cutoff rigidity
alt_edges = [8, 9, 10, 11, 12, 13, 14.5]
rc_edges = [0, 1, 3, 6, 10, 16]
grid = []
for i in range(len(alt_edges) - 1):
    row = []
    for j in range(len(rc_edges) - 1):
        k = (alt >= alt_edges[i]) & (alt < alt_edges[i + 1]) & (rc >= rc_edges[j]) & (rc < rc_edges[j + 1])
        row.append(None if k.sum() < 30 else
                   {"A": round(float(np.median(A[k])), 1), "N": round(float(np.median(N[k])), 1),
                    "hi": round(float((A[k] > 15).mean() * 100)), "n": int(k.sum())})
    grid.append(row)

# 2) results
def runs(path):
    fr = []
    for f in sorted(Path(path).glob("rot*_seed*/metrics.csv")):
        rot, seed = f.parent.name.split("_")
        fr.append(pd.read_csv(f).assign(rotation=rot[3:], seed=int(seed[4:])))
    return pd.concat(fr) if fr else pd.DataFrame()

nice = {"NAIRAS V3": "NAIRAS V3", "MLP (residual)": "MLP", "MLP (direct)": "MLP (direct)",
        "FlatMLP (residual)": "Flat MLP", "CNN-MLP (residual)": "CNN–MLP", "TCN (residual)": "TCN",
        "Transformer (residual)": "Transformer", "Transformer (direct)": "Transformer (direct)"}
models = []

def add(df, window, rename=None):
    t = df[df["set"] == "test"]
    for m, g in t.groupby("model"):
        if m == "NAIRAS V3" and window != "none":
            continue
        name = rename or nice.get(m, m)
        per = g.groupby("rotation")[["rmse_all", "rmse_high"]].mean()
        models.append({"name": name, "window": window, "n": int(len(g)),
                       "all": round(g.rmse_all.mean(), 3), "all_sd": round(g.rmse_all.std(), 3),
                       "high": round(g.rmse_high.mean(), 3), "high_sd": round(g.rmse_high.std(), 3),
                       "rot": {r: [round(a, 3), round(h, 3)] for r, (a, h) in per.iterrows()}})

add(runs("results/basic"), "none")
for c, w in (("5M", "1h"), ("1H", "24h")):
    df = runs(troot / c)
    if len(df):
        add(df, w)
    dw = runs(Path(str(troot) + "_weighted") / c)
    if len(dw):
        add(dw[dw.model != "NAIRAS V3"], w, rename="Transformer + high-dose weight")

# 3) storm case study (Sept 9 2017), 6 variables, standardized over the dataset
t = pd.to_datetime(pd.Series(d["time"]))
i = int(np.where((t.dt.strftime("%Y-%m-%d %H:%M") == "2017-09-09 06:24"))[0][0])
show = ["NM_OULU", "Index_Kp", "Index_Dst", "SW_Bz", "SW_V", "SW_pressure"]
X = d["temporal"]
storm = {"A": round(float(A[i]), 1), "N": round(float(N[i]), 1),
         "alt_km": round(float(alt[i]), 1), "rc": round(float(rc[i]), 1),
         "vars": ["Neutron monitor", "Kp", "Dst", "Solar wind Bz", "Solar wind speed", "Solar wind pressure"],
         "z": [], "raw": []}
for v in show:
    j = env.index(v)
    mu, sd = X[:, :, j].mean(), X[:, :, j].std()
    storm["z"].append([round(float(x), 2) for x in (X[i, :, j] - mu) / sd])
    storm["raw"].append([round(float(x), 1) for x in X[i, :, j]])

json.dump({"n": int(len(A)), "alt_edges": alt_edges, "rc_edges": rc_edges, "grid": grid,
           "models": models, "xgboost": 3.671, "storm": storm,
           "status": "final" if "final" in str(troot) else "preview"},
          open(OUT, "w"), separators=(",", ":"))
print("wrote", OUT, OUT.stat().st_size, "bytes;", len(models), "model rows")
