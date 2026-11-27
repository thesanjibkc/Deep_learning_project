"""Occlusion analysis of the trained residual Transformer (the preliminary report's Fig. 5, redone on
the final models): replace an input by its training-set average and measure the change in test
RMSE. Positive = the model is worse without that input.

For one assignment (cadence 1H = 24-hour history, 5M = 1-hour history; seed 42), on the TEST partition:
  variables : each of the 35 space-weather variables (whole window) and each of the 8 static inputs
  lags      : each time step (all 35 variables at that step)
  cells     : each (variable, time step) pair
Occluded values are set to 0 in standardized units, which is the training mean.

Usage: python3 src/occlusion.py [cadence] [rotation] [seed]
Writes results/explain/occlusion_{vars,lags,cells}_{cadence}_rot{rot}_seed{seed}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import load, split, Scaler, STATIC
from models import Transformer

cad, rot, seed = (sys.argv[1:4] + ["1H", "123", "42"][len(sys.argv[1:4]):])[:3]
run = Path("results/temporal_final") / cad / f"rot{rot}_seed{seed}"
OUT = Path("results/explain"); OUT.mkdir(parents=True, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

# preprocessing identical to training (train_temporal.py)
d = load(cad); tr, va, te = (f"P{c}" for c in rot); m = split(d, tr, va, te)
A, N, env = d["A"], d["N"], list(d["env"])
seq = d["temporal"].copy()
logv = [i for i, n in enumerate(env) if n.startswith(("SXR", "Particles"))]
seq[:, :, logv] = np.log10(np.clip(seq[:, :, logv], 1e-12, None))
s_seq = Scaler().fit(seq[m["train"]]); cur = d["static"]; s_cur = Scaler().fit(cur[m["train"]])
y = A - N; mu, sd = y[m["train"]].mean(), y[m["train"]].std()
T = seq.shape[1]
model = Transformer(T, 35, cur.shape[1], width=16, p=0.3).to(dev)
model.load_state_dict(torch.load(run / "Transformer_residual.pt", map_location=dev)); model.eval()
ti = np.where(m["test"])[0]
Zs = torch.tensor(s_seq(seq[ti]), dtype=torch.float32, device=dev)
Zc = torch.tensor(s_cur(cur[ti]), dtype=torch.float32, device=dev)
At, Nt = A[ti], N[ti]; hi = At > 15


def rmse(xs, xc):
    with torch.no_grad():
        out = torch.cat([model(xs[i:i + 8192], xc[i:i + 8192]) for i in range(0, len(xs), 8192)]).cpu().numpy()
    p = Nt + out * sd + mu
    return float(np.sqrt(np.mean((p - At) ** 2))), float(np.sqrt(np.mean((p[hi] - At[hi]) ** 2)))


b_all, b_hi = rmse(Zs, Zc)
vars_, lags, cells = [], [], []
for j, name in enumerate(env):
    xs = Zs.clone(); xs[:, :, j] = 0
    a, h = rmse(xs, Zc); vars_.append({"variable": name, "kind": "space weather", "d_all": a - b_all, "d_high": h - b_hi})
for j, name in enumerate(STATIC):
    xc = Zc.clone(); xc[:, j] = 0
    a, h = rmse(Zs, xc); vars_.append({"variable": name, "kind": "flight (current)", "d_all": a - b_all, "d_high": h - b_hi})
for t in range(T):
    xs = Zs.clone(); xs[:, t, :] = 0
    a, h = rmse(xs, Zc); lags.append({"steps_before": T - 1 - t, "d_all": a - b_all, "d_high": h - b_hi})
for j, name in enumerate(env):
    for t in range(T):
        xs = Zs.clone(); xs[:, t, j] = 0
        a, h = rmse(xs, Zc); cells.append({"variable": name, "steps_before": T - 1 - t, "d_all": a - b_all, "d_high": h - b_hi})

tag = f"{cad}_rot{rot}_seed{seed}"
pd.DataFrame(vars_).to_csv(OUT / f"occlusion_vars_{tag}.csv", index=False)
pd.DataFrame(lags).to_csv(OUT / f"occlusion_lags_{tag}.csv", index=False)
pd.DataFrame(cells).to_csv(OUT / f"occlusion_cells_{tag}.csv", index=False)
print(f"{tag}: base RMSE {b_all:.3f} / A>15 {b_hi:.3f}")
print(pd.DataFrame(vars_).sort_values("d_all", ascending=False).head(8).round(4).to_string(index=False))
