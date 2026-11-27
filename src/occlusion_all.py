"""Occlusion for every trained residual network (complements src/occlusion.py, which also maps
variable x hour for the Transformer). An input is replaced by its training mean (0 in standardized
units) and the change in test RMSE is recorded; positive = the model is worse without it.

Models: MLP (no history, 43 current inputs), and for the chosen history Flat MLP, CNN-MLP, TCN and
Transformer (35 space-weather variables over the window + 8 static inputs; also each time step).

Usage: python3 src/occlusion_all.py [cadence] [rotation] [seed]     (cadence 1H = 24 h, 5M = 1 h)
Writes results/explain/occl_all_{vars,lags}_{cadence}_rot{rot}_seed{seed}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from data import load, split, Scaler, STATIC
from models import MODELS

cad, rot, seed = (sys.argv[1:4] + ["1H", "123", "42"][len(sys.argv[1:4]):])[:3]
OUT = Path("results/explain"); OUT.mkdir(parents=True, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)

d = load(cad); m = split(d, *(f"P{c}" for c in rot))
A, N, env = d["A"], d["N"], list(d["env"])
ti = np.where(m["test"])[0]; At, Nt = A[ti], N[ti]; hi = At > 15
y = A - N; mu, sd = y[m["train"]].mean(), y[m["train"]].std()


def score(pred_std):
    p = Nt + pred_std * sd + mu
    return float(np.sqrt(np.mean((p - At) ** 2))), float(np.sqrt(np.mean((p[hi] - At[hi]) ** 2)))


rows_v, rows_l = [], []

# ---- MLP (no history): 8 static + 35 current values, as train_basic.py
class BasicMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


d5 = load("5M"); xb = d5["current"]; sb = Scaler().fit(xb[m["train"]]); Xb = f32(sb(xb[ti]))
mlp = BasicMLP(43).to(dev)
mlp.load_state_dict(torch.load(Path("results/basic") / f"rot{rot}_seed{seed}" / "MLP_(residual).pt", map_location=dev)); mlp.eval()
run_mlp = lambda X: mlp(X).squeeze(-1).detach().cpu().numpy()
with torch.no_grad():
    b = score(run_mlp(Xb))
    names = STATIC + list(env)
    for j, name in enumerate(names):
        X = Xb.clone(); X[:, j] = 0; a, h = score(run_mlp(X))
        rows_v.append({"model": "MLP", "variable": name, "kind": "flight (current)" if j < 8 else "space weather",
                       "d_all": a - b[0], "d_high": h - b[1]})

# ---- temporal networks
seq = d["temporal"].copy()
logv = [i for i, n in enumerate(env) if n.startswith(("SXR", "Particles"))]
seq[:, :, logv] = np.log10(np.clip(seq[:, :, logv], 1e-12, None))
s_seq = Scaler().fit(seq[m["train"]]); s_cur = Scaler().fit(d["static"][m["train"]])
Zs, Zc = f32(s_seq(seq[ti])), f32(s_cur(d["static"][ti])); T = seq.shape[1]
run_dir = Path("results/temporal_final") / cad / f"rot{rot}_seed{seed}"
for arch, label in (("FlatMLP", "Flat MLP"), ("CNN-MLP", "CNN–MLP"), ("TCN", "TCN"), ("Transformer", "Transformer")):
    net = MODELS[arch](T, 35, 8, width=16, p=0.3).to(dev)
    net.load_state_dict(torch.load(run_dir / f"{arch}_residual.pt", map_location=dev)); net.eval()
    run = lambda xs, xc: torch.cat([net(xs[i:i + 8192], xc[i:i + 8192]) for i in range(0, len(xs), 8192)]).cpu().numpy()
    with torch.no_grad():
        b = score(run(Zs, Zc))
        for j, name in enumerate(env):
            xs = Zs.clone(); xs[:, :, j] = 0; a, h = score(run(xs, Zc))
            rows_v.append({"model": label, "variable": name, "kind": "space weather", "d_all": a - b[0], "d_high": h - b[1]})
        for j, name in enumerate(STATIC):
            xc = Zc.clone(); xc[:, j] = 0; a, h = score(run(Zs, xc))
            rows_v.append({"model": label, "variable": name, "kind": "flight (current)", "d_all": a - b[0], "d_high": h - b[1]})
        for t in range(T):
            xs = Zs.clone(); xs[:, t, :] = 0; a, h = score(run(xs, Zc))
            rows_l.append({"model": label, "steps_before": T - 1 - t, "d_all": a - b[0], "d_high": h - b[1]})

tag = f"{cad}_rot{rot}_seed{seed}"
pd.DataFrame(rows_v).to_csv(OUT / f"occl_all_vars_{tag}.csv", index=False)
pd.DataFrame(rows_l).to_csv(OUT / f"occl_all_lags_{tag}.csv", index=False)
print(tag, "done")
