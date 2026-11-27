"""Whose HISTORY matters?  Separates the value of a variable's past from its current value.

For the trained 24-hour residual Transformer (test partition of one run), and for each
space-weather group (plus all 35 variables together):
  - freeze:  replace the T-1 past steps by the current (last) value, i.e. the model sees a
             flat history but the correct present. The RMSE rise is the value of the
             variable's change over the window.
  - shuffle: give each observation the past steps of another random test observation,
             keeping its own current value.
src/explain_errors.py shuffles whole windows (present + past), so it cannot separate the two.

Usage: python3 src/history_importance.py [cadence] [rotation] [seed] [results root]
Writes results/explain/history_importance_{cadence}_rot{rot}_seed{seed}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import load, split, Scaler
from models import Transformer

cad, rot, seed = (sys.argv[1:4] + ["1H", "123", "42"][len(sys.argv[1:4]):])[:3]
root = Path(sys.argv[4] if len(sys.argv) > 4 else "results/temporal_final")
run = root / cad / f"rot{rot}_seed{seed}"
OUT = Path("results/explain"); OUT.mkdir(parents=True, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); rng = np.random.default_rng(0)

# --- same preprocessing as src/explain_errors.py
d = load(cad); tr, va, te = (f"P{c}" for c in rot); m = split(d, tr, va, te)
A, N, env = d["A"], d["N"], list(d["env"])
seq = d["temporal"].copy()
logv = [i for i, n in enumerate(env) if n.startswith(("SXR", "Particles"))]
seq[:, :, logv] = np.log10(np.clip(seq[:, :, logv], 1e-12, None))
s_seq = Scaler().fit(seq[m["train"]])
cur = np.concatenate([d["static"], N[:, None]], 1)
s_cur = Scaler().fit(cur[m["train"]])
y = A - N; mu, sd = y[m["train"]].mean(), y[m["train"]].std()

T = seq.shape[1]
model = Transformer(T, 35, cur.shape[1], width=16, p=0.3).to(dev)
model.load_state_dict(torch.load(run / "Transformer_residual.pt", map_location=dev))
model.eval()
ti = np.where(m["test"])[0]
Xs0, Xc0, At, Nt = seq[ti], cur[ti], A[ti], N[ti]
hi = At > 15


def predict(xs):
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), 4096):
            a = torch.tensor(s_seq(xs[i:i + 4096]), dtype=torch.float32, device=dev)
            b = torch.tensor(s_cur(Xc0[i:i + 4096]), dtype=torch.float32, device=dev)
            out.append(model(a, b).cpu().numpy())
    return Nt + np.concatenate(out) * sd + mu


rm = lambda p, k=slice(None): float(np.sqrt(np.mean((p[k] - At[k]) ** 2)))
base = predict(Xs0)
b_all, b_hi = rm(base), rm(base, hi)

groups = {
    "Neutron monitors": "NM_", "Solar X-rays": "SXR", "Energetic particles": "Particles",
    "Solar wind": "SW_", "Geomagnetic indices (Kp, ap, Dst)": "Index", "Solar activity": "Solar",
}
groups = {g: [env.index(e) for e in env if e.startswith(p)] for g, p in groups.items()}
groups["All 35 variables"] = list(range(35))

past = slice(0, T - 1)
rows = []
for g, idx in groups.items():
    xs = Xs0.copy()
    for j in idx:
        xs[:, past, j] = xs[:, T - 1:T, j]
    p = predict(xs)
    f_all, f_hi = rm(p) - b_all, rm(p, hi) - b_hi
    sa, sh = [], []
    for _ in range(3):
        xs = Xs0.copy(); perm = rng.permutation(len(xs))
        for j in idx:
            xs[:, past, j] = Xs0[perm][:, past, j]
        p = predict(xs); sa.append(rm(p) - b_all); sh.append(rm(p, hi) - b_hi)
    rows.append({"group": g, "freeze_all": f_all, "freeze_high": f_hi,
                 "shuffle_all": np.mean(sa), "shuffle_high": np.mean(sh)})

tag = f"{cad}_rot{rot}_seed{seed}"
pd.DataFrame(rows).to_csv(OUT / f"history_importance_{tag}.csv", index=False)
print(tag, f"base RMSE {b_all:.3f} / high {b_hi:.3f}")
print(pd.DataFrame(rows).round(3).to_string(index=False))
