"""Which individual variable matters?  Per-variable permutation importance of the trained
residual Transformer (1-hour history: cadence 5M; 24-hour history: cadence 1H) on the TEST partition of one assignment (seed 42).

For each of the 35 space-weather variables (the whole window shuffled together across test
observations) and each of the 8 current-time inputs, the rise in test RMSE, overall and for
A > 15, averaged over 3 shuffles. src/explain_errors.py does the same per GROUP of variables.

Usage: python3 src/variable_importance.py [cadence] [rotation] [seed]
Writes results/explain/importance_vars_{cadence}_rot{rot}_seed{seed}.csv
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
torch.manual_seed(0); rng = np.random.default_rng(0)

# same preprocessing as src/explain_errors.py
d = load(cad); tr, va, te = (f"P{c}" for c in rot); m = split(d, tr, va, te)
A, N, env = d["A"], d["N"], list(d["env"])
seq = d["temporal"].copy()
logv = [i for i, n in enumerate(env) if n.startswith(("SXR", "Particles"))]
seq[:, :, logv] = np.log10(np.clip(seq[:, :, logv], 1e-12, None))
s_seq = Scaler().fit(seq[m["train"]])
cur = d["static"]                       # NAIRAS is not a network input
s_cur = Scaler().fit(cur[m["train"]])
y = A - N; mu, sd = y[m["train"]].mean(), y[m["train"]].std()
T = seq.shape[1]
model = Transformer(T, 35, cur.shape[1], width=16, p=0.3).to(dev)
model.load_state_dict(torch.load(run / "Transformer_residual.pt", map_location=dev))
model.eval()
ti = np.where(m["test"])[0]
Xs0, Xc0, At, Nt = seq[ti], cur[ti], A[ti], N[ti]
hi = At > 15


def predict(xs, xc):
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), 4096):
            a = torch.tensor(s_seq(xs[i:i + 4096]), dtype=torch.float32, device=dev)
            b = torch.tensor(s_cur(xc[i:i + 4096]), dtype=torch.float32, device=dev)
            out.append(model(a, b).cpu().numpy())
    return Nt + np.concatenate(out) * sd + mu


rm = lambda p, k=slice(None): float(np.sqrt(np.mean((p[k] - At[k]) ** 2)))
base = predict(Xs0, Xc0); b_all, b_hi = rm(base), rm(base, hi)
rows = []
for j, name in enumerate(env):
    da, dh = [], []
    for _ in range(3):
        xs = Xs0.copy(); xs[:, :, j] = xs[rng.permutation(len(xs)), :, j]
        p = predict(xs, Xc0); da.append(rm(p) - b_all); dh.append(rm(p, hi) - b_hi)
    rows.append({"variable": name, "kind": "space weather (history)", "d_all": np.mean(da), "d_high": np.mean(dh)})
for j, name in enumerate(STATIC):
    da, dh = [], []
    for _ in range(3):
        xc = Xc0.copy(); xc[:, j] = xc[rng.permutation(len(xc)), j]
        p = predict(Xs0, xc); da.append(rm(p) - b_all); dh.append(rm(p, hi) - b_hi)
    rows.append({"variable": name, "kind": "flight (current)", "d_all": np.mean(da), "d_high": np.mean(dh)})
df = pd.DataFrame(rows).sort_values("d_all", ascending=False)
df.to_csv(OUT / f"importance_vars_{cad}_rot{rot}_seed{seed}.csv", index=False)
print(f"{cad} rot {rot}: base {b_all:.3f}"); print(df.head(10).round(3).to_string(index=False))
