"""Why is the error large?  Two analyses on the TEST partition of one trained run.

1) Permutation importance of the trained residual Transformer:
   shuffle one feature group (all hours) or one hour (all variables) across test
   observations and measure how much RMSE rises, overall and for A > 15.
2) What is left: the remaining error e = A - prediction, compared with NAIRAS's
   error r = A - N. For each grouping (location band, altitude band, storm level,
   year, aircraft) we report the share of error variance the grouping explains
   (eta squared). Aircraft (Vehicle ID) is never a model input, so a large share
   there points to an instrument effect.

Usage: python3 src/explain_errors.py [cadence] [rotation] [seed] [results root]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from data import load, split, Scaler, DATA, PARTS
from models import Transformer

cad, rot, seed = (sys.argv[1:4] + ["1H", "123", "42"][len(sys.argv[1:4]):])[:3]
root = Path(sys.argv[4] if len(sys.argv) > 4 else "results/temporal_final")
run = root / cad / f"rot{rot}_seed{seed}"
OUT = Path("results/explain"); OUT.mkdir(parents=True, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); rng = np.random.default_rng(0)

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
base = predict(Xs0, Xc0)
b_all, b_hi = rm(base), rm(base, hi)

groups_t = {
    "Neutron monitors": [e for e in env if e.startswith("NM_")],
    "Solar X-rays": [e for e in env if e.startswith("SXR")],
    "Energetic particles": [e for e in env if e.startswith("Particles")],
    "Solar wind": [e for e in env if e.startswith("SW_")],
    "Geomagnetic indices (Kp, ap, Dst)": [e for e in env if e.startswith("Index")],
    "Solar activity (sunspots, F10.7, polar field)": [e for e in env if e.startswith("Solar")],
}
groups_c = {"Position (lat, lon)": [0, 1], "Altitude": [2, 3],
            "Geomagnetic shielding (Rc, L, geomag lat/lon)": [4, 5, 6, 7]}
REPEATS = 3
imp = []
for g, names in groups_t.items():
    idx = [env.index(n) for n in names]
    da, dh = [], []
    for _ in range(REPEATS):
        xs = Xs0.copy(); perm = rng.permutation(len(xs)); xs[:, :, idx] = xs[perm][:, :, idx]
        p = predict(xs, Xc0); da.append(rm(p) - b_all); dh.append(rm(p, hi) - b_hi)
    imp.append({"group": g, "kind": "space weather (history)", "d_all": np.mean(da), "d_high": np.mean(dh)})
for g, idx in groups_c.items():
    da, dh = [], []
    for _ in range(REPEATS):
        xc = Xc0.copy(); perm = rng.permutation(len(xc)); xc[:, idx] = xc[perm][:, idx]
        p = predict(Xs0, xc); da.append(rm(p) - b_all); dh.append(rm(p, hi) - b_hi)
    imp.append({"group": g, "kind": "flight (current)", "d_all": np.mean(da), "d_high": np.mean(dh)})
imp = pd.DataFrame(imp).sort_values("d_all", ascending=False)

lag = []
for t in range(T):
    da, dh = [], []
    for _ in range(REPEATS):
        xs = Xs0.copy(); perm = rng.permutation(len(xs)); xs[:, t, :] = xs[perm][:, t, :]
        p = predict(xs, Xc0); da.append(rm(p) - b_all); dh.append(rm(p, hi) - b_hi)
    lag.append({"steps_before": T - 1 - t, "d_all": np.mean(da), "d_high": np.mean(dh)})
lag = pd.DataFrame(lag)

# ---- 2) what is left
raw = []
for p in PARTS:
    for line in (DATA / cad / f"ARMAS_TIMESERIES_{cad}_20132023_{p}.json").open():
        if line.strip():
            raw.append(json.loads(line)["Vehicle_ID"])
veh = np.array(raw)[ti]
st = d["static"][ti]
kp24 = d["temporal"][ti, :, env.index("Index_Kp")].max(1) / 10
year = pd.to_datetime(pd.Series(d["time"][ti])).dt.year.values
G = pd.DataFrame({
    "remaining": At - base, "nairas": At - Nt,
    "Shielding band (Rc)": pd.cut(st[:, 6], [-1, 1, 3, 6, 10, 20]).astype(str),
    "Altitude band": pd.cut(st[:, 3] / 1000, [0, 10, 11, 12, 20]).astype(str),
    "Storm level (max Kp, 24 h)": pd.cut(kp24, [-1, 3, 5, 9.5]).astype(str),
    "Year": year.astype(str), "Aircraft (Vehicle ID, not an input)": veh})


def eta2(col, target):
    g = G.groupby(col)[target]
    between = (g.size() * (g.mean() - G[target].mean()) ** 2).sum()
    return float(between / ((G[target] - G[target].mean()) ** 2).sum())


left = pd.DataFrame([{"grouping": c, "NAIRAS error": eta2(c, "nairas"), "remaining error": eta2(c, "remaining")}
                     for c in G.columns[2:]])
veh_tab = G.groupby("Aircraft (Vehicle ID, not an input)").agg(
    n=("remaining", "size"), nairas_mean=("nairas", "mean"), remaining_mean=("remaining", "mean")
).sort_values("n", ascending=False)

tag = f"{cad}_rot{rot}_seed{seed}"
imp.to_csv(OUT / f"importance_groups_{tag}.csv", index=False)
lag.to_csv(OUT / f"importance_hours_{tag}.csv", index=False)
left.to_csv(OUT / f"remaining_error_{tag}.csv", index=False)
veh_tab.to_csv(OUT / f"remaining_by_aircraft_{tag}.csv")
print(f"base test RMSE all {b_all:.3f}  high {b_hi:.3f}")
print(imp.round(3).to_string(index=False))
print(lag.round(3).to_string(index=False))
print(left.round(3).to_string(index=False))
print(veh_tab.round(2).to_string())
