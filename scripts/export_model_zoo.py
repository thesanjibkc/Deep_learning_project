"""Export real model behaviour on one unseen flight, plus the trained weights, for the Model Zoo page.

Flight: the G-IV flight on 9 Sept 2017 that contains the storm observation (06:24 UTC).
Split: rotation 231 (train P2, validate P3, test P1), seed 42, so every model is shown on a flight
it never saw. Models: NAIRAS, MLP, Flat MLP, CNN-MLP, TCN, Transformer, Transformer + high-dose weight
(24-hour history).

Writes poster/zoo/zoo_data.json with
  line      downsampled timeline: measured, NAIRAS and every model's prediction along the flight
  snaps     ~17 snapshots with the standardized inputs each network receives
  weights   every trained parameter of the six networks, so the page can run them live in JavaScript
  sc        the scaler statistics needed to standardize edited inputs
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load, split, Scaler, DATA, PARTS
from models import FlatMLP, CNNMLP, TCN, Transformer

ROT, SEED = "231", 42
torch.manual_seed(0)
d = load("1H"); N_ALL, A_ALL = d["N"], d["A"]
env = list(d["env"]); T, V = 24, 35
tr, va, te = (f"P{c}" for c in ROT); m = split(d, tr, va, te)
t_all = pd.to_datetime(pd.Series(d["time"]))

# ---- pick the flight segment (same aircraft, contiguous, containing the storm observation)
veh = []
for p in PARTS:
    for line in (DATA / "1H" / f"ARMAS_TIMESERIES_1H_20132023_{p}.json").open():
        if line.strip():
            veh.append(json.loads(line)["Vehicle_ID"])
veh = np.array(veh)
storm_i = int(np.where(t_all.dt.strftime("%Y-%m-%d %H:%M") == "2017-09-09 06:24")[0][0])
same = np.where((veh == veh[storm_i]) & (t_all.dt.strftime("%Y-%m-%d").values == "2017-09-09"))[0]
same = same[np.argsort(t_all.iloc[same].values)]
secs = t_all.iloc[same].values.astype("datetime64[s]").astype(np.int64)
seg_id = np.concatenate([[0], np.cumsum(np.diff(secs) > 600)])
flight = same[seg_id == seg_id[list(same).index(storm_i)]]
assert m["test"][flight].all(), "flight must be in the test partition"
t0 = t_all.iloc[flight[0]]
minutes = ((t_all.iloc[flight] - t0).dt.total_seconds() / 60).values
print(f"flight: {veh[storm_i]}  {len(flight)} rows  {minutes[-1]:.0f} min  start {t0}")

# ---- preprocessing identical to train_temporal.py / train_basic.py
seq = d["temporal"].copy()
logv = [i for i, n in enumerate(env) if n.startswith(("SXR", "Particles"))]
seq[:, :, logv] = np.log10(np.clip(seq[:, :, logv], 1e-12, None))
s_seq = Scaler().fit(seq[m["train"]])
Z = s_seq(seq).astype(np.float32)                                   # n, T, V (standardized)
cur_t = np.concatenate([d["static"], N_ALL[:, None]], 1)
s_cur = Scaler().fit(cur_t[m["train"]])
Xc = s_cur(cur_t).astype(np.float32)
y = A_ALL - N_ALL; mu, sd = float(y[m["train"]].mean()), float(y[m["train"]].std())
xb = np.concatenate([d["current"], N_ALL[:, None]], 1)                # basic MLP residual input (44)
s_b = Scaler().fit(xb[m["train"]]); Xb = s_b(xb).astype(np.float32)


class BasicMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def load_sd(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu")); return model.eval()


tdir = ROOT / f"results/temporal_final/1H/rot{ROT}_seed{SEED}"
wdir = ROOT / f"results/temporal_final_weighted/1H/rot{ROT}_seed{SEED}"
bdir = ROOT / f"results/basic/rot{ROT}_seed{SEED}"
nc = Xc.shape[1]
net = {
    "mlp": load_sd(BasicMLP(44), bdir / "MLP_(residual).pt"),
    "flat": load_sd(FlatMLP(T, V, nc, width=16, p=0.3), tdir / "FlatMLP_residual.pt"),
    "cnn": load_sd(CNNMLP(T, V, nc, width=16, p=0.3), tdir / "CNN-MLP_residual.pt"),
    "tcn": load_sd(TCN(T, V, nc, width=16, p=0.3), tdir / "TCN_residual.pt"),
    "trf": load_sd(Transformer(T, V, nc, width=16, p=0.3), tdir / "Transformer_residual.pt"),
    "trfw": load_sd(Transformer(T, V, nc, width=16, p=0.3), wdir / "Transformer_residual.pt"),
}
KEYS = ["mlp", "flat", "cnn", "tcn", "trf", "trfw"]
params = {k: sum(p.numel() for p in net[k].parameters()) for k in KEYS}
f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)


def run(key, z, xc, xb_, n):
    """Standardized forward -> predicted dose (µSv/h). z: (B,T,V); xc: (B,C); xb_: (B,44); n: (B,)"""
    with torch.no_grad():
        out = net[key](f32(xb_)).squeeze(-1) if key == "mlp" else net[key](f32(z), f32(xc))
    return n + out.numpy() * (sd if key != "mlp" else sd) + mu


# ---- sanity: reproduce the saved predictions for this flight
for key, stem in (("flat", "FlatMLP_residual"), ("cnn", "CNN-MLP_residual"), ("tcn", "TCN_residual"),
                  ("trf", "Transformer_residual")):
    saved = np.load(tdir / f"pred_{stem}.npy")[flight]
    mine = run(key, Z[flight], Xc[flight], Xb[flight], N_ALL[flight])
    print(f"  check {key}: max |diff| vs saved = {np.abs(saved - mine).max():.5f}")
saved = np.load(wdir / "pred_Transformer_residual.npy")[flight]
print(f"  check trfw: {np.abs(saved - run('trfw', Z[flight], Xc[flight], Xb[flight], N_ALL[flight])).max():.5f}")
saved = np.load(bdir / "pred_MLP_residual.npy")[flight]
print(f"  check mlp: {np.abs(saved - run('mlp', Z[flight], Xc[flight], Xb[flight], N_ALL[flight])).max():.5f}")

# ---- timeline (downsample to <= 160 points)
step = max(1, len(flight) // 160); sel = np.arange(0, len(flight), step)
fl = flight[sel]
preds_all = {k: run(k, Z[flight], Xc[flight], Xb[flight], N_ALL[flight]) for k in KEYS}
r2 = lambda a: [round(float(x), 2) for x in a]
line = {"t": r2(minutes[sel]), "A": r2(A_ALL[fl]), "N": r2(N_ALL[fl]),
        "alt": r2(d["static"][fl, 3] / 1000), "rc": r2(d["static"][fl, 6]),
        "preds": {k: r2(preds_all[k][sel]) for k in KEYS}}
rmse = {k: float(np.sqrt(np.mean((preds_all[k] - A_ALL[flight]) ** 2))) for k in KEYS}
rmse["nairas"] = float(np.sqrt(np.mean((N_ALL[flight] - A_ALL[flight]) ** 2)))
print("flight RMSE:", {k: round(v, 2) for k, v in rmse.items()})

# ---- snapshots with internals
n_snap = 16
snap_idx = np.unique(np.linspace(0, len(sel) - 1, n_snap).round().astype(int))
storm_pos = int(np.argmin(np.abs(flight - storm_i)))
storm_sel = int(np.argmin(np.abs(sel - storm_pos)))
if storm_sel not in snap_idx:
    snap_idx = np.sort(np.append(snap_idx, storm_sel))
SHOW = ["NM_OULU", "Index_Kp", "Index_Dst", "SW_Bz", "SW_V", "SW_pressure"]
SHOW_LABEL = ["Neutron monitor", "Kp", "Dst", "Solar wind Bz", "Solar wind speed", "Solar wind pressure"]
sj = [env.index(v) for v in SHOW]


def pack(model):
    return {n: {"s": list(t.shape), "d": np.round(t.detach().cpu().numpy().astype(np.float32), 5).ravel().tolist()}
            for n, t in model.state_dict().items()}


WEIGHTS = {k: pack(net[k]) for k in KEYS}
rr = lambda a: np.round(np.asarray(a, dtype=np.float64).ravel(), 5).tolist()
SCALERS = {"seq_mu": rr(s_seq.mu), "seq_sd": rr(s_seq.sd), "cur_mu": rr(s_cur.mu), "cur_sd": rr(s_cur.sd),
           "b_mu": rr(s_b.mu), "b_sd": rr(s_b.sd), "ymu": round(mu, 5), "ysd": round(sd, 5)}
fam = [env.index(n) for n in env if n.startswith(("Index_", "SW_"))]
snaps = []
for si in snap_idx:
    gi = fl[si]
    snaps.append({
        "i": int(si), "t": float(line["t"][si]), "A": line["A"][si], "N": line["N"][si],
        "alt": line["alt"][si], "rc": line["rc"][si],
        "lat": round(float(d["static"][gi, 0]), 1), "lon": round(float(d["static"][gi, 1]), 1),
        # standardized model inputs, so the page can run the networks itself
        "zs": np.round(Z[gi], 4).tolist(), "xc": np.round(Xc[gi], 4).tolist(), "xb": np.round(Xb[gi], 4).tolist(),
        "preds": {k: line["preds"][k][si] for k in KEYS}})
storm_snap = int(np.where(snap_idx == storm_sel)[0][0])

out = {
    "meta": {"aircraft": str(veh[storm_i]).replace("*", ""), "start": str(t0)[:16] + " UTC", "minutes": round(float(minutes[-1])),
             "rows": int(len(flight)), "split": f"train {tr}, validate {va}, test {te}", "seed": SEED,
             "storm_peak_kp": 8.3, "storm_label": "9 Sept 2017 flight, morning after a severe geomagnetic storm"},
    "models": [{"key": "nairas", "label": "NAIRAS V3", "params": 0},
               {"key": "mlp", "label": "MLP", "params": params["mlp"]},
               {"key": "flat", "label": "Flat MLP", "params": params["flat"]},
               {"key": "cnn", "label": "CNN–MLP", "params": params["cnn"]},
               {"key": "tcn", "label": "TCN", "params": params["tcn"]},
               {"key": "trf", "label": "Transformer", "params": params["trf"]},
               {"key": "trfw", "label": "Transformer, high-dose weight", "params": params["trfw"]}],
    "vars": SHOW_LABEL, "line": line, "flight_rmse": {k: round(v, 3) for k, v in rmse.items()},
    "snaps": snaps, "storm_snap": storm_snap, "weights": WEIGHTS, "sc": SCALERS,
    "fam": fam, "sj": sj, "kp_idx": env.index("Index_Kp"),
}
Path(ROOT / "poster/zoo").mkdir(parents=True, exist_ok=True)
p = ROOT / "poster/zoo/zoo_data.json"
json.dump(out, open(p, "w"), separators=(",", ":"))
print("wrote", p, p.stat().st_size // 1024, "KB;", len(snaps), "snapshots;", len(line["t"]), "timeline points")
