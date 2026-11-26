"""Record what the networks learn, epoch by epoch (train P1, validate P2; 24-hour data).

Saves results/tracking/tracking.npz with, for every epoch:
  - train / validation RMSE for the Flat MLP and the Transformer
  - Transformer: attention from the summary token to each of the 24 hours,
      averaged over high-dose validation flights, and for the Sept 9 2017 storm flight
  - Transformer: full 25x25 attention matrix (layer 1) for the storm flight
  - Flat MLP: first-layer weight strength for each (hour, variable) input
Uses the final regularized setting (width 16, dropout 0.3, weight decay 0.01).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load, split, Scaler
from models import FlatMLP, Transformer

torch.manual_seed(42); np.random.seed(42)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
d = load("1H"); m = split(d, "P1", "P2", "P3")
A, N = d["A"], d["N"]
seq = d["temporal"].copy()
logv = [i for i, n in enumerate(d["env"]) if n.startswith(("SXR", "Particles"))]
seq[:, :, logv] = np.log10(np.clip(seq[:, :, logv], 1e-12, None))
Xs = torch.tensor(Scaler().fit(seq[m["train"]])(seq), dtype=torch.float32, device=dev)
cur = np.concatenate([d["static"], N[:, None]], 1)
Xc = torch.tensor(Scaler().fit(cur[m["train"]])(cur), dtype=torch.float32, device=dev)
y = A - N; mu, sd = y[m["train"]].mean(), y[m["train"]].std()
yt = torch.tensor((y - mu) / sd, dtype=torch.float32, device=dev)
itr = torch.tensor(np.where(m["train"])[0], device=dev)
t = pd.to_datetime(pd.Series(d["time"]))
storm = int(np.where(t.dt.strftime("%Y-%m-%d %H:%M") == "2017-09-09 06:24")[0][0])
hi_val = torch.tensor(np.where(m["val"] & (A > 15))[0], device=dev)
EPOCHS = 30
out = {}


def rmse(model):
    model.eval(); p = []
    with torch.no_grad():
        for i in range(0, len(A), 4096):
            p.append(model(Xs[i:i + 4096], Xc[i:i + 4096]))
    p = N + torch.cat(p).cpu().numpy() * sd + mu
    return [float(np.sqrt(np.mean((p[m[s]] - A[m[s]]) ** 2))) for s in ("train", "val")]


for name, cls in (("flat", FlatMLP), ("trf", Transformer)):
    model = cls(24, 35, cur.shape[1], width=16, p=0.3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4 if name == "trf" else 1e-3, weight_decay=0.01)
    curve, att_hi, att_storm, mat_storm, wmap = [], [], [], [], []

    def snapshot():
        curve.append(rmse(model))
        if name == "trf":
            model.eval()
            with torch.no_grad():
                model(Xs[hi_val], Xc[hi_val]); att_hi.append(model.last_attention.mean((0, 1)).cpu().numpy())
                h = torch.cat([model.summary, model.embed(Xs[storm:storm + 1])], 1) + model.pos
                blk = model.blocks[0]; z = blk.n1(h)
                _, w = blk.attn(z, z, z, need_weights=True, average_attn_weights=True)
                mat_storm.append(w[0].cpu().numpy())
                model(Xs[storm:storm + 1], Xc[storm:storm + 1]); att_storm.append(model.last_attention.mean(1)[0].cpu().numpy())
        else:
            W = model.net[0].weight.detach().abs().sum(0)[:24 * 35].reshape(24, 35).cpu().numpy()
            wmap.append(W)

    snapshot()                                   # epoch 0 = untrained
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = itr[torch.randperm(len(itr), device=dev)]
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = nn.functional.mse_loss(model(Xs[b], Xc[b]), yt[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        snapshot()
        print(name, ep, [round(x, 3) for x in curve[-1]], flush=True)
    out[f"{name}_curve"] = np.array(curve)
    if name == "trf":
        out.update(trf_att_hi=np.array(att_hi), trf_att_storm=np.array(att_storm), trf_mat_storm=np.array(mat_storm))
    else:
        out["flat_wmap"] = np.array(wmap)

o = ROOT / "results/tracking"; o.mkdir(parents=True, exist_ok=True)
np.savez(o / "tracking.npz", env=d["env"], **out)
print("saved", o / "tracking.npz")
