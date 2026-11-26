"""Train the temporal neural networks on the 5M (1-hour) or 1H (24-hour) history.

Runs (all predict the correction r = A - N, prediction = N + r, unless "direct"):
  FlatMLP (residual), CNN-MLP (residual), TCN (residual),
  Transformer (residual), Transformer (direct)

Usage:  python3 train_temporal.py --cadence 1H --rotation 123 --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load, split, Scaler
from metrics import regime_metrics
from models import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--cadence", default="1H", choices=["5M", "1H"])
ap.add_argument("--rotation", default="123")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--models", default="FlatMLP,CNN-MLP,TCN,Transformer,Transformer-direct")
ap.add_argument("--width", type=int, default=64)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--lr", type=float, default=0)   # 0 = default per model
ap.add_argument("--tag", default="")             # results/temporal<tag>/...
ap.add_argument("--curfull", type=int, default=0)  # 1 = current branch gets all 43 current features
ap.add_argument("--loss", default="mse", choices=["mse", "huber"])
ap.add_argument("--hiw", type=float, default=1.0)  # loss weight for A > 15
args = ap.parse_args()

tr, va, te = (f"P{c}" for c in args.rotation)
OUT = Path(f"results/temporal{args.tag}") / args.cadence / f"rot{args.rotation}_seed{args.seed}"
OUT.mkdir(parents=True, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

d = load(args.cadence)
m = split(d, tr, va, te)
A, N = d["A"], d["N"]

# Heavy-tailed fluxes (X-rays, particles) span orders of magnitude: use log10.
seq = d["temporal"].copy()
logv = [i for i, n in enumerate(d["env"]) if n.startswith(("SXR", "Particles"))]
seq[:, :, logv] = np.log10(np.clip(seq[:, :, logv], 1e-12, None))
s_seq = Scaler().fit(seq[m["train"]])
Xseq = torch.tensor(s_seq(seq), dtype=torch.float32, device=dev)

rows, histories = [], {}


def record(name, pred, extra=None):
    for s in ("train", "val", "test"):
        rows.append({"model": name, "set": s, **regime_metrics(A[m[s]], pred[m[s]]), **(extra or {})})


record("NAIRAS V3", N)


def run(name, arch, residual):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cur0 = d["current"] if args.curfull else d["static"]
    cur = np.concatenate([cur0, N[:, None]], 1) if residual else cur0
    y = A - N if residual else A
    base = N if residual else np.zeros_like(N)
    s_cur = Scaler().fit(cur[m["train"]])
    Xcur = torch.tensor(s_cur(cur), dtype=torch.float32, device=dev)
    ymu, ysd = y[m["train"]].mean(), y[m["train"]].std()
    yt = torch.tensor((y - ymu) / ysd, dtype=torch.float32, device=dev)
    itr = torch.tensor(np.where(m["train"])[0], device=dev)
    wt = torch.tensor(np.where(A > 15, args.hiw, 1.0), dtype=torch.float32, device=dev)

    T, V = seq.shape[1], seq.shape[2]
    model = MODELS[arch](T, V, cur.shape[1], width=args.width, p=args.dropout).to(dev)
    lr = args.lr or (5e-4 if arch == "Transformer" else 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)

    def predict():
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(A), 4096):
                out.append(model(Xseq[i:i + 4096], Xcur[i:i + 4096]))
        return base + torch.cat(out).cpu().numpy() * ysd + ymu

    best, best_state, wait, hist, t0 = np.inf, None, 0, [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = itr[torch.randperm(len(itr), device=dev)]
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            err = model(Xseq[b], Xcur[b]) - yt[b]
            per = err ** 2 if args.loss == "mse" else nn.functional.huber_loss(err, torch.zeros_like(err), reduction="none", delta=1.0)
            loss = (wt[b] * per).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        p = predict()
        rm = {s: float(np.sqrt(np.mean((p[m[s]] - A[m[s]]) ** 2))) for s in ("train", "val")}
        hist.append({"epoch": ep, "train_rmse": rm["train"], "val_rmse": rm["val"]})
        if rm["val"] < best - 1e-4:
            best, wait = rm["val"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    histories[name] = hist
    pred = predict()
    np.save(OUT / f"pred_{name.replace(' ', '_').replace('(', '').replace(')', '')}.npy", pred.astype(np.float32))
    record(name, pred, {"params": sum(p.numel() for p in model.parameters()),
                             "best_epoch": int(np.argmin([h["val_rmse"] for h in hist]) + 1),
                             "train_seconds": round(time.time() - t0, 1)})
    torch.save(model.state_dict(), OUT / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.pt")
    print(f"  {name:<26} val {best:.3f}  epochs {len(hist):>3}  {time.time() - t0:6.1f}s", flush=True)


for spec in args.models.split(","):
    direct = spec.endswith("-direct")
    arch = spec.replace("-direct", "")
    run(f"{arch} ({'direct' if direct else 'residual'})", arch, residual=not direct)

res = pd.DataFrame(rows)
res.to_csv(OUT / "metrics.csv", index=False)
json.dump(histories, open(OUT / "histories.json", "w"))
show = res[res["set"] == "test"][["model", "rmse_all", "rmse_high", "r2_all"]]
print(f"\n{args.cadence} rotation {args.rotation} seed {args.seed}: test {te}")
print(show.round(3).to_string(index=False))
