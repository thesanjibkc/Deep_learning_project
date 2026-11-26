"""Train the basic (no-history) models on the 43 current-time features.

Models
  NAIRAS V3            physics model, no training
  MLP (direct)         A  ~ 43 features
  MLP (residual)       r  ~ 43 features + NAIRAS,  prediction = N + r

Usage:  python3 train_basic.py --rotation 123     (train P1, validate P2, test P3)
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

ap = argparse.ArgumentParser()
ap.add_argument("--rotation", default="123")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--patience", type=int, default=15)
args = ap.parse_args()

tr, va, te = (f"P{c}" for c in args.rotation)
OUT = Path("results/basic") / f"rot{args.rotation}_seed{args.seed}"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

d = load("5M")                         # current-time features are identical in 5M and 1H
m = split(d, tr, va, te)
A, N = d["A"], d["N"]
X_direct = d["current"]
X_resid = np.concatenate([d["current"], N[:, None]], axis=1)

rows, histories = [], {}


def record(name, pred, extra=None):
    for s in ("train", "val", "test"):
        rows.append({"model": name, "set": s, **regime_metrics(A[m[s]], pred[m[s]]), **(extra or {})})


# 0) NAIRAS
record("NAIRAS V3", N)

# 1) MLP (the 2025 static XGBoost result is cited from the paper, not retrained)
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X, y, base, name):
    sx = Scaler().fit(X[m["train"]])
    ymu, ysd = y[m["train"]].mean(), y[m["train"]].std()
    Xt = torch.tensor(sx(X), dtype=torch.float32, device=dev)
    yt = torch.tensor((y - ymu) / ysd, dtype=torch.float32, device=dev)
    itr = torch.tensor(np.where(m["train"])[0], device=dev)
    model = MLP(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, wait, hist = np.inf, None, 0, []

    def predict():
        model.eval()
        with torch.no_grad():
            return base + model(Xt).cpu().numpy() * ysd + ymu

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = itr[torch.randperm(len(itr), device=dev)]
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = nn.functional.mse_loss(model(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
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
    n_par = sum(p.numel() for p in model.parameters())
    best_ep = int(np.argmin([h["val_rmse"] for h in hist]) + 1)
    record(name, predict(), {"params": n_par, "best_epoch": best_ep,
                             "train_seconds": round(time.time() - t0, 1)})
    torch.save(model.state_dict(), OUT / f"{name.replace(' ', '_')}.pt")


train_mlp(X_direct, A, 0 * N, "MLP (direct)")
train_mlp(X_resid, A - N, N, "MLP (residual)")

res = pd.DataFrame(rows)
res.to_csv(OUT / "metrics.csv", index=False)
json.dump(histories, open(OUT / "histories.json", "w"))
show = res[res["set"] == "test"][["model", "rmse_all", "rmse_high", "bias_high", "r2_all"]]
print(f"\nRotation {args.rotation}: train {tr}, validation {va}, test {te}  (device {dev})")
print(show.round(3).to_string(index=False))
print("\nvalidation RMSE:", res[res["set"] == "val"].set_index("model")["rmse_all"].round(3).to_dict())
