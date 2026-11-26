"""Final statistics: seed ensembles and paired flight-bootstrap significance tests.

For each of the 6 train/validation/test assignments, the 3 seeds' test predictions are
averaged (seed ensemble). Paired comparisons resample whole FLIGHTS (aircraft + date)
within each assignment, recompute RMSE(model) - RMSE(reference), and average over the
6 assignments. A 95% interval entirely below zero means the model is reliably better.

Writes results/final/ensemble.csv and results/final/significance.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import load, split, DATA, PARTS

OUT = Path("results/final"); OUT.mkdir(parents=True, exist_ok=True)
ROTS = ["123", "132", "213", "231", "312", "321"]; SEEDS = [42, 1, 2]
d = load("1H"); A, N = d["A"], d["N"]
veh = []
for p in PARTS:
    for line in (DATA / "1H" / f"ARMAS_TIMESERIES_1H_20132023_{p}.json").open():
        if line.strip():
            veh.append(json.loads(line)["Vehicle_ID"])
flight = pd.Series([f"{v}|{t[:10]}" for v, t in zip(veh, d["time"])]).factorize()[0]

MODELS = {   # label -> (results dir, file stem)
    "MLP (no history)": ("results/basic", "MLP_residual"),
    "MLP direct (no history)": ("results/basic", "MLP_direct"),
}
for c, w in (("5M", "1 h"), ("1H", "24 h")):
    for stem, lab in (("FlatMLP_residual", "Flat MLP"), ("CNN-MLP_residual", "CNN–MLP"), ("TCN_residual", "TCN"),
                      ("Transformer_residual", "Transformer"), ("Transformer_direct", "Transformer direct")):
        MODELS[f"{lab}, {w}"] = (f"results/temporal_final/{c}", stem)
    MODELS[f"Transformer + high-dose weight, {w}"] = (f"results/temporal_final_weighted/{c}", "Transformer_residual")


def ens(model, rot):
    root, stem = MODELS[model]
    ps = [np.load(Path(root) / f"rot{rot}_seed{s}" / f"pred_{stem}.npy") for s in SEEDS]
    return np.mean(ps, 0)


rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
rows = []
for model in MODELS:
    per = []
    for r in ROTS:
        te = split(d, *(f"P{c}" for c in r))["test"]
        p = ens(model, r)
        per.append((rm(p[te] - A[te]), rm(p[te & (A > 15)] - A[te & (A > 15)])))
    per = np.array(per)
    rows.append({"model": model, "ens_all": per[:, 0].mean(), "ens_all_sd": per[:, 0].std(ddof=1),
                 "ens_high": per[:, 1].mean(), "ens_high_sd": per[:, 1].std(ddof=1)})
nai = []
for r in ROTS:
    te = split(d, *(f"P{c}" for c in r))["test"]
    nai.append((rm(N[te] - A[te]), rm(N[te & (A > 15)] - A[te & (A > 15)])))
nai = np.array(nai)
rows.append({"model": "NAIRAS V3", "ens_all": nai[:, 0].mean(), "ens_all_sd": nai[:, 0].std(ddof=1),
             "ens_high": nai[:, 1].mean(), "ens_high_sd": nai[:, 1].std(ddof=1)})
E = pd.DataFrame(rows).sort_values("ens_all")
E.to_csv(OUT / "ensemble.csv", index=False)
print("Seed-ensemble test RMSE (mean ± sd over 6 assignments)")
print(E.round(3).to_string(index=False))

# ---- paired flight bootstrap
B = 1000
rng = np.random.default_rng(0)
cache = {}


def preds(name, r):
    if name == "NAIRAS V3":
        return N
    if (name, r) not in cache:
        cache[(name, r)] = ens(name, r)
    return cache[(name, r)]


def compare(model, ref, regime):
    diffs = np.zeros(B); point = []
    for r in ROTS:
        te = split(d, *(f"P{c}" for c in r))["test"]
        k = te & (A > 15) if regime == "high" else te
        idx = np.where(k)[0]
        f = flight[idx]; uf, inv = np.unique(f, return_inverse=True)
        e1 = (preds(model, r)[idx] - A[idx]) ** 2
        e2 = (preds(ref, r)[idx] - A[idx]) ** 2
        s1 = np.bincount(inv, e1, len(uf)); s2 = np.bincount(inv, e2, len(uf)); n = np.bincount(inv, None, len(uf))
        point.append(np.sqrt(s1.sum() / n.sum()) - np.sqrt(s2.sum() / n.sum()))
        draw = rng.integers(0, len(uf), (B, len(uf)))
        diffs += np.sqrt(s1[draw].sum(1) / n[draw].sum(1)) - np.sqrt(s2[draw].sum(1) / n[draw].sum(1))
    diffs /= len(ROTS)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"model": model, "reference": ref, "regime": regime, "diff": float(np.mean(point)),
            "ci_low": float(lo), "ci_high": float(hi), "p_not_better": float((diffs >= 0).mean()),
            "assignments_better": int(sum(x < 0 for x in point))}


tests = [
    ("Transformer, 24 h", "NAIRAS V3"), ("Transformer, 24 h", "MLP (no history)"),
    ("CNN–MLP, 24 h", "MLP (no history)"), ("TCN, 24 h", "MLP (no history)"),
    ("Transformer, 24 h", "Transformer, 1 h"), ("Transformer, 24 h", "Transformer direct, 24 h"),
    ("Transformer, 24 h", "Flat MLP, 24 h"), ("MLP (no history)", "MLP direct (no history)"),
    ("Transformer + high-dose weight, 24 h", "Transformer, 24 h"),
    ("Transformer + high-dose weight, 1 h", "MLP (no history)"),
]
S = pd.DataFrame([compare(m, r, g) for m, r in tests for g in ("all", "high")])
S.to_csv(OUT / "significance.csv", index=False)
print("\nPaired flight bootstrap: RMSE(model) - RMSE(reference), averaged over 6 assignments")
print(S.round(3).to_string(index=False))
