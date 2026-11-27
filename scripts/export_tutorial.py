"""Real numbers flowing through every network, for the 'How the Six Models Differ' tutorial page.

One reading: G-IV jet, 9 Sept 2017 06:24 UTC (the storm reading used throughout the report).
Models: trained on assignment 231 (train P2, validate P3, test P1), seed 42, so this reading is
test data for all of them. Preprocessing is identical to training (train_basic.py,
train_temporal.py, scripts/export_model_zoo.py).

For each network we record the standardized inputs, the activations after every layer, the
output in standard units, and the final dose, plus one fully worked neuron of the first layer
(its largest input contributions, bias, sum and ReLU) so the page can show the arithmetic.

Writes poster/models_explained/tutorial_data.json
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
from data import load, split, Scaler, STATIC
from models import FlatMLP, CNNMLP, TCN, Transformer

ROT, SEED = "231", 42
d = load("1H"); A_ALL, N_ALL = d["A"], d["N"]; env = list(d["env"]); T, V = 24, 35
m = split(d, *(f"P{c}" for c in ROT))
t_all = pd.to_datetime(pd.Series(d["time"]))
i = int(np.where(t_all.dt.strftime("%Y-%m-%d %H:%M") == "2017-09-09 06:24")[0][0])
assert m["test"][i]

# preprocessing, as in training
seq = d["temporal"].copy()
logv = [k for k, n in enumerate(env) if n.startswith(("SXR", "Particles"))]
seq[:, :, logv] = np.log10(np.clip(seq[:, :, logv], 1e-12, None))
s_seq = Scaler().fit(seq[m["train"]]); Z = s_seq(seq).astype(np.float32)
cur_t = d["static"]                       # NAIRAS is not a network input
s_cur = Scaler().fit(cur_t[m["train"]]); Xc = s_cur(cur_t).astype(np.float32)
y = A_ALL - N_ALL; mu, sd = float(y[m["train"]].mean()), float(y[m["train"]].std())
amu, asd = float(A_ALL[m["train"]].mean()), float(A_ALL[m["train"]].std())
xb = d["current"]; s_b = Scaler().fit(xb[m["train"]]); Xb = s_b(xb).astype(np.float32)
xd = d["current"]; s_d = Scaler().fit(xd[m["train"]]); Xd = s_d(xd).astype(np.float32)


class BasicMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))


def load_sd(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu")); return model.eval()


tdir = ROOT / f"results/temporal_final/1H/rot{ROT}_seed{SEED}"
bdir = ROOT / f"results/basic/rot{ROT}_seed{SEED}"
nc = Xc.shape[1]
net = {"direct": load_sd(BasicMLP(43), bdir / "MLP_(direct).pt"),
       "mlp": load_sd(BasicMLP(43), bdir / "MLP_(residual).pt"),
       "flat": load_sd(FlatMLP(T, V, nc, width=16, p=0.3), tdir / "FlatMLP_residual.pt"),
       "cnn": load_sd(CNNMLP(T, V, nc, width=16, p=0.3), tdir / "CNN-MLP_residual.pt"),
       "tcn": load_sd(TCN(T, V, nc, width=16, p=0.3), tdir / "TCN_residual.pt"),
       "trf": load_sd(Transformer(T, V, nc, width=16, p=0.3), tdir / "Transformer_residual.pt")}
f = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
R = lambda a, k=3: np.round(np.asarray(a, dtype=float), k).tolist()
CUR_NAMES = STATIC + [f"{e} (now)" for e in env]


def worked_neuron(lin, x, names):
    """The first-layer neuron with the largest output: its biggest terms w*x, the rest, bias, sum."""
    W, b = lin.weight.detach().numpy(), lin.bias.detach().numpy()
    pre = W @ x + b; j = int(np.argmax(pre))
    terms = W[j] * x; top = np.argsort(-np.abs(terms))[:5]
    return {"neuron": j, "n_inputs": int(len(x)),
            "terms": [{"name": names[k], "x": round(float(x[k]), 3), "w": round(float(W[j, k]), 3),
                       "wx": round(float(terms[k]), 3)} for k in top],
            "rest": round(float(terms.sum() - terms[top].sum()), 3), "bias": round(float(b[j]), 3),
            "sum": round(float(pre[j]), 3), "relu": round(float(max(pre[j], 0)), 3)}


def mlp_trace(model, x, names, base, scale, shift):
    lin1, lin2, lin3 = model.net[0], model.net[3], model.net[6]
    with torch.no_grad():
        h1 = torch.relu(lin1(f(x))); h2 = torch.relu(lin2(h1)); out = lin3(h2)
    z = float(out); val = z * scale + shift
    return {"inputs": R(x), "input_names": names, "h1": R(h1), "h2": R(h2), "z": round(z, 3),
            "value": round(val, 3), "dose": round(base + val, 3), "neuron": worked_neuron(lin1, x, names),
            "active1": int((h1 > 0).sum()), "active2": int((h2 > 0).sum())}


out = {"reading": {"time": "2017-09-09 06:24 UTC", "A": round(float(A_ALL[i]), 2), "N": round(float(N_ALL[i]), 2),
                   "alt_km": round(float(d["static"][i, 3]) / 1000, 1), "rc": round(float(d["static"][i, 6]), 2),
                   "lat": round(float(d["static"][i, 0]), 2), "lon": round(float(d["static"][i, 1]), 2)},
       "split": "train P2, validate P3, test P1; seed 42", "env": env, "static": STATIC,
       "Z": R(Z[i], 2), "raw_kp": R(d["temporal"][i, :, env.index("Index_Kp")] / 10, 1),
       "corr_scale": {"mu": round(mu, 3), "sd": round(sd, 3)}, "dose_scale": {"mu": round(amu, 3), "sd": round(asd, 3)},
       "models": {}}

N0 = float(N_ALL[i])
out["models"]["direct"] = mlp_trace(net["direct"], Xd[i], CUR_NAMES, 0.0, asd, amu)
out["models"]["mlp"] = mlp_trace(net["mlp"], Xb[i], CUR_NAMES, N0, sd, mu)

# Flat MLP
fm = net["flat"].net
xf = np.concatenate([Z[i].reshape(-1), Xc[i]])
flat_names = [f"{env[v]} {T - 1 - t} h ago" for t in range(T) for v in range(V)] + STATIC
with torch.no_grad():
    h1 = torch.relu(fm[0](f(xf))); h2 = torch.relu(fm[3](h1)); o = float(fm[6](h2))
out["models"]["flat"] = {"h1": R(h1), "h2": R(h2), "z": round(o, 3), "value": round(o * sd + mu, 3),
                         "dose": round(N0 + o * sd + mu, 3), "neuron": worked_neuron(fm[0], xf, flat_names),
                         "n_in": int(len(xf)), "kp_idx": [5 * V + env.index("Index_Kp"), 23 * V + env.index("Index_Kp")]}

# CNN-MLP
cm = net["cnn"]; xs = f(Z[i][None]); xcur = f(Xc[i][None])
with torch.no_grad():
    c1 = torch.relu(cm.conv[0](xs.transpose(1, 2)))                 # 1,16,24
    c2 = torch.relu(cm.conv[3](c1))
    pooled = torch.cat([c2.mean(2), c2[:, :, -1]], 1)
    ce = cm.cur(xcur); zcat = torch.cat([pooled, ce], 1)
    hh = torch.relu(cm.out[0](zcat)); o = float(cm.out[3](hh))
out["models"]["cnn"] = {"c1": R(c1[0]), "c2": R(c2[0]), "mean": R(c2.mean(2)[0]), "last": R(c2[0, :, -1]),
                        "cur": R(ce[0]), "h": R(hh[0]), "z": round(o, 3), "value": round(o * sd + mu, 3),
                        "dose": round(N0 + o * sd + mu, 3)}

# TCN
tm = net["tcn"]
with torch.no_grad():
    h = tm.inp(xs.transpose(1, 2)); layers = [h[0]]
    for blk in tm.blocks:
        h = blk(h); layers.append(h[0])
    ce = tm.cur(xcur); zcat = torch.cat([h[:, :, -1], ce], 1)
    hh = torch.relu(tm.out[0](zcat)); o = float(tm.out[3](hh))
out["models"]["tcn"] = {"layers": [R(L) for L in layers], "dilations": [blk.c1.dilation[0] for blk in tm.blocks],
                        "last": R(h[0, :, -1]), "cur": R(ce[0]), "h": R(hh[0]), "z": round(o, 3),
                        "value": round(o * sd + mu, 3), "dose": round(N0 + o * sd + mu, 3)}

# Transformer
tr = net["trf"]
with torch.no_grad():
    emb = tr.embed(xs)                                                # 1,24,16
    hseq = torch.cat([tr.summary.expand(1, -1, -1), emb], 1) + tr.pos
    att = []
    for blk in tr.blocks:
        hseq, w = blk(hseq); att.append(w[0, 0, 1:])
    s_out = tr.norm(hseq[:, 0]); ce = tr.cur(xcur); zcat = torch.cat([s_out, ce], 1)
    hh = torch.relu(tr.out[0](zcat)); o = float(tr.out[3](hh))
out["models"]["trf"] = {"tokens": R(emb[0], 2), "att": [R(a, 4) for a in att], "s": R(s_out[0]), "cur": R(ce[0]),
                        "h": R(hh[0]), "z": round(o, 3), "value": round(o * sd + mu, 3), "dose": round(N0 + o * sd + mu, 3)}

# sanity: match the saved predictions of this exact run
for k, path in (("direct", bdir / "pred_MLP_direct.npy"), ("mlp", bdir / "pred_MLP_residual.npy"),
                ("flat", tdir / "pred_FlatMLP_residual.npy"), ("cnn", tdir / "pred_CNN-MLP_residual.npy"),
                ("tcn", tdir / "pred_TCN_residual.npy"), ("trf", tdir / "pred_Transformer_residual.npy")):
    saved = float(np.load(path)[i]); mine = out["models"][k]["dose"]
    print(f"{k:7s} dose {mine:7.3f}  saved {saved:7.3f}  diff {abs(mine - saved):.4f}")
    out["models"][k]["params"] = int(sum(p.numel() for p in net[k].parameters()))

p = ROOT / "poster/models_explained/tutorial_data.json"
p.write_text(json.dumps(out, separators=(",", ":")))
print("wrote", p, p.stat().st_size // 1024, "KB")
