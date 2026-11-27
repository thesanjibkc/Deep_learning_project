"""Write poster/animation/tracking.json (used by the 'Watching Attention Learn' page and the slides)
from results/tracking/tracking.npz (scripts/track_training.py) and the final 18-run results.

The storm-flight description (inputs shown on the page) is kept from the existing file; every
learned quantity (curves, attention, weight shares, best epoch, parameters, final errors) is
replaced with the current run.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "poster/animation/tracking.json"
d = json.loads(P.read_text())
z = np.load(ROOT / "results/tracking/tracking.npz")
R = lambda a, k=4: np.round(np.asarray(a, dtype=float), k).tolist()

tc, fc = z["trf_curve"], z["flat_curve"]
wm = z["flat_wmap"].sum(2); wm = wm / wm.sum(1, keepdims=True)
d["epochs"] = list(range(len(tc)))
d["transformer"].update({"train_rmse": R(tc[:, 0], 3), "val_rmse": R(tc[:, 1], 3),
                         "attention_storm_flight": R(z["trf_att_storm"]), "attention_high_dose_avg": R(z["trf_att_hi"]),
                         "attention_matrix_layer1_storm": R(z["trf_mat_storm"], 3)})
d["flat_mlp"].update({"train_rmse": R(fc[:, 0], 3), "val_rmse": R(fc[:, 1], 3), "weight_share_per_hour": R(wm)})
d["best_epoch_transformer"] = int(tc[:, 1].argmin())


def mean_test(root, model, col="rmse_all"):
    fr = [pd.read_csv(f) for f in glob.glob(str(ROOT / root / "rot*_seed*/metrics.csv"))]
    x = pd.concat(fr); x = x[(x.set == "test") & (x.model == model)]
    return round(float(x[col].mean()), 3), int(x["params"].mean()) if "params" in x and x["params"].notna().any() else None


tf, p_tf = mean_test("results/temporal_final/1H", "Transformer (residual)")
fl, p_fl = mean_test("results/temporal_final/1H", "FlatMLP (residual)")
d["transformer"]["params"], d["flat_mlp"]["params"] = p_tf, p_fl
d["final_results_18_runs"] = {
    "NAIRAS V3": mean_test("results/basic", "NAIRAS V3")[0],
    "MLP no history": mean_test("results/basic", "MLP (residual)")[0],
    "Flat MLP 24h": fl,
    "Transformer 24h": tf,
    "Transformer 1h": mean_test("results/temporal_final/5M", "Transformer (residual)")[0],
    "TCN 24h": mean_test("results/temporal_final/1H", "TCN (residual)")[0],
    "CNN-MLP 24h": mean_test("results/temporal_final/1H", "CNN-MLP (residual)")[0],
    "XGBoost 2025 (cited)": 3.671,
    "high_dose": {"NAIRAS V3": mean_test("results/basic", "NAIRAS V3", "rmse_high")[0],
                  "MLP no history": mean_test("results/basic", "MLP (residual)", "rmse_high")[0],
                  "Transformer 24h": mean_test("results/temporal_final/1H", "Transformer (residual)", "rmse_high")[0],
                  "Transformer + high-dose weight 1h": mean_test("results/temporal_final_weighted/5M", "Transformer (residual)", "rmse_high")[0]}}
d["about"] = (d.get("about", "").split(" NAIRAS is not")[0] +
              " NAIRAS is not a network input; the networks learn the correction to it.")
P.write_text(json.dumps(d, separators=(",", ":")))
print("wrote", P, "best epoch", d["best_epoch_transformer"], d["final_results_18_runs"])
