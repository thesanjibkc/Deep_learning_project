"""Reproducible preliminary analysis for the 5-minute ARMAS project.

P1 is used for fitting, P2 for validation and model comparison, and P3 for
the final test after model fitting and validation are complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ARMASdatasets_COMPLETE-20261006T110557Z-1-001" / "ARMASdatasets_COMPLETE"
OUT = ROOT / "outputs" / "preliminary_5m"
OUT.mkdir(parents=True, exist_ok=True)

STATIC_FEATURES = [
    "Latitude",
    "Longitude",
    "Altitude(Bar)",
    "Altitude(GPS)",
    "Geomagnetic_latitude",
    "Geomagnetic_longitude",
    "Geomagnetic_Rc",
    "Geomagnetic_Lshell",
]
TEMPORAL_FEATURES = [
    "NM_NEWK", "NM_OULU", "NM_THUL", "NM_SOPO", "SXR_short", "SXR_long",
    "Particles_P1", "Particles_P5", "Particles_P10", "Particles_P30",
    "Particles_P50", "Particles_P100", "Particles_E20", "SW_B", "SW_Bx",
    "SW_By", "SW_Bz", "SW_V", "SW_Vx", "SW_Vy", "SW_Vz", "SW_density",
    "SW_temperature", "SW_pressure", "Index_Kp", "Index_Dst", "Index_Ap",
    "Solar_sunspots", "Solar_f107", "Solar_NPF", "Solar_SPF", "Solar_APF",
    "Solar_NPF20", "Solar_SPF20", "Solar_APF20",
]


def numeric(value):
    return np.nan if value is None else float(value)


def load_partition(name: str, keep_features: bool = True):
    path = DATA_DIR / f"ARMAS_TIMESERIES_5M_20132023_{name}.json"
    rows = []
    current, temporal = [], []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            y = numeric(record["ARMAS"])
            n3 = numeric(record["NAIRASV3"])
            rows.append(
                {
                    "partition": name,
                    "datetime": record["Datetime"],
                    "armas": y,
                    "nairas_v3": n3,
                    "high": y > 15,
                }
            )
            if keep_features:
                static = [numeric(record[k]) for k in STATIC_FEATURES]
                arrays = [[numeric(v) for v in record[k]] for k in TEMPORAL_FEATURES]
                if any(len(a) != 12 for a in arrays):
                    raise ValueError(f"{name} contains a non-12-step temporal array")
                current.append(static + [a[0] for a in arrays])
                # Arrange each row as twelve time steps with 35 variables, oldest to current.
                temporal.append(static + np.asarray(arrays, dtype=np.float32)[:, ::-1].T.ravel().tolist())
    frame = pd.DataFrame(rows)
    if not keep_features:
        return frame, None, None
    return frame, np.asarray(current, dtype=np.float32), np.asarray(temporal, dtype=np.float32)


def metrics(y_true, y_pred, model):
    result = []
    for regime, mask in (
        ("All", np.ones(len(y_true), dtype=bool)),
        ("ARMAS <= 15", y_true <= 15),
        ("ARMAS > 15", y_true > 15),
    ):
        yt, yp = y_true[mask], y_pred[mask]
        result.append(
            {
                "Model": model,
                "Regime": regime,
                "N": int(mask.sum()),
                "MAE": mean_absolute_error(yt, yp),
                "RMSE": np.sqrt(mean_squared_error(yt, yp)),
                "R2": r2_score(yt, yp),
                "Bias": float(np.mean(yp - yt)),
            }
        )
    return result


def direct_estimator(kind: str):
    if kind == "ridge":
        reg = Ridge(alpha=10.0)
    else:
        reg = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=35,
            early_stopping=False,
            random_state=42,
        )
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), reg)
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


def residual_estimator(kind: str):
    if kind == "ridge":
        reg = Ridge(alpha=10.0)
    else:
        reg = MLPRegressor(
            hidden_layer_sizes=(96, 48),
            activation="relu",
            solver="adam",
            batch_size=512,
            learning_rate_init=8e-4,
            max_iter=35,
            early_stopping=False,
            random_state=42,
        )
    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), reg)
    return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())


def main():
    p1, x1_current, x1_temporal = load_partition("P1", True)
    p2, x2_current, x2_temporal = load_partition("P2", True)
    p3, x3_current, x3_temporal = load_partition("P3", True)

    frames = [p1, p2, p3]
    summaries = []
    for df in frames:
        summaries.append(
            {
                "Partition": df["partition"].iloc[0],
                "Rows": len(df),
                "Start": pd.to_datetime(df["datetime"], utc=True).min().date().isoformat(),
                "End": pd.to_datetime(df["datetime"], utc=True).max().date().isoformat(),
                "Mean ARMAS": df["armas"].mean(),
                "Median ARMAS": df["armas"].median(),
                "ARMAS > 15": int(df["high"].sum()),
                "High-dose percent": 100 * df["high"].mean(),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "dataset_summary.csv", index=False)

    y1 = p1["armas"].to_numpy()
    y2 = p2["armas"].to_numpy()
    n1 = p1["nairas_v3"].to_numpy()
    n2 = p2["nairas_v3"].to_numpy()
    y3 = p3["armas"].to_numpy()
    n3 = p3["nairas_v3"].to_numpy()
    results = []
    predictions = {
        "Training mean": np.full_like(y2, y1.mean()),
        "NAIRAS V3": n2,
    }

    ridge_direct = direct_estimator("ridge")
    ridge_direct.fit(x1_current, y1)
    predictions["Current ridge"] = ridge_direct.predict(x2_current)

    ridge_residual = residual_estimator("ridge")
    ridge_residual.fit(x1_current, y1 - n1)
    predictions["Residual ridge"] = n2 + ridge_residual.predict(x2_current)

    temporal_ridge = residual_estimator("ridge")
    temporal_ridge.fit(x1_temporal, y1 - n1)
    predictions["Temporal residual ridge"] = n2 + temporal_ridge.predict(x2_temporal)

    mlp_direct = direct_estimator("mlp")
    mlp_direct.fit(x1_current, y1)
    predictions["Current MLP"] = mlp_direct.predict(x2_current)

    temporal_residual = residual_estimator("mlp")
    temporal_residual.fit(x1_temporal, y1 - n1)
    predictions["Temporal residual MLP"] = n2 + temporal_residual.predict(x2_temporal)

    for model, pred in predictions.items():
        results.extend(metrics(y2, pred, model))
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUT / "p2_validation_metrics.csv", index=False)

    test_predictions = {
        "Training mean": np.full_like(y3, y1.mean()),
        "NAIRAS V3": n3,
        "Current ridge": ridge_direct.predict(x3_current),
        "Residual ridge": n3 + ridge_residual.predict(x3_current),
        "Temporal residual ridge": n3 + temporal_ridge.predict(x3_temporal),
        "Current MLP": mlp_direct.predict(x3_current),
        "Temporal residual MLP": n3 + temporal_residual.predict(x3_temporal),
    }
    test_results = []
    for model, pred in test_predictions.items():
        test_results.extend(metrics(y3, pred, model))
    test_metrics_df = pd.DataFrame(test_results)
    test_metrics_df.to_csv(OUT / "p3_test_metrics.csv", index=False)
    pd.DataFrame({"ARMAS": y3, **test_predictions}).to_csv(OUT / "p3_test_predictions.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    bins = np.linspace(0, 30, 61)
    full_armas = pd.concat([df["armas"] for df in frames], ignore_index=True)
    ax.hist(full_armas.clip(upper=30), bins=bins, histtype="stepfilled", linewidth=1.8,
            alpha=0.32, edgecolor="#16324F", color="#2E5EAA", density=True,
            label=f"Full dataset (n={len(full_armas):,})")
    ax.axvline(15, color="#9B2226", linestyle="--", linewidth=1.5, label="High-dose threshold")
    ax.set(xlabel="ARMAS effective dose rate (microSv/h; values above 30 clipped)",
           ylabel="Density", title="Measured dose-rate distribution - full dataset")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "figure1_partition_distributions.png", dpi=220)
    plt.close(fig)

    overall = metrics_df[metrics_df["Regime"] == "All"].sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    bar_colors = ["#2E5EAA" if "Temporal" in m else "#6C8EAD" for m in overall["Model"]]
    bars = ax.barh(overall["Model"], overall["RMSE"], color=bar_colors)
    ax.invert_yaxis()
    ax.set(xlabel="P2 validation RMSE (microSv/h)", title="Preliminary validation performance")
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figure2_validation_rmse.png", dpi=220)
    plt.close(fig)

    high = metrics_df[metrics_df["Regime"] == "ARMAS > 15"].sort_values("RMSE")
    high[["Model", "N", "MAE", "RMSE", "R2", "Bias"]].to_csv(
        OUT / "p2_high_dose_metrics.csv", index=False
    )
    print(summary.to_string(index=False))
    print("\nP2 validation metrics\n", metrics_df.to_string(index=False))
    print("\nP3 test metrics\n", test_metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
