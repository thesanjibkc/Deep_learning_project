"""Complete, beginner-readable preliminary study for the ARMAS 5-minute data.

The script follows one strict experimental rule:

* P1 fits preprocessing and models.
* P2 is used for preliminary validation and model comparison.
* P3 is the final test partition and is scored only after P2 model comparison.

Run this file from the project root. Tables and figures are written to
``outputs/armas_5m_full_study``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# 1. Paths and feature names
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ARMASdatasets_COMPLETE-20261006T110557Z-1-001" / "ARMASdatasets_COMPLETE"
OUT = ROOT / "outputs" / "armas_5m_full_study"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
HIGH_DOSE_THRESHOLD = 15.0
LAG_MINUTES = np.arange(55, -1, -5)  # oldest to current after reversing source arrays

STATIC_FEATURES = [
    "Latitude", "Longitude", "Altitude(Bar)", "Altitude(GPS)",
    "Geomagnetic_latitude", "Geomagnetic_longitude",
    "Geomagnetic_Rc", "Geomagnetic_Lshell",
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


@dataclass
class Partition:
    """All arrays for one flight-separated partition."""

    name: str
    datetime: np.ndarray
    vehicle: np.ndarray
    y: np.ndarray
    nairas: np.ndarray
    static: np.ndarray
    sequence: np.ndarray  # shape: rows x 12 time steps x 35 variables


def to_float(value):
    """Convert JSON numbers to float and JSON null to NumPy missing value."""

    return np.nan if value is None else float(value)


def load_partition(name: str) -> Partition:
    """Read one newline-delimited JSON partition.

    Source temporal arrays store current time at index 0. We reverse them once
    so every tensor runs from 55 minutes before the observation to current time.
    """

    path = DATA_DIR / f"ARMAS_TIMESERIES_5M_20132023_{name}.json"
    datetime, vehicle, y, nairas, static, sequence = [], [], [], [], [], []

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            arrays = np.asarray(
                [[to_float(v) for v in record[key]] for key in TEMPORAL_FEATURES],
                dtype=np.float32,
            )
            if arrays.shape != (35, 12):
                raise ValueError(f"{path.name}, line {line_number}: expected 35 x 12 temporal values")

            datetime.append(record["Datetime"])
            vehicle.append(record.get("Vehicle_ID", ""))
            y.append(to_float(record["ARMAS"]))
            nairas.append(to_float(record["NAIRASV3"]))
            static.append([to_float(record[key]) for key in STATIC_FEATURES])
            sequence.append(arrays[:, ::-1].T)  # rows are now oldest -> current

    return Partition(
        name=name,
        datetime=np.asarray(datetime),
        vehicle=np.asarray(vehicle),
        y=np.asarray(y, dtype=np.float64),
        nairas=np.asarray(nairas, dtype=np.float64),
        static=np.asarray(static, dtype=np.float32),
        sequence=np.asarray(sequence, dtype=np.float32),
    )


# -----------------------------------------------------------------------------
# 2. Feature construction
# -----------------------------------------------------------------------------

def current_features(part: Partition) -> np.ndarray:
    """Eight static variables plus the current value of 35 variables."""

    return np.column_stack([part.static, part.sequence[:, -1, :]])


def residual_current_features(part: Partition) -> np.ndarray:
    """Current features plus NAIRAS V3 for a physics-guided correction."""

    return np.column_stack([current_features(part), part.nairas])


def full_history_features(part: Partition, steps: int = 12) -> np.ndarray:
    """Static variables, the most recent temporal steps, and NAIRAS V3."""

    if steps not in (1, 3, 6, 12):
        raise ValueError("steps must be 1, 3, 6, or 12")
    recent = part.sequence[:, -steps:, :].reshape(len(part.y), steps * 35)
    return np.column_stack([part.static, recent, part.nairas])


def temporal_summary_features(part: Partition) -> np.ndarray:
    """Summarize each temporal variable with six transparent statistics."""

    seq = part.sequence.astype(np.float64)
    time = np.arange(seq.shape[1], dtype=np.float64)
    centered = time - time.mean()
    slope = np.nansum((seq - np.nanmean(seq, axis=1, keepdims=True)) * centered[None, :, None], axis=1)
    slope /= np.sum(centered**2)
    summaries = [
        np.nanmean(seq, axis=1),
        np.nanstd(seq, axis=1),
        np.nanmin(seq, axis=1),
        np.nanmax(seq, axis=1),
        seq[:, -1, :] - seq[:, 0, :],
        slope,
    ]
    return np.column_stack([part.static, *summaries, part.nairas]).astype(np.float32)


CURRENT_NAMES = STATIC_FEATURES + [f"{name}_current" for name in TEMPORAL_FEATURES]
RESIDUAL_CURRENT_NAMES = CURRENT_NAMES + ["NAIRASV3"]
SUMMARY_NAMES = STATIC_FEATURES + [
    f"{name}_{stat}"
    for stat in ("mean", "std", "min", "max", "change", "slope")
    for name in TEMPORAL_FEATURES
] + ["NAIRASV3"]


# -----------------------------------------------------------------------------
# 3. Metrics and model helpers
# -----------------------------------------------------------------------------

def regression_metrics(y_true, prediction, model_name):
    """Return overall, lower-dose, and high-dose metrics."""

    rows = []
    regimes = [
        ("All", np.ones(len(y_true), dtype=bool)),
        ("ARMAS <= 15", y_true <= HIGH_DOSE_THRESHOLD),
        ("ARMAS > 15", y_true > HIGH_DOSE_THRESHOLD),
    ]
    for regime, mask in regimes:
        actual = y_true[mask]
        predicted = prediction[mask]
        rows.append(
            {
                "Model": model_name,
                "Regime": regime,
                "N": int(mask.sum()),
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "R2": r2_score(actual, predicted),
                "Bias": np.mean(predicted - actual),
            }
        )
    return rows


def ridge_model():
    """Ridge regression with train-only imputation and scaling."""

    regressor = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=10.0),
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def mlp_model(hidden=(96, 48), max_iter=90):
    """A small feed-forward network fitted on all of P1."""

    regressor = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            batch_size=512,
            learning_rate_init=5e-4,
            max_iter=max_iter,
            early_stopping=False,
            random_state=RANDOM_SEED,
        ),
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def extra_trees_model():
    """A nonlinear tree ensemble used as a strong tabular reference."""

    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=240,
            min_samples_leaf=3,
            max_features=0.75,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    )


def fit_residual(model, x_train, train: Partition, x_valid, valid: Partition):
    """Fit ARMAS - NAIRAS and reconstruct ARMAS predictions."""

    model.fit(x_train, train.y - train.nairas)
    return valid.nairas + model.predict(x_valid), model


def paired_bootstrap(y, reference, candidate, repeats=600):
    """Bootstrap the candidate-minus-reference RMSE difference on P2."""

    rng = np.random.default_rng(RANDOM_SEED)
    differences = np.empty(repeats)
    n = len(y)
    for i in range(repeats):
        index = rng.integers(0, n, n)
        ref_rmse = np.sqrt(np.mean((reference[index] - y[index]) ** 2))
        new_rmse = np.sqrt(np.mean((candidate[index] - y[index]) ** 2))
        differences[i] = new_rmse - ref_rmse
    return {
        "Mean RMSE difference": differences.mean(),
        "CI low": np.quantile(differences, 0.025),
        "CI high": np.quantile(differences, 0.975),
        "Probability candidate better": np.mean(differences < 0),
    }


def safe_correlation(x, y):
    """Pearson correlation using rows where both values are finite."""

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1]


# -----------------------------------------------------------------------------
# 4. Exploratory analysis
# -----------------------------------------------------------------------------

def dataset_summary(parts):
    rows = []
    for part in parts:
        dates = pd.to_datetime(part.datetime, utc=True)
        rows.append(
            {
                "Partition": part.name,
                "Rows": len(part.y),
                "Start": dates.min().date().isoformat(),
                "End": dates.max().date().isoformat(),
                "Mean ARMAS": np.nanmean(part.y),
                "Median ARMAS": np.nanmedian(part.y),
                "ARMAS > 15": int(np.sum(part.y > HIGH_DOSE_THRESHOLD)),
                "High-dose percent": 100 * np.mean(part.y > HIGH_DOSE_THRESHOLD),
                "Unique vehicle labels": len(np.unique(part.vehicle)),
                "Duplicate timestamps": int(pd.Series(part.datetime).duplicated().sum()),
            }
        )
    return pd.DataFrame(rows)


def missingness_table(parts):
    rows = []
    for part in parts:
        x = current_features(part)
        for j, name in enumerate(CURRENT_NAMES):
            rows.append({"Partition": part.name, "Feature": name, "Missing percent": 100 * np.mean(~np.isfinite(x[:, j]))})
    return pd.DataFrame(rows)


def lag_correlation_table(train: Partition):
    residual = train.y - train.nairas
    rows = []
    for feature_index, feature in enumerate(TEMPORAL_FEATURES):
        for lag_index, lag in enumerate(LAG_MINUTES):
            rows.append(
                {
                    "Feature": feature,
                    "Minutes before observation": int(lag),
                    "Correlation with ARMAS minus NAIRAS": safe_correlation(
                        train.sequence[:, lag_index, feature_index], residual
                    ),
                }
            )
    return pd.DataFrame(rows)


def create_eda_figures(parts, lag_correlations):
    """Create four EDA figures used in the scientific report."""

    plt.style.use("seaborn-v0_8-whitegrid")
    # Target distribution from all observations in P1, P2, and P3.
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    bins = np.linspace(0, 30, 61)
    full_y = np.concatenate([part.y for part in parts])
    ax.hist(np.clip(full_y, None, 30), bins=bins, density=True, histtype="stepfilled",
            alpha=0.32, linewidth=1.8, color="#2E5EAA", edgecolor="#16324F",
            label=f"Full dataset (n={len(full_y):,})")
    ax.axvline(15, color="#9B2226", linestyle="--", label="High-dose threshold")
    ax.set(xlabel="ARMAS effective dose rate (microSv/h; values above 30 clipped)", ylabel="Density",
           title="Measured dose-rate distribution - full dataset")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "figure01_target_distribution.png", dpi=220)
    plt.close(fig)

    # NAIRAS relation and residual behavior on P2.
    p2 = next(p for p in parts if p.name == "P2")
    rng = np.random.default_rng(RANDOM_SEED)
    sample = rng.choice(len(p2.y), min(6500, len(p2.y)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    axes[0].hexbin(p2.y[sample], p2.nairas[sample], gridsize=42, mincnt=1, cmap="Blues")
    limit = np.nanpercentile(np.r_[p2.y[sample], p2.nairas[sample]], 99.5)
    axes[0].plot([0, limit], [0, limit], color="#9B2226", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Observed ARMAS", ylabel="NAIRAS V3", title="P2 observed versus physics estimate",
                xlim=(0, limit), ylim=(0, limit))
    axes[1].scatter(p2.y[sample], p2.y[sample] - p2.nairas[sample], s=7, alpha=0.22, color="#2E5EAA")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].axvline(15, color="#9B2226", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Observed ARMAS", ylabel="ARMAS - NAIRAS", title="Physics-model discrepancy")
    fig.tight_layout()
    fig.savefig(OUT / "figure02_nairas_relationship.png", dpi=220)
    plt.close(fig)

    # Missingness in the most incomplete current-time variables.
    missing = missingness_table(parts)
    pivot = missing.pivot(index="Feature", columns="Partition", values="Missing percent")
    top = pivot.max(axis=1).sort_values(ascending=False).head(15).index
    plot_data = pivot.loc[top, ["P1", "P2", "P3"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    image = ax.imshow(plot_data.to_numpy(), aspect="auto", cmap="Blues", vmin=0,
                      vmax=max(1, np.nanmax(plot_data.to_numpy())))
    ax.set_xticks(range(3), plot_data.columns)
    ax.set_yticks(range(len(plot_data)), plot_data.index)
    ax.set_title("Current-time missingness for the 15 most incomplete inputs")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Missing percent")
    fig.tight_layout()
    fig.savefig(OUT / "figure03_missingness.png", dpi=220)
    plt.close(fig)

    # Top feature-by-lag correlations with the P1 residual.
    ranking = (lag_correlations.assign(abs_corr=lambda d: d["Correlation with ARMAS minus NAIRAS"].abs())
               .groupby("Feature")["abs_corr"].max().sort_values(ascending=False))
    top_features = ranking.head(12).index.tolist()
    heat = (lag_correlations[lag_correlations["Feature"].isin(top_features)]
            .pivot(index="Feature", columns="Minutes before observation",
                   values="Correlation with ARMAS minus NAIRAS")
            .loc[top_features, LAG_MINUTES])
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    bound = max(0.05, np.nanmax(np.abs(heat.to_numpy())))
    image = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    ax.set_xticks(range(len(LAG_MINUTES)), LAG_MINUTES)
    ax.set_yticks(range(len(top_features)), top_features)
    ax.set_xlabel("Minutes before ARMAS observation")
    ax.set_title("P1 correlation with ARMAS minus NAIRAS")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(OUT / "figure04_lag_correlation.png", dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Validation experiments
# -----------------------------------------------------------------------------

def run_validation_experiments(train: Partition, valid: Partition):
    """Fit all P1 models and return P2 predictions and supporting tables."""

    y_train, y_valid = train.y, valid.y
    current_train, current_valid = current_features(train), current_features(valid)
    residual_current_train = residual_current_features(train)
    residual_current_valid = residual_current_features(valid)
    summary_train, summary_valid = temporal_summary_features(train), temporal_summary_features(valid)

    predictions = {
        "P1 mean": np.full_like(y_valid, np.nanmean(y_train)),
        "NAIRAS V3": valid.nairas.copy(),
    }
    fitted = {}

    # Transparent linear baselines.
    model = ridge_model()
    model.fit(current_train, y_train)
    predictions["Current ridge direct"] = model.predict(current_valid)
    fitted["Current ridge direct"] = model

    predictions["Current ridge residual"], fitted["Current ridge residual"] = fit_residual(
        ridge_model(), residual_current_train, train, residual_current_valid, valid
    )
    static_train = np.column_stack([train.static, train.nairas])
    static_valid = np.column_stack([valid.static, valid.nairas])
    predictions["Static ridge residual"], fitted["Static ridge residual"] = fit_residual(
        ridge_model(), static_train, train, static_valid, valid
    )
    predictions["Full-history ridge residual"], fitted["Full-history ridge residual"] = fit_residual(
        ridge_model(), full_history_features(train, 12), train, full_history_features(valid, 12), valid
    )

    # Nonlinear references.
    direct_trees = extra_trees_model()
    direct_trees.fit(current_train, y_train)
    predictions["Current Extra Trees direct"] = direct_trees.predict(current_valid)
    fitted["Current Extra Trees direct"] = direct_trees

    predictions["Temporal-summary Extra Trees residual"], fitted["Temporal-summary Extra Trees residual"] = fit_residual(
        extra_trees_model(), summary_train, train, summary_valid, valid
    )

    # Neural baselines. Warnings are recorded in the run summary rather than hidden.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        current_mlp = mlp_model(hidden=(64, 32), max_iter=90)
        current_mlp.fit(current_train, y_train)
        predictions["Current MLP direct"] = current_mlp.predict(current_valid)
        fitted["Current MLP direct"] = current_mlp

        temporal_mlp = mlp_model(hidden=(96, 48), max_iter=90)
        predictions["Temporal-summary MLP residual"], fitted["Temporal-summary MLP residual"] = fit_residual(
            temporal_mlp, summary_train, train, summary_valid, valid
        )
        warning_rows = [{"Warning": str(item.message)} for item in caught]
    pd.DataFrame(warning_rows).to_csv(OUT / "neural_training_warnings.csv", index=False)

    # Weighted residual boosting: the only difference is the P1 high-dose weight.
    weight_rows = []
    weighted_models = {}
    for weight in (1.0, 2.0, 4.0):
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=260,
            max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_SEED,
        )
        sample_weight = np.where(train.y > HIGH_DOSE_THRESHOLD, weight, 1.0)
        model.fit(residual_current_train, train.y - train.nairas, sample_weight=sample_weight)
        name = f"Weighted residual boosting w={weight:g}"
        prediction = valid.nairas + model.predict(residual_current_valid)
        predictions[name] = prediction
        fitted[name] = model
        weighted_models[weight] = model
        for row in regression_metrics(valid.y, prediction, name):
            weight_rows.append({"High-dose weight": weight, **row})

    all_metric_rows = []
    for name, prediction in predictions.items():
        all_metric_rows.extend(regression_metrics(valid.y, prediction, name))
    metrics = pd.DataFrame(all_metric_rows)
    weight_results = pd.DataFrame(weight_rows)

    # Ridge feature-group ablation.
    ablations = {
        "Static + NAIRAS": (
            np.column_stack([train.static, train.nairas]),
            np.column_stack([valid.static, valid.nairas]),
        ),
        "Environmental current + NAIRAS": (
            np.column_stack([train.sequence[:, -1, :], train.nairas]),
            np.column_stack([valid.sequence[:, -1, :], valid.nairas]),
        ),
        "Static + current + NAIRAS": (residual_current_train, residual_current_valid),
        "Temporal summaries + NAIRAS": (summary_train, summary_valid),
        "Full 12-step history + NAIRAS": (full_history_features(train, 12), full_history_features(valid, 12)),
    }
    ablation_rows = []
    for name, (x_train, x_valid) in ablations.items():
        prediction, _ = fit_residual(ridge_model(), x_train, train, x_valid, valid)
        for row in regression_metrics(valid.y, prediction, name):
            ablation_rows.append({"Ablation": name, **row})

    # Temporal-window ablation with the same ridge residual model.
    window_rows = []
    for steps in (1, 3, 6, 12):
        prediction, _ = fit_residual(
            ridge_model(), full_history_features(train, steps), train,
            full_history_features(valid, steps), valid,
        )
        for row in regression_metrics(valid.y, prediction, f"{steps} steps"):
            window_rows.append({"Steps": steps, "Minutes of context": (steps - 1) * 5, **row})

    return {
        "predictions": predictions,
        "fitted": fitted,
        "metrics": metrics,
        "weight_results": weight_results,
        "ablation_results": pd.DataFrame(ablation_rows),
        "window_results": pd.DataFrame(window_rows),
        "residual_current_train": residual_current_train,
        "residual_current_valid": residual_current_valid,
    }


def predict_fitted_models(study, train: Partition, test: Partition):
    """Apply P1-fitted models to P3 without refitting or model selection."""

    predictions = {
        "P1 mean": np.full_like(test.y, np.nanmean(train.y)),
        "NAIRAS V3": test.nairas.copy(),
    }
    current = current_features(test)
    residual_current = residual_current_features(test)
    static = np.column_stack([test.static, test.nairas])
    summary = temporal_summary_features(test)
    full_history = full_history_features(test, 12)
    for name, model in study["fitted"].items():
        if name in {"Current ridge direct", "Current Extra Trees direct", "Current MLP direct"}:
            predictions[name] = model.predict(current)
        elif name == "Static ridge residual":
            predictions[name] = test.nairas + model.predict(static)
        elif name == "Full-history ridge residual":
            predictions[name] = test.nairas + model.predict(full_history)
        elif name in {"Temporal-summary Extra Trees residual", "Temporal-summary MLP residual"}:
            predictions[name] = test.nairas + model.predict(summary)
        else:
            predictions[name] = test.nairas + model.predict(residual_current)
    rows = []
    for name, prediction in predictions.items():
        rows.extend(regression_metrics(test.y, prediction, name))
    return predictions, pd.DataFrame(rows)


def create_experiment_figures(valid: Partition, study):
    metrics = study["metrics"]
    predictions = study["predictions"]
    weight_results = study["weight_results"]
    windows = study["window_results"]

    overall = metrics[metrics["Regime"] == "All"].sort_values("RMSE")
    high = metrics[metrics["Regime"] == "ARMAS > 15"].set_index("Model")

    # Overall and high-dose RMSE comparison.
    selected_names = overall.head(9)["Model"].tolist()
    y = np.arange(len(selected_names))
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(y + 0.19, [overall.set_index("Model").loc[n, "RMSE"] for n in selected_names],
            height=0.38, label="All P2", color="#2E5EAA")
    ax.barh(y - 0.19, [high.loc[n, "RMSE"] for n in selected_names],
            height=0.38, label="ARMAS > 15", color="#BB6B3B")
    ax.set_yticks(y, selected_names)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (microSv/h)")
    ax.set_title("P2 validation performance by radiation regime")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure05_model_comparison.png", dpi=220)
    plt.close(fig)

    # Temporal-window ablation.
    all_windows = windows[windows["Regime"] == "All"]
    high_windows = windows[windows["Regime"] == "ARMAS > 15"]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.plot(all_windows["Minutes of context"], all_windows["RMSE"], marker="o", label="All P2")
    ax.plot(high_windows["Minutes of context"], high_windows["RMSE"], marker="o", label="ARMAS > 15")
    ax.set(xlabel="Temporal context before current observation (minutes)", ylabel="RMSE (microSv/h)",
           title="Ridge residual temporal-window ablation")
    ax.set_xticks(all_windows["Minutes of context"])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure06_temporal_window_ablation.png", dpi=220)
    plt.close(fig)

    # High-dose weight trade-off.
    weight_all = weight_results[weight_results["Regime"] == "All"].sort_values("High-dose weight")
    weight_high = weight_results[weight_results["Regime"] == "ARMAS > 15"].sort_values("High-dose weight")
    fig, ax1 = plt.subplots(figsize=(6.8, 3.9))
    ax1.plot(weight_all["High-dose weight"], weight_all["RMSE"], marker="o", color="#2E5EAA", label="All P2 RMSE")
    ax1.plot(weight_high["High-dose weight"], weight_high["RMSE"], marker="o", color="#BB6B3B", label="High-dose RMSE")
    ax1.set(xlabel="P1 high-dose training weight", ylabel="RMSE (microSv/h)",
            title="Accuracy trade-off from imbalance weighting")
    ax1.set_xticks([1, 2, 4])
    ax1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure07_weight_tradeoff.png", dpi=220)
    plt.close(fig)

    # Pick the lowest-overall-RMSE learned model for descriptive diagnostics.
    learned = overall[~overall["Model"].isin(["P1 mean", "NAIRAS V3"])]
    best_name = learned.iloc[0]["Model"]
    best_prediction = predictions[best_name]
    pd.DataFrame({"Selected model": [best_name]}).to_csv(OUT / "selected_preliminary_model.csv", index=False)

    rng = np.random.default_rng(RANDOM_SEED)
    sample = rng.choice(len(valid.y), min(7000, len(valid.y)), replace=False)
    limit = np.nanpercentile(np.r_[valid.y[sample], best_prediction[sample]], 99.5)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    axes[0].hexbin(valid.y[sample], best_prediction[sample], gridsize=44, mincnt=1, cmap="Blues")
    axes[0].plot([0, limit], [0, limit], color="#9B2226", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Observed ARMAS", ylabel="Predicted ARMAS", title=best_name, xlim=(0, limit), ylim=(0, limit))
    residual = best_prediction - valid.y
    axes[1].scatter(valid.y[sample], residual[sample], s=7, alpha=0.20, color="#2E5EAA")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].axvline(15, color="#9B2226", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Observed ARMAS", ylabel="Prediction error", title="Residual structure")
    fig.tight_layout()
    fig.savefig(OUT / "figure08_best_model_diagnostics.png", dpi=220)
    plt.close(fig)

    # Dose-bin RMSE reveals where aggregate scores hide error.
    bins = [0, 5, 10, 15, 20, 30, np.inf]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-30", ">30"]
    dose_rows = []
    for left, right, label in zip(bins[:-1], bins[1:], labels):
        mask = (valid.y >= left) & (valid.y < right)
        for name, prediction in (("NAIRAS V3", valid.nairas), (best_name, best_prediction)):
            dose_rows.append({"Dose bin": label, "Model": name, "N": int(mask.sum()),
                              "RMSE": np.sqrt(np.mean((prediction[mask] - valid.y[mask]) ** 2))})
    dose = pd.DataFrame(dose_rows)
    dose.to_csv(OUT / "dose_bin_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    width = 0.38
    x = np.arange(len(labels))
    for offset, (name, color) in zip((-width / 2, width / 2), (("NAIRAS V3", "#6C8EAD"), (best_name, "#2E5EAA"))):
        values = dose[dose["Model"] == name].set_index("Dose bin").loc[labels, "RMSE"]
        ax.bar(x + offset, values, width=width, label=name, color=color)
    ax.set_xticks(x, labels)
    ax.set(xlabel="Observed ARMAS dose-rate bin (microSv/h)", ylabel="RMSE (microSv/h)",
           title="P2 error by observed dose-rate bin")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure09_dose_bin_rmse.png", dpi=220)
    plt.close(fig)

    # Interpret the best weighted boosting model with permutation importance.
    weight_all = weight_results[weight_results["Regime"] == "All"]
    best_weight = float(weight_all.loc[weight_all["RMSE"].idxmin(), "High-dose weight"])
    weighted_name = f"Weighted residual boosting w={best_weight:g}"
    model = study["fitted"][weighted_name]
    subset = rng.choice(len(valid.y), min(4500, len(valid.y)), replace=False)
    importance = permutation_importance(
        model,
        study["residual_current_valid"][subset],
        (valid.y - valid.nairas)[subset],
        scoring="neg_root_mean_squared_error",
        n_repeats=4,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "Feature": RESIDUAL_CURRENT_NAMES,
        "RMSE increase": importance.importances_mean,
        "Standard deviation": importance.importances_std,
    }).sort_values("RMSE increase", ascending=False)
    importance_df.to_csv(OUT / "permutation_importance.csv", index=False)
    top = importance_df.head(15).sort_values("RMSE increase")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(top["Feature"], top["RMSE increase"], xerr=top["Standard deviation"], color="#2E5EAA", alpha=0.9)
    ax.set(xlabel="Increase in residual RMSE after permutation", title=f"P2 permutation importance for {weighted_name}")
    fig.tight_layout()
    fig.savefig(OUT / "figure10_permutation_importance.png", dpi=220)
    plt.close(fig)

    return best_name, best_prediction


# -----------------------------------------------------------------------------
# 6. Main program
# -----------------------------------------------------------------------------

def main():
    print("Loading P1, P2, and P3 ...")
    p1, p2, p3 = (load_partition(name) for name in ("P1", "P2", "P3"))
    parts = [p1, p2, p3]

    summary = dataset_summary(parts)
    missing = missingness_table(parts)
    lag_correlations = lag_correlation_table(p1)
    summary.to_csv(OUT / "dataset_summary.csv", index=False)
    missing.to_csv(OUT / "missingness_by_partition.csv", index=False)
    lag_correlations.to_csv(OUT / "p1_feature_by_lag_correlations.csv", index=False)
    create_eda_figures(parts, lag_correlations)

    print("Training preliminary models on P1 and validating on P2 ...")
    study = run_validation_experiments(p1, p2)
    study["metrics"].to_csv(OUT / "p2_model_metrics.csv", index=False)
    study["weight_results"].to_csv(OUT / "p2_weight_sweep.csv", index=False)
    study["ablation_results"].to_csv(OUT / "p2_feature_ablation.csv", index=False)
    study["window_results"].to_csv(OUT / "p2_temporal_window_ablation.csv", index=False)

    print("Scoring the P2-selected, P1-fitted models once on P3 ...")
    test_predictions, test_metrics = predict_fitted_models(study, p1, p3)
    test_metrics.to_csv(OUT / "p3_test_metrics.csv", index=False)
    pd.DataFrame({"ARMAS": p3.y, **test_predictions}).to_csv(
        OUT / "p3_test_predictions.csv", index=False
    )

    best_name, best_prediction = create_experiment_figures(p2, study)
    bootstrap = paired_bootstrap(p2.y, p2.nairas, best_prediction)
    pd.DataFrame([{"Candidate": best_name, "Reference": "NAIRAS V3", **bootstrap}]).to_csv(
        OUT / "paired_bootstrap_vs_nairas.csv", index=False
    )

    # Compact console summary for a quick correctness check.
    overall = study["metrics"][study["metrics"]["Regime"] == "All"].sort_values("RMSE")
    high = study["metrics"][study["metrics"]["Regime"] == "ARMAS > 15"].sort_values("RMSE")
    print("\nDataset summary\n", summary.to_string(index=False))
    print("\nBest overall P2 models\n", overall[["Model", "RMSE", "MAE", "R2", "Bias"]].head(10).to_string(index=False))
    print("\nBest high-dose P2 models\n", high[["Model", "RMSE", "MAE", "R2", "Bias"]].head(10).to_string(index=False))
    print("\nPaired bootstrap versus NAIRAS V3\n", bootstrap)
    test_overall = test_metrics[test_metrics["Regime"] == "All"].sort_values("RMSE")
    print("\nP3 final-test models\n", test_overall[["Model", "RMSE", "MAE", "R2", "Bias"]].head(10).to_string(index=False))
    print(f"\nCompleted fixed P1/P2/P3 train-validation-test workflow. Outputs: {OUT}")


if __name__ == "__main__":
    main()
