import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks" / "exploration"
OUT.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(True),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


exploration = notebook([
    md(r"""
    # ARMAS 5-minute time-series exploration

    This notebook is the data-first checkpoint for the project **Temporal Deep Learning for Aviation Effective Dose-Rate Prediction**.

    It answers four questions before modeling:

    1. Do P1, P2, and P3 have the expected schema and 12-step histories?
    2. How do the ARMAS target and the high-radiation regime (`ARMAS > 15 µSv/h`) differ by partition?
    3. How strong is the NAIRAS V3 physics benchmark?
    4. Are there obvious missingness, non-finite values, duplicates, or distribution shifts?

    Modeling contract: P1 is training, P2 is validation, and P3 is untouched final testing in the first modeling pass. `Vehicle_ID`, `NAIRASV2`, and `NAIRASV3` are never predictors. NAIRAS V3 is only a benchmark.
    """),
    md("""
    ## 0. Setup

    Run locally from the repository root, or set `DATA_DIR` to the dataset folder in Google Drive. The loader streams JSON Lines and stores only scalar fields plus the current (last) value of each temporal sequence, keeping exploration memory-friendly.
    """),
    code(r"""
    # Uncomment only if your environment is missing these packages.
    # %pip install -q pandas numpy matplotlib seaborn scikit-learn

    import json
    import math
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    sns.set_theme(style="whitegrid", context="notebook")
    pd.set_option("display.max_columns", 100)
    RNG = np.random.default_rng(42)
    """),
    code(r"""
    # Local default. In Colab, point this to the folder in Drive instead.
    DATA_DIR = Path("ARMASdatasets_COMPLETE-20261006T110557Z-1-001/ARMASdatasets_COMPLETE")
    # Example Colab path:
    # from google.colab import drive; drive.mount('/content/drive')
    # DATA_DIR = Path('/content/drive/MyDrive/ARMASdatasets_COMPLETE')

    FILES = {
        p: DATA_DIR / f"ARMAS_TIMESERIES_5M_20132023_{p}.json"
        for p in ("P1", "P2", "P3")
    }
    assert all(path.exists() for path in FILES.values()), (
        "Set DATA_DIR to the folder containing the three ARMAS_TIMESERIES_5M files."
    )
    FILES
    """),
    md("""
    ## 1. Stream and validate the partitions

    Prediction: each record should contain 8 static variables, 35 list-valued environmental variables, and every list should have length 12. A cross-check against the matching static file shows that index `0` is the current-time value; later indices step backward into the preceding hour.
    """),
    code(r"""
    STATIC_FEATURES = [
        "Latitude", "Longitude", "Altitude(Bar)", "Altitude(GPS)",
        "Geomagnetic_latitude", "Geomagnetic_longitude",
        "Geomagnetic_Rc", "Geomagnetic_Lshell",
    ]
    EXCLUDED_PREDICTORS = ["Vehicle_ID", "NAIRASV2", "NAIRASV3"]
    META_COLUMNS = ["Datetime", "ARMAS", "NAIRASV3", "NAIRASV2", "Vehicle_ID"]

    def stream_partition(path, sequence_sample_size=500):
        compact_rows = []
        sequence_sample = []
        temporal_features = None
        length_sets = defaultdict(set)
        nonfinite = defaultdict(int)
        keys = None

        with path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                keys = keys or list(row)
                if temporal_features is None:
                    temporal_features = [k for k, v in row.items() if isinstance(v, list)]

                compact = {k: row[k] for k in META_COLUMNS + STATIC_FEATURES}
                for feature in temporal_features:
                    values = row[feature]
                    length_sets[feature].add(len(values))
                    arr = np.asarray(values, dtype=np.float64)
                    nonfinite[feature] += int((~np.isfinite(arr)).sum())
                    compact[f"{feature}__current"] = values[0] if values else np.nan
                compact_rows.append(compact)

                # Reservoir sample: representative sequences without retaining the full raw dataset.
                if len(sequence_sample) < sequence_sample_size:
                    sequence_sample.append({k: row[k] for k in temporal_features})
                else:
                    j = int(RNG.integers(0, row_index + 1))
                    if j < sequence_sample_size:
                        sequence_sample[j] = {k: row[k] for k in temporal_features}

        frame = pd.DataFrame(compact_rows)
        frame["Datetime"] = pd.to_datetime(frame["Datetime"], utc=True)
        audit = {
            "keys": keys,
            "temporal_features": temporal_features,
            "length_sets": dict(length_sets),
            "nonfinite": dict(nonfinite),
        }
        return frame, sequence_sample, audit

    compact, samples, audits = {}, {}, {}
    for partition, path in FILES.items():
        compact[partition], samples[partition], audits[partition] = stream_partition(path)
        print(partition, compact[partition].shape)
    """),
    code(r"""
    TEMPORAL_FEATURES = audits["P1"]["temporal_features"]
    assert len(STATIC_FEATURES) == 8
    assert len(TEMPORAL_FEATURES) == 35
    assert all(audits[p]["temporal_features"] == TEMPORAL_FEATURES for p in FILES)
    assert all(lengths == {12} for p in FILES for lengths in audits[p]["length_sets"].values())
    assert all(count == 0 for p in FILES for count in audits[p]["nonfinite"].values())
    assert all(compact[p].isna().sum().sum() == 0 for p in FILES)
    print("PASS: 8 static features, 35 temporal features, 12 steps, no null/non-finite values.")
    """),
    code(r"""
    # Verify temporal ordering against the independently supplied static dataset.
    static_path = DATA_DIR / "ARMAS_STATIC_20132023_P1.json"
    if static_path.exists():
        with static_path.open("r", encoding="utf-8") as handle:
            static_first = json.loads(next(handle))
        with FILES["P1"].open("r", encoding="utf-8") as handle:
            sequence_first = json.loads(next(handle))
        assert static_first["Datetime"] == sequence_first["Datetime"]
        assert all(np.isclose(static_first[f], sequence_first[f][0], equal_nan=True)
                   for f in TEMPORAL_FEATURES)
        print("PASS: sequence index 0 matches the current-time static values.")
    else:
        print("Static file not found; current-time ordering could not be independently rechecked.")
    """),
    md("""
    Reflection: if this assertion fails on a future dataset version, stop. Do not silently pad, truncate, or impute until the failure pattern is understood and a train-only preprocessing rule is chosen.
    """),
    md("""
    ## 2. Partition overview and target distribution
    """),
    code(r"""
    overview = []
    for partition, df in compact.items():
        overview.append({
            "partition": partition,
            "rows": len(df),
            "start": df["Datetime"].min(),
            "end": df["Datetime"].max(),
            "ARMAS_mean": df["ARMAS"].mean(),
            "ARMAS_median": df["ARMAS"].median(),
            "ARMAS_p95": df["ARMAS"].quantile(0.95),
            "ARMAS_max": df["ARMAS"].max(),
            "high_n": (df["ARMAS"] > 15).sum(),
            "high_pct": 100 * (df["ARMAS"] > 15).mean(),
        })
    overview = pd.DataFrame(overview).set_index("partition")
    overview.round(3)
    """),
    code(r"""
    full_data = pd.concat(compact.values(), ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(full_data["ARMAS"], bins=60, stat="density", element="step",
                 fill=True, alpha=.25, color="#2E5EAA", label="Full dataset", ax=axes[0])
    sns.ecdfplot(full_data, x="ARMAS", color="#2E5EAA", label="Full dataset", ax=axes[1])
    axes[0].axvline(15, color="black", linestyle="--", label="high-dose threshold")
    axes[0].set(xlabel="ARMAS (µSv/h)", title="Target distribution - full dataset")
    axes[1].axvline(15, color="black", linestyle="--")
    axes[1].set(xlabel="ARMAS (µSv/h)", title="Empirical CDF - full dataset")
    axes[0].legend(); axes[1].legend();
    plt.tight_layout()
    """),
    md("""
    Questions:

    - Are the three curves close enough that one partition is not an unusually easy or hard test set?
    - Does the high-dose fraction support reporting both overall and `ARMAS > 15` metrics?
    """),
    md("""
    ## 3. NAIRAS V3 benchmark

    This is a benchmark only. It is not included in `X`, and NAIRAS V2 is neither a predictor nor the primary benchmark.
    """),
    code(r"""
    def regression_metrics(y_true, y_pred):
        return {
            "n": len(y_true),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred),
            "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
        }

    benchmark_rows = []
    for partition, df in compact.items():
        for regime, mask in {
            "all": np.ones(len(df), dtype=bool),
            "ARMAS > 15": df["ARMAS"].to_numpy() > 15,
        }.items():
            result = regression_metrics(df.loc[mask, "ARMAS"], df.loc[mask, "NAIRASV3"])
            benchmark_rows.append({"partition": partition, "regime": regime, **result})
    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.round(3)
    """),
    code(r"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, (partition, df) in zip(axes, compact.items()):
        sample = df.sample(min(6000, len(df)), random_state=42)
        ax.hexbin(sample["ARMAS"], sample["NAIRASV3"], gridsize=45, mincnt=1, cmap="viridis")
        limits = [0, max(sample["ARMAS"].max(), sample["NAIRASV3"].max())]
        ax.plot(limits, limits, "r--", linewidth=1)
        ax.set(title=partition, xlabel="Measured ARMAS", ylabel="NAIRAS V3")
    fig.suptitle("Physics benchmark versus measurements")
    plt.tight_layout()
    """),
    md("""
    ## 4. Static/current feature relationships and shift

    Correlation is descriptive, not a feature-selection rule. The deep models will receive all proposal-approved inputs.
    """),
    code(r"""
    train_df = compact["P1"]
    numeric_predictors = STATIC_FEATURES + [f"{f}__current" for f in TEMPORAL_FEATURES]
    corr = train_df[numeric_predictors + ["ARMAS"]].corr(method="spearman")["ARMAS"].drop("ARMAS")
    corr.reindex(corr.abs().sort_values(ascending=False).index).head(20).to_frame("Spearman_with_ARMAS")
    """),
    code(r"""
    top = corr.abs().nlargest(15).index
    signed = corr.loc[top].sort_values()
    plt.figure(figsize=(8, 5))
    colors = np.where(signed >= 0, "#2a9d8f", "#e76f51")
    plt.barh(signed.index, signed.values, color=colors)
    plt.xlabel("Spearman correlation with ARMAS (P1 only)")
    plt.title("Strongest monotonic relationships in training data")
    plt.tight_layout()
    """),
    code(r"""
    # Standardized mean difference relative to P1; large magnitudes flag covariate shift.
    p1_mean = train_df[numeric_predictors].mean()
    p1_std = train_df[numeric_predictors].std().replace(0, np.nan)
    shift = pd.DataFrame({
        p: (compact[p][numeric_predictors].mean() - p1_mean) / p1_std
        for p in ("P2", "P3")
    })
    shift["max_abs_shift"] = shift.abs().max(axis=1)
    shift.sort_values("max_abs_shift", ascending=False).head(20).round(3)
    """),
    md("""
    ## 5. Inspect the one-hour trajectories

    Each line below is one 12-step history. Index 0 matches the static current-time dataset; the plot reverses the stored order so time reads naturally from 55 minutes ago to the current observation.
    """),
    code(r"""
    representative_features = [
        "NM_OULU", "SXR_long", "Particles_P10", "SW_Bz", "Index_Dst", "Solar_f107"
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    minutes = np.arange(-55, 5, 5)
    for ax, feature in zip(axes.flat, representative_features):
        arr = np.asarray([row[feature][::-1] for row in samples["P1"]], dtype=float)
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9], axis=0)
        ax.fill_between(minutes, q10, q90, alpha=0.25)
        ax.plot(minutes, q50, linewidth=2)
        ax.set(title=feature, xlabel="minutes relative to output", ylabel="value")
    fig.suptitle("P1 temporal profiles: median and 10–90% interval")
    plt.tight_layout()
    """),
    md("""
    ## 6. Duplicate/leakage audit

    The dataset documentation states that flights are separated by partition. Since there is no explicit flight ID, this checks exact observation duplicates using time, position, altitude, and target. It cannot prove flight separation; that guarantee must come from dataset construction.
    """),
    code(r"""
    duplicate_key = ["Datetime", "Latitude", "Longitude", "Altitude(Bar)", "ARMAS"]
    signatures = {
        p: set(pd.util.hash_pandas_object(df[duplicate_key], index=False).astype(str))
        for p, df in compact.items()
    }
    overlap = pd.DataFrame([
        {"pair": f"{a}-{b}", "exact_overlap": len(signatures[a] & signatures[b])}
        for a, b in (("P1", "P2"), ("P1", "P3"), ("P2", "P3"))
    ])
    overlap
    """),
    md("""
    ## 7. Exploration decision gate

    Before moving to modeling, record answers to these prompts:

    - Did all schema assertions pass?
    - Are any partition shifts large enough to mention in the report?
    - How does NAIRAS V3 performance change above 15 µSv/h?
    - Do the temporal profiles look physically plausible, and is index 0 confirmed as current time?
    - Are exact cross-partition duplicates zero?

    If these checks pass, continue to `02_stepwise_gpu_models_colab.ipynb`.
    """),
])


modeling = notebook([
    md(r"""
    # ARMAS stepwise deep-learning workflow (Colab GPU-ready)

    This notebook implements the project plan in auditable stages:

    1. current-time MLP (43 inputs),
    2. temporal 1D CNN (12 × 35 inputs),
    3. hybrid CNN–MLP (sequence plus 8 static inputs),
    4. NAIRAS V3 benchmark,
    5. cyclic P1/P2/P3 rotations and feature-group permutation.

    Start with one P1/P2/P3 run. Only after that run is correct should you enable all three rotations. Preprocessing is fitted on the training partition in every rotation. P3 remains untouched until the initial architecture choices are complete.
    """),
    md("""
    ## 0. Colab and GPU setup

    In Colab choose **Runtime → Change runtime type → T4 GPU**. Put the `ARMASdatasets_COMPLETE` folder in Google Drive. The dataset itself is modest in tensor memory; the GPU mainly accelerates the three neural networks and repeated rotations.
    """),
    code(r"""
    # Colab usually already provides these. Uncomment if an import fails.
    # %pip install -q numpy pandas scikit-learn matplotlib seaborn torch

    import copy
    import json
    import math
    import random
    import time
    from dataclasses import dataclass, asdict
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", DEVICE)
    if DEVICE.type == "cuda":
        print(torch.cuda.get_device_name(0))
    """),
    code(r"""
    # LOCAL
    DATA_DIR = Path("ARMASdatasets_COMPLETE-20261006T110557Z-1-001/ARMASdatasets_COMPLETE")
    OUTPUT_DIR = Path("outputs")

    # COLAB: comment the two lines above and uncomment these lines.
    # from google.colab import drive
    # drive.mount('/content/drive')
    # DATA_DIR = Path('/content/drive/MyDrive/ARMASdatasets_COMPLETE')
    # OUTPUT_DIR = Path('/content/drive/MyDrive/ARMAS_project_outputs')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FILES = {p: DATA_DIR / f"ARMAS_TIMESERIES_5M_20132023_{p}.json" for p in ("P1", "P2", "P3")}
    assert all(path.exists() for path in FILES.values()), "Fix DATA_DIR before continuing."
    """),
    md("""
    ## 1. Load tensors and enforce the predictor contract

    `NAIRASV2`, `NAIRASV3`, and `Vehicle_ID` are loaded only as metadata/benchmarks. They never enter a model tensor.
    """),
    code(r"""
    STATIC_FEATURES = [
        "Latitude", "Longitude", "Altitude(Bar)", "Altitude(GPS)",
        "Geomagnetic_latitude", "Geomagnetic_longitude",
        "Geomagnetic_Rc", "Geomagnetic_Lshell",
    ]
    TEMPORAL_FEATURES = [
        "NM_NEWK", "NM_OULU", "NM_THUL", "NM_SOPO",
        "SXR_short", "SXR_long",
        "Particles_P1", "Particles_P5", "Particles_P10", "Particles_P30",
        "Particles_P50", "Particles_P100", "Particles_E20",
        "SW_B", "SW_Bx", "SW_By", "SW_Bz", "SW_V", "SW_Vx", "SW_Vy", "SW_Vz",
        "SW_density", "SW_temperature", "SW_pressure",
        "Index_Kp", "Index_Dst", "Index_Ap",
        "Solar_sunspots", "Solar_f107", "Solar_NPF", "Solar_SPF", "Solar_APF",
        "Solar_NPF20", "Solar_SPF20", "Solar_APF20",
    ]
    FORBIDDEN_PREDICTORS = {"Vehicle_ID", "NAIRASV2", "NAIRASV3"}
    assert len(STATIC_FEATURES) == 8 and len(TEMPORAL_FEATURES) == 35

    @dataclass
    class Partition:
        name: str
        x_static: np.ndarray       # [N, 8]
        x_sequence: np.ndarray     # [N, 12, 35]
        y: np.ndarray              # [N]
        nairas_v3: np.ndarray      # benchmark only
        datetime: np.ndarray

        @property
        def x_current(self):
            # Proposal input: 8 current flight/geomagnetic + current values of 35 environmental variables.
            # load_partition reverses source order, so the current observation is the final step.
            return np.concatenate([self.x_static, self.x_sequence[:, -1, :]], axis=1)

    def load_partition(name, path):
        static, sequence, target, benchmark, datetimes = [], [], [], [], []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                source_seq = np.asarray([row[f] for f in TEMPORAL_FEATURES], dtype=np.float32).T
                if source_seq.shape != (12, 35):
                    raise ValueError(f"{name} line {line_number}: sequence shape {source_seq.shape}")
                # Source index 0 is current time. Reverse once so tensors run oldest -> current.
                seq = source_seq[::-1].copy()
                static.append([row[f] for f in STATIC_FEATURES])
                sequence.append(seq)
                target.append(row["ARMAS"])
                benchmark.append(row["NAIRASV3"])
                datetimes.append(row["Datetime"])
        part = Partition(
            name=name,
            x_static=np.asarray(static, dtype=np.float32),
            x_sequence=np.asarray(sequence, dtype=np.float32),
            y=np.asarray(target, dtype=np.float32),
            nairas_v3=np.asarray(benchmark, dtype=np.float32),
            datetime=np.asarray(datetimes),
        )
        arrays = [part.x_static, part.x_sequence, part.y, part.nairas_v3]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError(f"{name}: non-finite numeric values found")
        return part

    raw = {name: load_partition(name, path) for name, path in FILES.items()}
    for name, part in raw.items():
        print(name, part.x_static.shape, part.x_sequence.shape, part.y.shape, part.x_current.shape)
    """),
    md("""
    Expected first-run roles: P1 training, P2 validation, P3 testing. Do not inspect P3 model performance while choosing architecture, learning rate, batch size, or epochs.
    """),
    md("""
    ## 2. Train-only standardization

    The sequence scaler learns one mean and standard deviation per environmental feature from all training rows and time steps. The same fitted parameters transform validation and test data. The target is standardized for stable optimization and converted back to µSv/h for every metric.
    """),
    code(r"""
    @dataclass
    class ScaledPartition:
        name: str
        x_static: np.ndarray
        x_sequence: np.ndarray
        x_current: np.ndarray
        y_scaled: np.ndarray
        y_raw: np.ndarray
        nairas_v3: np.ndarray

    @dataclass
    class Preprocessors:
        static_scaler: StandardScaler
        sequence_scaler: StandardScaler
        current_scaler: StandardScaler
        target_scaler: StandardScaler

    def fit_preprocessors(train):
        return Preprocessors(
            static_scaler=StandardScaler().fit(train.x_static),
            sequence_scaler=StandardScaler().fit(train.x_sequence.reshape(-1, 35)),
            current_scaler=StandardScaler().fit(train.x_current),
            target_scaler=StandardScaler().fit(train.y.reshape(-1, 1)),
        )

    def transform_partition(part, prep):
        n = len(part.y)
        seq = prep.sequence_scaler.transform(part.x_sequence.reshape(-1, 35)).reshape(n, 12, 35)
        return ScaledPartition(
            name=part.name,
            x_static=prep.static_scaler.transform(part.x_static).astype(np.float32),
            x_sequence=seq.astype(np.float32),
            x_current=prep.current_scaler.transform(part.x_current).astype(np.float32),
            y_scaled=prep.target_scaler.transform(part.y.reshape(-1, 1)).ravel().astype(np.float32),
            y_raw=part.y,
            nairas_v3=part.nairas_v3,
        )
    """),
    md("""
    ## 3. Models

    Prediction before running: the MLP should already be strong because altitude and geomagnetic location are physically informative. The CNN only wins if the preceding hour adds signal beyond the final environmental sample. The hybrid is the main candidate because it combines both sources.
    """),
    code(r"""
    class CurrentMLP(nn.Module):
        def __init__(self, input_dim=43):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.15),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(64, 1),
            )
        def forward(self, x_current, x_sequence=None, x_static=None):
            return self.net(x_current).squeeze(1)

    class TemporalCNN(nn.Module):
        def __init__(self, n_features=35):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.10), nn.Linear(32, 1))
        def forward(self, x_current, x_sequence, x_static=None):
            x = x_sequence.transpose(1, 2)  # [batch, features, time]
            return self.head(self.encoder(x)).squeeze(1)

    class HybridCNNMLP(nn.Module):
        def __init__(self, n_temporal=35, n_static=8):
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Conv1d(n_temporal, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            self.static = nn.Sequential(nn.Linear(n_static, 32), nn.ReLU(), nn.Dropout(0.10))
            self.head = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 1))
        def forward(self, x_current, x_sequence, x_static):
            temporal_embedding = self.temporal(x_sequence.transpose(1, 2))
            static_embedding = self.static(x_static)
            return self.head(torch.cat([temporal_embedding, static_embedding], dim=1)).squeeze(1)
    """),
    md("""
    ## 4. Training, early stopping, and metrics

    Architecture decisions use validation RMSE. Test metrics are computed after the best validation checkpoint is restored. `QUICK_RUN=True` is a smoke test; it is not report-quality training.
    """),
    code(r"""
    @dataclass
    class TrainConfig:
        epochs: int = 60
        batch_size: int = 512
        learning_rate: float = 1e-3
        weight_decay: float = 1e-4
        patience: int = 10
        high_dose_weight: float = 1.0  # 1.0 = ordinary MSE; increase only as a documented experiment

    QUICK_RUN = True
    CONFIG = TrainConfig(epochs=4, patience=2) if QUICK_RUN else TrainConfig()
    print(CONFIG)

    def make_loader(part, batch_size, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(part.x_current),
            torch.from_numpy(part.x_sequence),
            torch.from_numpy(part.x_static),
            torch.from_numpy(part.y_scaled),
            torch.from_numpy(part.y_raw),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                          pin_memory=(DEVICE.type == "cuda"))

    def predict(model, part, batch_size=2048):
        model.eval()
        preds = []
        loader = make_loader(part, batch_size, shuffle=False)
        with torch.no_grad():
            for x_current, x_sequence, x_static, _, _ in loader:
                out = model(x_current.to(DEVICE), x_sequence.to(DEVICE), x_static.to(DEVICE))
                preds.append(out.cpu().numpy())
        return np.concatenate(preds)

    def metrics(y, pred):
        return {
            "n": len(y),
            "MAE": mean_absolute_error(y, pred),
            "RMSE": mean_squared_error(y, pred) ** 0.5,
            "R2": r2_score(y, pred),
        }

    def evaluate_regimes(y, pred):
        rows = []
        for regime, mask in {"all": np.ones(len(y), bool), "ARMAS > 15": y > 15}.items():
            rows.append({"regime": regime, **metrics(y[mask], pred[mask])})
        return rows

    def train_model(model, train, val, target_scaler, config=CONFIG):
        model = model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        train_loader = make_loader(train, config.batch_size, shuffle=True)
        best_state, best_rmse, stale = None, float("inf"), 0
        history = []

        for epoch in range(1, config.epochs + 1):
            model.train()
            losses = []
            for x_current, x_sequence, x_static, y_scaled, y_raw in train_loader:
                x_current, x_sequence, x_static = x_current.to(DEVICE), x_sequence.to(DEVICE), x_static.to(DEVICE)
                y_scaled, y_raw = y_scaled.to(DEVICE), y_raw.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                pred = model(x_current, x_sequence, x_static)
                weights = torch.where(y_raw > 15, config.high_dose_weight, 1.0)
                loss = (weights * (pred - y_scaled).square()).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            val_scaled = predict(model, val)
            val_pred = target_scaler.inverse_transform(val_scaled.reshape(-1, 1)).ravel()
            val_rmse = metrics(val.y_raw, val_pred)["RMSE"]
            history.append({"epoch": epoch, "train_loss": np.mean(losses), "val_RMSE": val_rmse})
            print(f"epoch {epoch:03d} | loss {np.mean(losses):.4f} | val RMSE {val_rmse:.4f}")

            if val_rmse < best_rmse - 1e-4:
                best_rmse, stale = val_rmse, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= config.patience:
                    print("early stopping")
                    break

        model.load_state_dict(best_state)
        return model, pd.DataFrame(history)
    """),
    md("""
    ## 5. First controlled run: P1 train, P2 validation, P3 final test

    Run this section model by model. During development, compare P1/P2 curves and metrics. Treat P3 results as final only after decisions are frozen.
    """),
    code(r"""
    prep = fit_preprocessors(raw["P1"])
    data = {name: transform_partition(part, prep) for name, part in raw.items()}

    MODEL_FACTORIES = {
        "MLP": CurrentMLP,
        "CNN": TemporalCNN,
        "Hybrid": HybridCNNMLP,
    }
    trained, histories, first_run_rows, predictions = {}, {}, [], {}

    for model_name, factory in MODEL_FACTORIES.items():
        print(f"\n===== {model_name} =====")
        seed_everything(42)
        model, history = train_model(factory(), data["P1"], data["P2"], prep.target_scaler)
        trained[model_name], histories[model_name] = model, history
        for partition in ("P2", "P3"):
            pred_scaled = predict(model, data[partition])
            pred = prep.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            predictions[(model_name, partition)] = pred
            for row in evaluate_regimes(data[partition].y_raw, pred):
                first_run_rows.append({"model": model_name, "partition": partition, **row})

    # Physics benchmark on the exact same rows.
    for partition in ("P2", "P3"):
        for row in evaluate_regimes(data[partition].y_raw, data[partition].nairas_v3):
            first_run_rows.append({"model": "NAIRAS V3", "partition": partition, **row})

    first_run_results = pd.DataFrame(first_run_rows)
    first_run_results.round(4)
    """),
    code(r"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, model_name in zip(axes, MODEL_FACTORIES):
        y = data["P3"].y_raw
        pred = predictions[(model_name, "P3")]
        hb = ax.hexbin(y, pred, gridsize=45, mincnt=1, cmap="viridis")
        limit = max(y.max(), pred.max())
        ax.plot([0, limit], [0, limit], "r--", linewidth=1)
        ax.set(title=f"{model_name}: P3", xlabel="measured ARMAS", ylabel="predicted ARMAS")
    plt.tight_layout()

    best_name = (first_run_results.query("partition == 'P2' and regime == 'all'")
                 .sort_values("RMSE").iloc[0]["model"])
    print("Best validation entry (benchmark included):", best_name)
    """),
    md("""
    Reflection:

    - Does CNN validation RMSE beat MLP? If not, the one-hour history has not yet demonstrated added value.
    - Does Hybrid beat both single-branch models?
    - Which model degrades least above 15 µSv/h?
    - Compare against NAIRAS V3, but never tune a neural model using test-set or benchmark outcomes.
    """),
    md("""
    ## 6. Feature-group permutation on the hybrid model

    Permutation breaks a group's relationship with the target while preserving each sample's 12-step trajectory. The increase in RMSE is the importance score. This is diagnostic, not causal.
    """),
    code(r"""
    TEMPORAL_GROUPS = {
        "neutron_monitors": [i for i, f in enumerate(TEMPORAL_FEATURES) if f.startswith("NM_")],
        "solar_xray": [i for i, f in enumerate(TEMPORAL_FEATURES) if f.startswith("SXR_")],
        "particles": [i for i, f in enumerate(TEMPORAL_FEATURES) if f.startswith("Particles_")],
        "solar_wind": [i for i, f in enumerate(TEMPORAL_FEATURES) if f.startswith("SW_")],
        "geomagnetic_indices": [i for i, f in enumerate(TEMPORAL_FEATURES) if f.startswith("Index_")],
        "solar_activity": [i for i, f in enumerate(TEMPORAL_FEATURES) if f.startswith("Solar_")],
    }

    def permute_group(part, group_indices, seed=42):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(part.y_raw))
        changed = copy.copy(part)
        changed.x_sequence = part.x_sequence.copy()
        changed.x_sequence[:, :, group_indices] = part.x_sequence[order][:, :, group_indices]
        return changed

    if "Hybrid" in trained:
        model = trained["Hybrid"]
        base_scaled = predict(model, data["P3"])
        base_pred = prep.target_scaler.inverse_transform(base_scaled.reshape(-1, 1)).ravel()
        base_rmse = metrics(data["P3"].y_raw, base_pred)["RMSE"]
        importance = []
        for group, indices in TEMPORAL_GROUPS.items():
            changed = permute_group(data["P3"], indices)
            pred_scaled = predict(model, changed)
            pred = prep.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            importance.append({"group": group, "delta_RMSE": metrics(changed.y_raw, pred)["RMSE"] - base_rmse})
        importance = pd.DataFrame(importance).sort_values("delta_RMSE", ascending=False)
        display(importance)
        sns.barplot(data=importance, x="delta_RMSE", y="group", color="#457b9d")
        plt.axvline(0, color="black", linewidth=1)
        plt.title("Hybrid feature-group permutation importance on P3")
        plt.tight_layout()
    """),
    md("""
    ## 7. Final experiment: three cyclic rotations

    Set `QUICK_RUN=False`, restart the runtime, rerun through the definitions, then run this section. Each rotation fits fresh scalers and fresh models. This may take time even on a GPU.
    """),
    code(r"""
    ROTATIONS = [
        ("P1", "P2", "P3"),
        ("P2", "P3", "P1"),
        ("P3", "P1", "P2"),
    ]

    def run_rotation(train_name, val_name, test_name, config):
        prep_r = fit_preprocessors(raw[train_name])
        scaled = {name: transform_partition(part, prep_r) for name, part in raw.items()}
        rows = []
        for model_name, factory in MODEL_FACTORIES.items():
            seed_everything(42)
            model, _ = train_model(factory(), scaled[train_name], scaled[val_name], prep_r.target_scaler, config)
            pred_scaled = predict(model, scaled[test_name])
            pred = prep_r.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            for result in evaluate_regimes(scaled[test_name].y_raw, pred):
                rows.append({"train": train_name, "validation": val_name, "test": test_name,
                             "model": model_name, **result})
        for result in evaluate_regimes(scaled[test_name].y_raw, scaled[test_name].nairas_v3):
            rows.append({"train": train_name, "validation": val_name, "test": test_name,
                         "model": "NAIRAS V3", **result})
        return rows

    # Safety gate: prevent an accidental report-quality claim from a 4-epoch smoke test.
    RUN_ALL_ROTATIONS = False
    if RUN_ALL_ROTATIONS:
        if QUICK_RUN:
            raise ValueError("Set QUICK_RUN=False and rerun before final rotations.")
        rotation_rows = []
        for roles in ROTATIONS:
            print("\nROTATION", roles)
            rotation_rows.extend(run_rotation(*roles, config=CONFIG))
        rotation_results = pd.DataFrame(rotation_rows)
        rotation_results.to_csv(OUTPUT_DIR / "rotation_results.csv", index=False)
        display(rotation_results.round(4))
        summary = (rotation_results.groupby(["model", "regime"])[["MAE", "RMSE", "R2"]]
                   .agg(["mean", "std"]).round(4))
        display(summary)
    else:
        print("Set RUN_ALL_ROTATIONS=True only for the final GPU run.")
    """),
    md("""
    ## 8. Save first-run artifacts

    The CSVs are report-ready inputs. The checkpoint includes model weights, feature names, preprocessing parameters, and configuration. Saving to Drive prevents loss when the Colab runtime disconnects.
    """),
    code(r"""
    first_run_results.to_csv(OUTPUT_DIR / "first_run_metrics.csv", index=False)
    for model_name, history in histories.items():
        history.to_csv(OUTPUT_DIR / f"history_{model_name.lower()}.csv", index=False)

    for model_name, model in trained.items():
        torch.save({
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "static_features": STATIC_FEATURES,
            "temporal_features": TEMPORAL_FEATURES,
            "train_partition": "P1",
            "validation_partition": "P2",
            "test_partition": "P3",
            "config": asdict(CONFIG),
            "scalers": prep,
        }, OUTPUT_DIR / f"{model_name.lower()}_p1_p2_p3.pt")
    print("Saved to", OUTPUT_DIR.resolve())
    """),
    md("""
    ## Final reporting checklist

    - Report MAE, RMSE, and R² as mean ± standard deviation across the three rotations.
    - Provide separate full-data and `ARMAS > 15 µSv/h` rows.
    - Include NAIRAS V3 on the identical test samples.
    - State that all preprocessing was fit on the training partition only.
    - Include predicted-versus-observed and residual plots.
    - Treat feature-group permutation as association, not causation.
    - Record Colab GPU type, PyTorch version, seed, epochs, early stopping, and runtime.
    """),
])


for filename, content in {
    "01_armas_5m_exploration.ipynb": exploration,
    "02_stepwise_gpu_models_colab.ipynb": modeling,
}.items():
    path = OUT / filename
    path.write_text(json.dumps(content, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")
