import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks" / "final"
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


cells = [
    md(r"""
    # ARMAS physics-guided temporal modeling — complete research pipeline

    This notebook implements the current project proposal, **Physics-Guided Temporal Transformer for Aviation Effective Dose-Rate Estimation**.

    Required comparison:

    | Entry | Prediction target | Inputs |
    |---|---|---|
    | NAIRAS V3 | none — external physics benchmark | supplied NAIRAS V3 estimate |
    | Current-time MLP | ARMAS directly | 8 static + current values of 35 environmental variables |
    | Hybrid CNN–MLP | ARMAS directly | 12×35 sequence + 8 static variables |
    | Direct Temporal Transformer | ARMAS directly | 12×35 sequence + 43 current-time variables |
    | Physics-guided Transformer | residual `ARMAS − NAIRAS V3` | sequence + static variables + NAIRAS V3 |

    The physics-guided prediction is `predicted ARMAS = NAIRAS V3 + predicted residual`.

    P1 trains, P2 validates, and P3 stays locked during initial development. Final results preserve the proposal's three primary cyclic rotations and add all six ordered train/validation/test assignments. `Vehicle_ID` and `NAIRASV2` are never predictors. NAIRAS V3 enters only the explicitly physics-guided residual model.
    """),
    md("""
    ## 0. Colab setup

    Select **Runtime → Change runtime type → T4 GPU**. Begin with `RUN_MODE="smoke"` only to verify the pipeline, then use `"development"` or `"full"` for interpretable learning curves and report-quality experiments. A three-epoch smoke test is never treated as a final result.
    """),
    code(r"""
    # Colab normally includes these packages.
    # %pip install -q numpy pandas scikit-learn matplotlib seaborn torch

    import copy
    import json
    import random
    import time
    from dataclasses import dataclass, asdict, replace
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    sns.set_theme(style="whitegrid", context="notebook")
    pd.set_option("display.max_columns", 100)

    # USER ACTION: Leave this True if you want every important figure saved in OUTPUT_DIR.
    SAVE_FIGURES = True

    def finish_figure(filename, fig=None):
        # Apply consistent layout, optionally save, and always display the current figure.
        fig = fig or plt.gcf()
        fig.tight_layout()
        if SAVE_FIGURES:
            fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
        plt.show()

    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    """),
    code(r"""
    # LOCAL PATHS
    DATA_DIR = Path("ARMASdatasets_COMPLETE-20261006T110557Z-1-001/ARMASdatasets_COMPLETE")
    OUTPUT_DIR = Path("outputs/physics_guided_residual")

    # COLAB PATHS: comment the two lines above and uncomment these four lines.
    # from google.colab import drive
    # drive.mount('/content/drive')
    # DATA_DIR = Path('/content/drive/MyDrive/ARMASdatasets_COMPLETE')
    # OUTPUT_DIR = Path('/content/drive/MyDrive/ARMAS_project_outputs/physics_guided_residual')

    STUDY_OUTPUT_SUBFOLDER = "physics_guided_residual"
    if OUTPUT_DIR.name != STUDY_OUTPUT_SUBFOLDER:
        raise ValueError(
            f"OUTPUT_DIR must end with '{STUDY_OUTPUT_SUBFOLDER}' so this study cannot overwrite "
            "the direct-ARMAS study."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Physics-guided residual-study outputs:", OUTPUT_DIR.resolve())
    FILES = {p: DATA_DIR / f"ARMAS_TIMESERIES_5M_20132023_{p}.json" for p in ("P1", "P2", "P3")}
    assert all(path.exists() for path in FILES.values()), "Fix DATA_DIR before continuing."
    """),
    md("""
    ## 1. Configuration panel

    Change settings here instead of searching through the notebook. The run modes make the difference between a code check and a real experiment explicit. Standard training uses MSE with no high-dose weighting. The imbalance experiment uses weighted Huber loss; choose the weight with validation data only.
    """),
    code(r"""
    @dataclass(frozen=True)
    class ExperimentConfig:
        name: str
        model_kind: str                 # mlp, cnn_hybrid, transformer_direct, transformer_residual
        target_mode: str = "direct"     # direct or residual
        sequence_steps: int = 12
        loss: str = "mse"               # mse or huber
        high_dose_weight: float = 1.0
        hidden_dim: int = 128
        cnn_channels: int = 64
        d_model: int = 64
        n_heads: int = 4
        n_layers: int = 2
        feedforward_dim: int = 128
        dropout: float = 0.10
        pooling: str = "current"        # current or mean
        learning_rate: float = 3e-4
        weight_decay: float = 1e-4
        batch_size: int = 512
        epochs: int = 50
        patience: int = 8
        scheduler_patience: int = 3
        lr_factor: float = 0.5
        min_lr: float = 1e-6
        min_delta: float = 1e-4
        seed: int = 42

    # ============================= USER ACTION =============================
    # 1) First run: choose "smoke" only to verify that every cell works.
    # 2) Model development: choose "development" for meaningful P1/P2 results.
    # 3) Final report: choose "full" only after configurations are frozen.
    # =======================================================================
    RUN_MODE = "development"  # choose from: smoke, development, full
    RUN_PROFILES = {
        "smoke": {"epochs": 3, "patience": 2},
        "development": {"epochs": 25, "patience": 6},
        "full": {"epochs": 80, "patience": 12},
    }
    if RUN_MODE not in RUN_PROFILES:
        raise ValueError(f"Unknown RUN_MODE={RUN_MODE!r}; choose {list(RUN_PROFILES)}")
    QUICK_RUN = RUN_MODE == "smoke"  # retained for the final-rotation safety check
    EPOCHS = RUN_PROFILES[RUN_MODE]["epochs"]
    PATIENCE = RUN_PROFILES[RUN_MODE]["patience"]
    print(f"Run mode: {RUN_MODE} | maximum epochs: {EPOCHS} | early-stopping patience: {PATIENCE}")

    STANDARD_CONFIGS = [
        ExperimentConfig("MLP", "mlp", epochs=EPOCHS, patience=PATIENCE),
        ExperimentConfig("Hybrid CNN-MLP", "cnn_hybrid", epochs=EPOCHS, patience=PATIENCE),
        ExperimentConfig("Direct Transformer", "transformer_direct", epochs=EPOCHS, patience=PATIENCE),
        ExperimentConfig("Physics-guided Transformer", "transformer_residual", target_mode="residual",
                         epochs=EPOCHS, patience=PATIENCE),
    ]

    # Validation-only imbalance experiments for the proposed residual model.
    WEIGHTED_HUBER_CONFIGS = [
        replace(STANDARD_CONFIGS[-1], name=f"Physics-guided Huber w={weight:g}",
                loss="huber", high_dose_weight=weight)
        for weight in (1.0, 2.0, 4.0)
    ]
    pd.DataFrame([asdict(c) for c in STANDARD_CONFIGS + WEIGHTED_HUBER_CONFIGS])
    """),
    md("""
    ### Research questions and decision rules

    1. **Does temporal context help?** Compare the current-time MLP, CNN, Transformer, and the 1/6/12-step ablation.
    2. **Does physics guidance help?** Compare direct ARMAS prediction with residual prediction and uncorrected NAIRAS V3.
    3. **Can the rare high-dose regime improve without sacrificing common cases?** Compare MSE with weighted Huber using P2 only.
    4. **Is extra network complexity justified?** Relate RMSE to parameters, approximate FLOPs, memory, and inference time.
    5. **Are conclusions stable?** Check multiple random seeds, three flight-separated rotations, and bootstrap confidence intervals.

    The primary selection metric is P2 overall RMSE. High-dose RMSE is a prespecified secondary metric. P3 is opened once after all design choices are frozen. This avoids optimistic test-set tuning.
    """),
    md("""
    ### Read this before running: exactly what you need to change

    | Stage | What you change | What you run | What you are allowed to conclude |
    |---|---|---|---|
    | 1. Code check | `RUN_MODE = "smoke"` | Through P2 validation | Only that the pipeline works; **do not report these metrics** |
    | 2. Model development | `RUN_MODE = "development"` | Baselines, tuning, and sections 9–12 | Select configurations using P2; plots are labeled validation |
    | 3. Freeze decision | Review the automatically selected weighted/tuned models | Record the names and P2 evidence | Do not change architecture or loss after the next stage |
    | 4. Locked test | Set `EVALUATE_P3 = True` once | Rerun sections 8–12 | Report the single P3 confirmation |
    | 5. Final robustness | Set `RUN_MODE = "full"` and keep `EVALUATE_P3 = True` | Run section 13 | Report all six assignments and the primary three-cycle summary |

    **If a cell says it did nothing:** read its printed message. Optional seed analysis is off unless you explicitly enable it. The six-assignment study is intentionally delayed until the full, unlocked stage because it uses every partition in every role.
    """),
    md("""
    ## 2. Load and validate the data

    The source sequences store current time at index 0. This was checked against all matching static rows. The loader reverses the sequence once so tensors are ordered from 55 minutes before observation to current time.
    """),
    code(r"""
    STATIC_FEATURES = [
        "Latitude", "Longitude", "Altitude(Bar)", "Altitude(GPS)",
        "Geomagnetic_latitude", "Geomagnetic_longitude",
        "Geomagnetic_Rc", "Geomagnetic_Lshell",
    ]
    TEMPORAL_FEATURES = [
        "NM_NEWK", "NM_OULU", "NM_THUL", "NM_SOPO", "SXR_short", "SXR_long",
        "Particles_P1", "Particles_P5", "Particles_P10", "Particles_P30",
        "Particles_P50", "Particles_P100", "Particles_E20",
        "SW_B", "SW_Bx", "SW_By", "SW_Bz", "SW_V", "SW_Vx", "SW_Vy", "SW_Vz",
        "SW_density", "SW_temperature", "SW_pressure",
        "Index_Kp", "Index_Dst", "Index_Ap",
        "Solar_sunspots", "Solar_f107", "Solar_NPF", "Solar_SPF", "Solar_APF",
        "Solar_NPF20", "Solar_SPF20", "Solar_APF20",
    ]
    assert len(STATIC_FEATURES) == 8 and len(TEMPORAL_FEATURES) == 35

    @dataclass
    class RawPartition:
        name: str
        x_static: np.ndarray
        x_sequence: np.ndarray
        y: np.ndarray
        nairas_v3: np.ndarray

        @property
        def residual(self):
            return self.y - self.nairas_v3

        @property
        def x_current(self):
            return np.concatenate([self.x_static, self.x_sequence[:, -1, :]], axis=1)

    def load_partition(name, path):
        static, sequence, y, n3 = [], [], [], []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                source = np.asarray([
                    [np.nan if value is None else value for value in row[f]]
                    for f in TEMPORAL_FEATURES
                ], dtype=np.float32).T
                if source.shape != (12, 35):
                    raise ValueError(f"{name} line {line_number}: expected (12, 35), got {source.shape}")
                static.append([np.nan if row[f] is None else row[f] for f in STATIC_FEATURES])
                sequence.append(source[::-1].copy())
                y.append(row["ARMAS"])
                n3.append(row["NAIRASV3"])
        part = RawPartition(name, np.asarray(static, np.float32), np.asarray(sequence, np.float32),
                            np.asarray(y, np.float32), np.asarray(n3, np.float32))
        if np.isinf(part.x_static).any() or np.isinf(part.x_sequence).any():
            raise ValueError(f"{name}: infinite inputs found")
        if not np.isfinite(part.y).all() or not np.isfinite(part.nairas_v3).all():
            raise ValueError(f"{name}: ARMAS and NAIRAS V3 must be complete for targets/benchmarking")
        return part

    raw = {name: load_partition(name, path) for name, path in FILES.items()}
    overview = pd.DataFrame([
        {"partition": name, "rows": len(p.y), "ARMAS_mean": p.y.mean(),
         "residual_mean": p.residual.mean(), "A <= 15": int((p.y <= 15).sum()),
         "A > 15": int((p.y > 15).sum())}
        for name, p in raw.items()
    ])
    expected_rows = {"P1": 32369, "P2": 30752, "P3": 29294}
    assert {name: len(part.y) for name, part in raw.items()} == expected_rows
    assert sum(expected_rows.values()) == 92415
    display(overview.round(3))
    """),
    md("""
    ### 2.1 Data audit, imbalance, missingness, and partition shift

    Before modeling, quantify what makes the problem difficult. The tables below show target quantiles, the rarity of high-dose observations, missing-input rates, and simple standardized mean differences between partitions. Large partition differences explain why flight-separated evaluation is more demanding than a random row split.
    """),
    code(r"""
    audit_rows = []
    for name, part in raw.items():
        for variable, values in {"ARMAS": part.y, "NAIRAS V3": part.nairas_v3,
                                 "ARMAS - NAIRAS V3": part.residual}.items():
            q = np.quantile(values, [0, .01, .25, .5, .75, .95, .99, 1])
            audit_rows.append({"partition": name, "variable": variable, "mean": values.mean(),
                               "std": values.std(), "min": q[0], "p01": q[1], "p25": q[2],
                               "median": q[3], "p75": q[4], "p95": q[5], "p99": q[6], "max": q[7]})
    distribution_audit = pd.DataFrame(audit_rows)
    display(distribution_audit.round(3))

    missing_rows = []
    for name, part in raw.items():
        for j, feature in enumerate(STATIC_FEATURES):
            missing_rows.append({"partition": name, "feature": feature, "feature_type": "static",
                                 "missing_percent": 100 * np.isnan(part.x_static[:, j]).mean()})
        for j, feature in enumerate(TEMPORAL_FEATURES):
            missing_rows.append({"partition": name, "feature": feature, "feature_type": "temporal",
                                 "missing_percent": 100 * np.isnan(part.x_sequence[:, :, j]).mean()})
    missingness = pd.DataFrame(missing_rows)
    display(missingness.sort_values("missing_percent", ascending=False).head(20).round(3))

    shift_rows = []
    current_names = STATIC_FEATURES + TEMPORAL_FEATURES
    reference = raw["P1"].x_current
    for other_name in ("P2", "P3"):
        other = raw[other_name].x_current
        pooled_sd = np.sqrt((np.nanvar(reference, axis=0) + np.nanvar(other, axis=0)) / 2)
        smd = (np.nanmean(other, axis=0) - np.nanmean(reference, axis=0)) / np.where(pooled_sd, pooled_sd, np.nan)
        shift_rows += [{"comparison": f"{other_name} vs P1", "feature": feature,
                        "standardized_mean_difference": value}
                       for feature, value in zip(current_names, smd)]
    partition_shift = pd.DataFrame(shift_rows)
    display(partition_shift.assign(abs_SMD=lambda x: x.standardized_mean_difference.abs())
                           .sort_values("abs_SMD", ascending=False).head(20).round(3))

    print("High-dose prevalence (%):")
    display(overview.assign(high_dose_percent=lambda x: 100 * x["A > 15"] / x.rows)
                    [["partition", "rows", "high_dose_percent"]].round(3))

    full_y = np.concatenate([part.y for part in raw.values()])
    full_residual = np.concatenate([part.residual for part in raw.values()])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.kdeplot(full_y, label="Full dataset", ax=axes[0], color="#2E5EAA")
    sns.kdeplot(full_residual, label="Full dataset", ax=axes[1], color="#2E5EAA")
    axes[0].set(title="ARMAS distribution - full dataset", xlabel="ARMAS (µSv/h)")
    axes[1].set(title="Physics residual - full dataset", xlabel="ARMAS - NAIRAS V3 (µSv/h)")
    for ax in axes: ax.legend()
    finish_figure("data_partition_distributions.png", fig)
    """),
    md("""
    ### 2.2 Visual data-quality dashboard

    **What you need to do:** inspect these figures before training. If one partition has a visibly different target distribution, a large standardized mean difference (`|SMD| > 0.5`), or substantially different missingness, mention partition shift in the report. Do not delete difficult observations merely to improve the score.
    """),
    code(r"""
    # VISUAL 1: distribution, imbalance, missingness, and partition shift.
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    sns.ecdfplot(full_y, label="Full dataset", color="#2E5EAA", ax=axes[0, 0])
    axes[0, 0].axvline(15, color="red", linestyle="--", label="15 µSv/h threshold")
    axes[0, 0].set(title="ARMAS empirical cumulative distribution - full dataset",
                   xlabel="ARMAS (µSv/h)", ylabel="Cumulative probability")
    axes[0, 0].legend()

    prevalence = overview.assign(high_dose_percent=lambda x: 100 * x["A > 15"] / x.rows)
    sns.barplot(data=prevalence, x="partition", y="high_dose_percent", color="#457b9d", ax=axes[0, 1])
    axes[0, 1].set(title="High-dose prevalence by partition", ylabel="ARMAS > 15 (%)")
    for container in axes[0, 1].containers:
        axes[0, 1].bar_label(container, fmt="%.2f%%")

    top_missing = (missingness.groupby("feature").missing_percent.max()
                   .nlargest(15).index)
    missing_matrix = (missingness[missingness.feature.isin(top_missing)]
                      .pivot(index="feature", columns="partition", values="missing_percent"))
    sns.heatmap(missing_matrix, annot=True, fmt=".1f", cmap="mako_r",
                cbar_kws={"label": "Missing (%)"}, ax=axes[1, 0])
    axes[1, 0].set_title("Top missing-input rates")

    top_shift = (partition_shift.assign(abs_SMD=lambda x: x.standardized_mean_difference.abs())
                 .sort_values("abs_SMD", ascending=False).head(20))
    sns.barplot(data=top_shift, x="standardized_mean_difference", y="feature",
                hue="comparison", orient="h", ax=axes[1, 1])
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].axvline(.5, color="red", linestyle=":")
    axes[1, 1].axvline(-.5, color="red", linestyle=":")
    axes[1, 1].set(title="Largest partition shifts", xlabel="Standardized mean difference")
    finish_figure("data_quality_dashboard.png", fig)
    """),
    code(r"""
    # VISUAL 2: how the raw NAIRAS physics estimate agrees with ARMAS in each partition.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, (name, part) in zip(axes, raw.items()):
        ax.hexbin(part.y, part.nairas_v3, gridsize=45, mincnt=1, cmap="viridis")
        limit = max(part.y.max(), part.nairas_v3.max())
        ax.plot([0, limit], [0, limit], "r--", label="perfect agreement")
        ax.axvline(15, color="white", linestyle=":")
        ax.set(title=name, xlabel="Measured ARMAS (µSv/h)")
    axes[0].set_ylabel("NAIRAS V3 (µSv/h)")
    finish_figure("nairas_vs_armas_by_partition.png", fig)

    # VISUAL 3: where the physics residual changes with measured dose.
    sample_rows = []
    rng = np.random.default_rng(42)
    for name, part in raw.items():
        index = rng.choice(len(part.y), min(10000, len(part.y)), replace=False)
        sample_rows.append(pd.DataFrame({"partition": name, "ARMAS": part.y[index],
                                         "physics_residual": part.residual[index],
                                         "regime": np.where(part.y[index] > 15, "A > 15", "A <= 15")}))
    residual_sample = pd.concat(sample_rows, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(data=residual_sample, x="ARMAS", y="physics_residual", hue="partition",
                    alpha=.22, s=12, ax=axes[0])
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].axvline(15, color="red", linestyle=":")
    axes[0].set_title("ARMAS–NAIRAS discrepancy versus measured dose")
    sns.boxenplot(data=residual_sample, x="partition", y="physics_residual", hue="regime", ax=axes[1])
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_title("Physics residual by partition and regime")
    finish_figure("physics_residual_diagnostics.png", fig)
    """),
    md("""
    ## 3. Feature-direction exploration

    The proposal asks which variables and preceding time steps are associated with large ARMAS–NAIRAS discrepancies. The heatmap below computes Spearman correlation between each feature at each lag and the residual. It is descriptive association, not causation.
    """),
    code(r"""
    def lag_correlation_matrix(part, max_rows=None):
        n = len(part.y) if max_rows is None else min(max_rows, len(part.y))
        index = np.random.default_rng(42).choice(len(part.y), n, replace=False)
        residual_rank = pd.Series(part.residual[index]).rank().to_numpy()
        matrix = np.empty((35, 12), dtype=float)
        for feature in range(35):
            for lag in range(12):
                feature_rank = pd.Series(part.x_sequence[index, lag, feature]).rank()
                matrix[feature, lag] = feature_rank.corr(pd.Series(residual_rank), method="pearson")
        return pd.DataFrame(matrix, index=TEMPORAL_FEATURES,
                            columns=[f"-{55 - 5*i}m" if i < 11 else "current" for i in range(12)])

    lag_corr = lag_correlation_matrix(raw["P1"])
    plt.figure(figsize=(13, 12))
    sns.heatmap(lag_corr, center=0, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    plt.title("P1 feature-by-lag Spearman correlation with ARMAS − NAIRAS V3")
    plt.xlabel("Time relative to ARMAS observation")
    finish_figure("p1_feature_lag_correlation.png")
    """),
    code(r"""
    # Static-feature direction, overall and by radiation regime.
    static_rows = []
    for feature_index, feature in enumerate(STATIC_FEATURES):
        for regime, mask in {"all": np.ones(len(raw["P1"].y), bool),
                             "A <= 15": raw["P1"].y <= 15,
                             "A > 15": raw["P1"].y > 15}.items():
            rho = pd.Series(raw["P1"].x_static[mask, feature_index]).corr(
                pd.Series(raw["P1"].residual[mask]), method="spearman")
            static_rows.append({"feature": feature, "regime": regime, "Spearman": rho})
    static_direction = pd.DataFrame(static_rows)
    display(static_direction.pivot(index="feature", columns="regime", values="Spearman").round(3))
    """),
    code(r"""
    # USER ACTION: use these plots to choose variables/lags to discuss, not to remove inputs.
    max_lag_signal = (lag_corr.abs().max(axis=1).sort_values(ascending=False)
                      .rename("maximum_absolute_Spearman").reset_index()
                      .rename(columns={"index": "feature"}))
    top_temporal_features = max_lag_signal.head(10).feature.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    sns.barplot(data=max_lag_signal.head(15), x="maximum_absolute_Spearman", y="feature",
                color="#457b9d", ax=axes[0])
    axes[0].set_title("Strongest temporal associations at any lag")
    lag_profile = (lag_corr.loc[top_temporal_features].reset_index()
                   .rename(columns={"index": "feature"})
                   .melt(id_vars="feature", var_name="lag", value_name="Spearman"))
    sns.lineplot(data=lag_profile, x="lag", y="Spearman", hue="feature", marker="o", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Lag profiles of the ten strongest temporal features")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    finish_figure("temporal_correlation_rank_and_lag_profiles.png", fig)

    static_matrix = static_direction.pivot(index="feature", columns="regime", values="Spearman")
    plt.figure(figsize=(8, 6))
    sns.heatmap(static_matrix, annot=True, fmt=".2f", center=0, cmap="coolwarm")
    plt.title("Static-feature association with ARMAS–NAIRAS residual")
    finish_figure("static_feature_residual_correlations.png")
    """),
    code(r"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, part) in zip(axes, raw.items()):
        sns.histplot(part.residual, bins=70, ax=ax)
        ax.axvline(0, color="red", linestyle="--")
        ax.set(title=f"{name}: ARMAS − NAIRAS V3", xlabel="Residual (µSv/h)")
    finish_figure("physics_residual_histograms.png", fig)
    """),
    md("""
    Reflection: look for feature rows whose correlation changes across lags rather than staying constant. Those are the strongest candidates for genuinely temporal information. Highly constant rows may be mostly current-condition indicators.
    """),
    md("""
    ## 4. Train-only preprocessing

    Separate scalers are fitted for static inputs, temporal inputs, current-time inputs, NAIRAS V3, direct ARMAS targets, and residual targets. Every scaler is learned from the training partition only in each rotation.
    """),
    code(r"""
    @dataclass
    class Preprocessors:
        static: object
        sequence: object
        current: object
        nairas: object
        direct_target: StandardScaler
        residual_target: StandardScaler

    @dataclass
    class ScaledPartition:
        name: str
        x_static: np.ndarray
        x_sequence: np.ndarray
        x_current: np.ndarray
        nairas_scaled: np.ndarray
        nairas_raw: np.ndarray
        direct_scaled: np.ndarray
        residual_scaled: np.ndarray
        y_raw: np.ndarray
        residual_raw: np.ndarray

    def fit_preprocessors(train):
        # Median imputation and standardization are both learned from training data only.
        def input_pipeline(values):
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(values)
        return Preprocessors(
            static=input_pipeline(train.x_static),
            sequence=input_pipeline(train.x_sequence.reshape(-1, 35)),
            current=input_pipeline(train.x_current),
            nairas=input_pipeline(train.nairas_v3.reshape(-1, 1)),
            direct_target=StandardScaler().fit(train.y.reshape(-1, 1)),
            residual_target=StandardScaler().fit(train.residual.reshape(-1, 1)),
        )

    def transform(part, prep):
        n = len(part.y)
        sequence = prep.sequence.transform(part.x_sequence.reshape(-1, 35)).reshape(n, 12, 35)
        return ScaledPartition(
            name=part.name,
            x_static=prep.static.transform(part.x_static).astype(np.float32),
            x_sequence=sequence.astype(np.float32),
            x_current=prep.current.transform(part.x_current).astype(np.float32),
            nairas_scaled=prep.nairas.transform(part.nairas_v3.reshape(-1, 1)).astype(np.float32),
            nairas_raw=part.nairas_v3,
            direct_scaled=prep.direct_target.transform(part.y.reshape(-1, 1)).ravel().astype(np.float32),
            residual_scaled=prep.residual_target.transform(part.residual.reshape(-1, 1)).ravel().astype(np.float32),
            y_raw=part.y,
            residual_raw=part.residual,
        )
    """),
    md("""
    ## 5. Model definitions

    All models share one forward signature. Inputs that a model is not allowed to use are ignored inside that model. This prevents accidental NAIRAS leakage into direct baselines.
    """),
    code(r"""
    class CurrentMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            h = config.hidden_dim
            self.net = nn.Sequential(
                nn.Linear(43, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(config.dropout),
                nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(h // 2, 1),
            )
        def forward(self, current, sequence, static, nairas, return_attention=False):
            output = self.net(current).squeeze(1)
            return (output, None) if return_attention else output

    class HybridCNNMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            c = config.cnn_channels
            self.temporal = nn.Sequential(
                nn.Conv1d(35, c, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(c),
                nn.Conv1d(c, c, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            self.static = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Dropout(config.dropout))
            self.head = nn.Sequential(nn.Linear(c + 32, 64), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(64, 1))
        def forward(self, current, sequence, static, nairas, return_attention=False):
            joined = torch.cat([self.temporal(sequence.transpose(1, 2)), self.static(static)], dim=1)
            output = self.head(joined).squeeze(1)
            return (output, None) if return_attention else output

    class AttentionBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.norm1 = nn.LayerNorm(config.d_model)
            self.attention = nn.MultiheadAttention(config.d_model, config.n_heads,
                                                   dropout=config.dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(config.d_model)
            self.ffn = nn.Sequential(nn.Linear(config.d_model, config.feedforward_dim), nn.GELU(),
                                     nn.Dropout(config.dropout), nn.Linear(config.feedforward_dim, config.d_model))
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x, need_weights=False):
            normalized = self.norm1(x)
            attended, weights = self.attention(normalized, normalized, normalized,
                                               need_weights=need_weights, average_attn_weights=False)
            x = x + self.dropout(attended)
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x, weights

    class TemporalEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            if config.d_model % config.n_heads:
                raise ValueError("d_model must be divisible by n_heads")
            self.pooling = config.pooling
            self.projection = nn.Linear(35, config.d_model)
            self.position = nn.Parameter(torch.zeros(1, 12, config.d_model))
            self.blocks = nn.ModuleList([AttentionBlock(config) for _ in range(config.n_layers)])
            self.norm = nn.LayerNorm(config.d_model)

        def forward(self, sequence, return_attention=False):
            steps = sequence.shape[1]
            x = self.projection(sequence) + self.position[:, -steps:, :]
            attention_maps = []
            for block in self.blocks:
                x, weights = block(x, need_weights=return_attention)
                if return_attention:
                    attention_maps.append(weights)
            x = self.norm(x)
            pooled = x[:, -1] if self.pooling == "current" else x.mean(dim=1)
            return pooled, attention_maps

    class DirectTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.temporal = TemporalEncoder(config)
            self.current = nn.Sequential(nn.Linear(43, 32), nn.GELU(), nn.Dropout(config.dropout))
            self.head = nn.Sequential(nn.Linear(config.d_model + 32, 64), nn.GELU(),
                                      nn.Dropout(config.dropout), nn.Linear(64, 1))
        def forward(self, current, sequence, static, nairas, return_attention=False):
            temporal, maps = self.temporal(sequence, return_attention)
            output = self.head(torch.cat([temporal, self.current(current)], dim=1)).squeeze(1)
            return (output, maps) if return_attention else output

    class PhysicsGuidedTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.temporal = TemporalEncoder(config)
            self.static = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Dropout(config.dropout))
            self.head = nn.Sequential(nn.Linear(config.d_model + 32 + 1, 64), nn.GELU(),
                                      nn.Dropout(config.dropout), nn.Linear(64, 1))
        def forward(self, current, sequence, static, nairas, return_attention=False):
            temporal, maps = self.temporal(sequence, return_attention)
            joined = torch.cat([temporal, self.static(static), nairas], dim=1)
            output = self.head(joined).squeeze(1)
            return (output, maps) if return_attention else output

    def build_model(config):
        classes = {"mlp": CurrentMLP, "cnn_hybrid": HybridCNNMLP,
                   "transformer_direct": DirectTransformer,
                   "transformer_residual": PhysicsGuidedTransformer}
        return classes[config.model_kind](config)

    architecture_table = pd.DataFrame([
        {"model": c.name, "target": c.target_mode,
         "parameters": sum(p.numel() for p in build_model(c).parameters() if p.requires_grad)}
        for c in STANDARD_CONFIGS
    ])
    display(architecture_table)
    """),
    code(r"""
    # VISUAL: exactly which information each model is allowed to use.
    model_design = pd.DataFrame({
        "43 current inputs": [1, 0, 1, 0],
        "12×35 sequence": [0, 1, 1, 1],
        "8 static inputs": [1, 1, 1, 1],
        "NAIRAS V3 input": [0, 0, 0, 1],
        "predicts ARMAS": [1, 1, 1, 0],
        "predicts residual": [0, 0, 0, 1],
    }, index=["MLP", "Hybrid CNN-MLP", "Direct Transformer", "Physics-guided Transformer"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.heatmap(model_design, annot=True, fmt="d", cmap="Blues", cbar=False,
                linewidths=.5, ax=axes[0])
    axes[0].set_title("Model inputs and prediction targets (1 = used)")
    sns.barplot(data=architecture_table.sort_values("parameters"), x="parameters", y="model",
                color="#457b9d", ax=axes[1])
    axes[1].set_xscale("log")
    axes[1].set_title("Trainable parameters by required model")
    finish_figure("model_design_and_parameter_counts.png", fig)
    """),
    md("""
    ### 5.1 Network complexity and deployment cost

    Accuracy alone does not establish that a larger model is worthwhile. This section measures trainable parameters, parameter storage, approximate forward-pass FLOPs, peak GPU memory, and latency. FLOPs are profiler estimates and hardware latency is device-specific, so report the device and batch size beside them. Use `layer_parameter_table(...)` when you want an architecture-by-architecture breakdown for an appendix.
    """),
    code(r"""
    def example_inputs(config, batch_size=64):
        return (
            torch.zeros(batch_size, 43, device=DEVICE),
            torch.zeros(batch_size, config.sequence_steps, 35, device=DEVICE),
            torch.zeros(batch_size, 8, device=DEVICE),
            torch.zeros(batch_size, 1, device=DEVICE),
        )

    def layer_parameter_table(model):
        rows = []
        for name, module in model.named_modules():
            direct_parameters = sum(p.numel() for p in module.parameters(recurse=False))
            if direct_parameters:
                rows.append({"layer": name or "<root>", "type": type(module).__name__,
                             "parameters": direct_parameters,
                             "trainable": sum(p.numel() for p in module.parameters(recurse=False)
                                              if p.requires_grad)})
        return pd.DataFrame(rows)

    def profile_model(config, batch_size=64, repeats=30):
        model = build_model(config).to(DEVICE).eval()
        inputs = example_inputs(config, batch_size)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

        with torch.no_grad():
            for _ in range(5): model(*inputs)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats): model(*inputs)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
        latency_ms = 1000 * (time.perf_counter() - start) / repeats

        approximate_flops = np.nan
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if DEVICE.type == "cuda": activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.no_grad(), torch.profiler.profile(activities=activities, with_flops=True) as prof:
                model(*inputs)
            approximate_flops = sum(event.flops for event in prof.key_averages()) / batch_size
        except Exception as error:
            print(f"FLOP profiling unavailable for {config.name}: {error}")

        result = {"model": config.name, "device": str(DEVICE), "batch_size": batch_size,
                  "total_parameters": total, "trainable_parameters": trainable,
                  "parameter_and_buffer_MB": (parameter_bytes + buffer_bytes) / 2**20,
                  "approx_FLOPs_per_observation": approximate_flops,
                  "batch_latency_ms": latency_ms,
                  "latency_ms_per_observation": latency_ms / batch_size}
        del model, inputs
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        return result

    RUN_COMPLEXITY_BENCHMARK = True
    if RUN_COMPLEXITY_BENCHMARK:
        complexity_table = pd.DataFrame([profile_model(config) for config in STANDARD_CONFIGS])
        display(complexity_table.round(4))
        complexity_table.to_csv(OUTPUT_DIR / "network_complexity.csv", index=False)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.scatterplot(data=complexity_table, x="trainable_parameters",
                        y="latency_ms_per_observation", hue="model", s=120, ax=axes[0])
        axes[0].set_xscale("log"); axes[0].set_title("Parameters versus inference latency")
        sns.barplot(data=complexity_table.sort_values("trainable_parameters"),
                    x="trainable_parameters", y="model", color="#457b9d", ax=axes[1])
        axes[1].set_xscale("log"); axes[1].set_title("Trainable parameter count")
        sns.barplot(data=complexity_table.sort_values("approx_FLOPs_per_observation"),
                    x="approx_FLOPs_per_observation", y="model", color="#e76f51", ax=axes[2])
        axes[2].set_xscale("log"); axes[2].set_title("Approximate FLOPs per observation")
        finish_figure("network_complexity_dashboard.png", fig)
    """),
    md("""
    ## 6. Training, reconstruction, and metrics

    The residual model is optimized on standardized residuals, then reconstructed as `NAIRAS V3 + predicted residual`. Every model is scored against measured ARMAS in physical units.
    """),
    code(r"""
    def make_loader(part, config, shuffle):
        sequence = part.x_sequence[:, -config.sequence_steps:, :]
        target = part.residual_scaled if config.target_mode == "residual" else part.direct_scaled
        dataset = TensorDataset(torch.from_numpy(part.x_current), torch.from_numpy(sequence),
                                torch.from_numpy(part.x_static), torch.from_numpy(part.nairas_scaled),
                                torch.from_numpy(target), torch.from_numpy(part.y_raw))
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    def predict_standardized(model, part, config):
        model.eval(); outputs = []
        with torch.no_grad():
            for current, sequence, static, nairas, _, _ in make_loader(part, config, False):
                outputs.append(model(current.to(DEVICE), sequence.to(DEVICE), static.to(DEVICE),
                                     nairas.to(DEVICE)).cpu().numpy())
        return np.concatenate(outputs)

    def reconstruct_armas(standardized_output, part, prep, config):
        if config.target_mode == "residual":
            residual = prep.residual_target.inverse_transform(standardized_output.reshape(-1, 1)).ravel()
            return part.nairas_raw + residual
        return prep.direct_target.inverse_transform(standardized_output.reshape(-1, 1)).ravel()

    def metric_row(y, prediction):
        return {"n": len(y), "MAE": mean_absolute_error(y, prediction),
                "RMSE": mean_squared_error(y, prediction) ** 0.5, "R2": r2_score(y, prediction),
                "bias": float(np.mean(prediction - y))}

    def evaluate_regimes(y, prediction):
        masks = {"all": np.ones(len(y), bool), "A <= 15": y <= 15, "A > 15": y > 15}
        return [{"regime": name, **metric_row(y[mask], prediction[mask])} for name, mask in masks.items()]

    def objective_loss(output, target, y_raw, config):
        # Use the same standardized-space objective for train and validation curves.
        if config.loss == "huber":
            per_row = F.smooth_l1_loss(output, target, reduction="none")
        elif config.loss == "mse":
            per_row = (output - target).square()
        else:
            raise ValueError("loss must be mse or huber")
        weights = torch.where(y_raw > 15, config.high_dose_weight, 1.0)
        return (per_row * weights).mean()

    def validation_objective(model, part, config):
        model.eval(); total, count = 0.0, 0
        with torch.no_grad():
            for current, sequence, static, nairas, target, y_raw in make_loader(part, config, False):
                tensors = [x.to(DEVICE) for x in (current, sequence, static, nairas, target, y_raw)]
                current, sequence, static, nairas, target, y_raw = tensors
                output = model(current, sequence, static, nairas)
                batch_loss = objective_loss(output, target, y_raw, config)
                total += batch_loss.item() * len(target); count += len(target)
        return total / count

    def train_model(model, train, validation, prep, config):
        seed_everything(config.seed)
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.lr_factor,
            patience=config.scheduler_patience, min_lr=config.min_lr)
        best_state, best_rmse, stale, history = None, float("inf"), 0, []
        training_start = time.perf_counter()
        if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.perf_counter()
            model.train(); train_total, train_count = 0.0, 0
            for current, sequence, static, nairas, target, y_raw in make_loader(train, config, True):
                tensors = [x.to(DEVICE) for x in (current, sequence, static, nairas, target, y_raw)]
                current, sequence, static, nairas, target, y_raw = tensors
                optimizer.zero_grad(set_to_none=True)
                output = model(current, sequence, static, nairas)
                loss = objective_loss(output, target, y_raw, config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_total += loss.item() * len(target); train_count += len(target)

            train_loss = train_total / train_count
            val_loss = validation_objective(model, validation, config)
            val_scaled = predict_standardized(model, validation, config)
            val_prediction = reconstruct_armas(val_scaled, validation, prep, config)
            validation_metrics = metric_row(validation.y_raw, val_prediction)
            val_rmse, val_mae = validation_metrics["RMSE"], validation_metrics["MAE"]
            current_lr = optimizer.param_groups[0]["lr"]
            improved = val_rmse < best_rmse - config.min_delta
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "val_MAE": val_mae, "val_RMSE": val_rmse,
                            "learning_rate": current_lr, "best_so_far": improved,
                            "epoch_seconds": time.perf_counter() - epoch_start,
                            "elapsed_seconds": time.perf_counter() - training_start,
                            "peak_gpu_memory_MB": (torch.cuda.max_memory_allocated() / 2**20
                                                   if DEVICE.type == "cuda" else np.nan)})
            print(f"{config.name} | epoch {epoch:03d} | train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | val RMSE {val_rmse:.4f} | lr {current_lr:.2e}")
            scheduler.step(val_rmse)
            if improved:
                best_rmse, stale, best_state = val_rmse, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= config.patience:
                    print(f"Early stopping; best validation RMSE = {best_rmse:.4f}"); break
        model.load_state_dict(best_state)
        return model, pd.DataFrame(history)
    """),
    md("""
    ## 7. P1 training and P2 model selection

    This stage intentionally does not evaluate P3. It trains the four promised learned models and the weighted-Huber alternatives, then adds NAIRAS V3 on the identical P2 observations.
    """),
    code(r"""
    prep = fit_preprocessors(raw["P1"])
    data = {name: transform(part, prep) for name, part in raw.items()}
    VALIDATION_CONFIGS = STANDARD_CONFIGS + WEIGHTED_HUBER_CONFIGS
    RUN_VALIDATION_EXPERIMENTS = True
    trained, histories, validation_predictions, validation_rows = {}, {}, {}, []

    if RUN_VALIDATION_EXPERIMENTS:
        experiment_phases = [
            ("PHASE 1 — overall, unweighted model comparison", STANDARD_CONFIGS),
            ("PHASE 2 — imbalance-aware weighted-Huber comparison", WEIGHTED_HUBER_CONFIGS),
        ]
        for phase_name, phase_configs in experiment_phases:
            print(f"\n{'=' * 20} {phase_name} {'=' * 20}")
            for config in phase_configs:
                print(f"\n===== {config.name} =====")
                model, history = train_model(build_model(config), data["P1"], data["P2"], prep, config)
                output = predict_standardized(model, data["P2"], config)
                prediction = reconstruct_armas(output, data["P2"], prep, config)
                trained[config.name] = (model, config)
                histories[config.name] = history
                validation_predictions[config.name] = prediction
                validation_rows += [{"model": config.name, **row}
                                    for row in evaluate_regimes(data["P2"].y_raw, prediction)]
        validation_rows += [{"model": "NAIRAS V3", **row}
                            for row in evaluate_regimes(data["P2"].y_raw, data["P2"].nairas_raw)]
        validation_results = pd.DataFrame(validation_rows)
        baseline_rmse = (validation_results[validation_results.model == "NAIRAS V3"]
                         .set_index("regime")["RMSE"])
        validation_results["RMSE_skill_vs_NAIRAS_percent"] = validation_results.apply(
            lambda row: 100 * (baseline_rmse[row.regime] - row.RMSE) / baseline_rmse[row.regime], axis=1)
        print("PHASE 1 RESULT — overall metrics on every P2 observation")
        display(validation_results[validation_results.regime == "all"].sort_values("RMSE").round(4))
        print("PHASE 2 RESULT — the same predictions divided by radiation regime")
        display(validation_results[validation_results.regime != "all"]
                .sort_values(["regime", "RMSE"]).round(4))
        validation_results.to_csv(OUTPUT_DIR / "p2_validation_results.csv", index=False)
    """),
    md("""
    ### 7.0 Visual normalization check

    **What you need to do:** confirm P1 transformed inputs are centered near zero with scale near one. P2/P3 do not need to be perfectly centered because they use P1 statistics; visible offsets are evidence of distribution shift, not a reason to refit the scaler on validation or test data.
    """),
    code(r"""
    normalization_rows = []
    for partition_name, part in data.items():
        for block_name, values in {
                "static": part.x_static,
                "current": part.x_current,
                "temporal": part.x_sequence,
                "direct target": part.direct_scaled,
                "residual target": part.residual_scaled}.items():
            flat = np.asarray(values).ravel()
            normalization_rows.append({"partition": partition_name, "block": block_name,
                                       "mean": flat.mean(), "std": flat.std(),
                                       "p01": np.quantile(flat, .01),
                                       "median": np.median(flat), "p99": np.quantile(flat, .99)})
    normalization_audit = pd.DataFrame(normalization_rows)
    display(normalization_audit.round(3))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=normalization_audit, x="block", y="mean", hue="partition", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Mean after P1-fitted preprocessing")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=normalization_audit, x="block", y="std", hue="partition", ax=axes[1])
    axes[1].axhline(1, color="black", linestyle="--")
    axes[1].set_title("Standard deviation after P1-fitted preprocessing")
    axes[1].tick_params(axis="x", rotation=30)
    finish_figure("normalization_audit.png", fig)
    normalization_audit.to_csv(OUTPUT_DIR / "normalization_audit.csv", index=False)
    """),
    md("""
    ### 7.1 Learning curves and convergence diagnosis

    These curves answer more than "did the loss go down?" The standardized training and validation objectives reveal underfitting or overfitting, while validation RMSE stays in the physical unit (µSv/h). The dot marks the checkpoint selected by validation RMSE. A widening train/validation gap suggests overfitting; two high, flat curves suggest underfitting; a falling curve at the final epoch suggests that the epoch budget was too small.

    **What you need to do:** if validation RMSE is still falling at the last epoch, switch to a longer run mode. If training loss falls while validation loss rises, prefer more regularization or a smaller model. If both curves are flat, inspect the learning-rate panel before changing capacity. Never choose an epoch using P3.
    """),
    code(r"""
    def plot_learning_curves(history_map, label="models", save=True):
        if not history_map:
            print("No histories yet. Run the validation experiments first.")
            return

        names = list(history_map)
        ncols = 2
        nrows = int(np.ceil(len(names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            history = history_map[name]
            ax.plot(history.epoch, history.train_loss, label="train objective", linewidth=2)
            ax.plot(history.epoch, history.val_loss, label="validation objective", linewidth=2)
            best_index = history.val_RMSE.idxmin()
            best_epoch = int(history.loc[best_index, "epoch"])
            ax.axvline(best_epoch, color="black", linestyle=":", alpha=.7,
                       label=f"best epoch = {best_epoch}")
            ax.set(title=name, xlabel="epoch", ylabel="standardized objective")
            ax.legend(fontsize=8)
        for ax in axes.ravel()[len(names):]:
            ax.set_visible(False)
        fig.suptitle("Training and validation loss curves", y=1.01, fontsize=15)
        fig.tight_layout()
        if save:
            fig.savefig(OUTPUT_DIR / f"learning_curves_loss_{label}.png", dpi=180, bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots(figsize=(11, 5))
        for name, history in history_map.items():
            ax.plot(history.epoch, history.val_RMSE, label=name, linewidth=1.8)
            best = history.loc[history.val_RMSE.idxmin()]
            ax.scatter(best.epoch, best.val_RMSE, s=30)
        ax.set(xlabel="epoch", ylabel="P2 validation RMSE (µSv/h)",
               title="Model convergence in physical units")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        if save:
            fig.savefig(OUTPUT_DIR / f"learning_curves_validation_rmse_{label}.png", dpi=180,
                        bbox_inches="tight")
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for name, history in history_map.items():
            axes[0, 0].plot(history.epoch, history.learning_rate, label=name)
            axes[0, 1].plot(history.epoch, history.val_loss - history.train_loss, label=name)
            axes[1, 0].plot(history.epoch, history.epoch_seconds, label=name)
            axes[1, 1].plot(history.epoch, history.elapsed_seconds / 60, label=name)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set(title="Learning-rate schedule", xlabel="epoch", ylabel="learning rate")
        axes[0, 1].axhline(0, color="black", linewidth=1)
        axes[0, 1].set(title="Validation minus training objective", xlabel="epoch", ylabel="objective gap")
        axes[1, 0].set(title="Time per epoch", xlabel="epoch", ylabel="seconds")
        axes[1, 1].set(title="Cumulative training time", xlabel="epoch", ylabel="minutes")
        axes[0, 0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.suptitle("Optimization and computational diagnostics", y=1.01)
        fig.tight_layout()
        if save:
            fig.savefig(OUTPUT_DIR / f"optimization_diagnostics_{label}.png", dpi=180,
                        bbox_inches="tight")
        plt.show()

    standard_histories = {name: histories[name] for name in [c.name for c in STANDARD_CONFIGS]
                          if name in histories}
    imbalance_histories = {name: histories[name] for name in [c.name for c in WEIGHTED_HUBER_CONFIGS]
                           if name in histories}
    print("PHASE 1 LOSS CURVES — ordinary training on all observations")
    plot_learning_curves(standard_histories, label="phase1_standard")
    print("PHASE 2 LOSS CURVES — weighted-Huber imbalance experiments")
    plot_learning_curves(imbalance_histories, label="phase2_weighted_huber")
    """),
    code(r"""
    if validation_rows:
        comparison = validation_results.pivot(index="model", columns="regime", values="RMSE")
        display(comparison.sort_values("all").round(4))
        comparison.plot(kind="bar", figsize=(12, 5))
        plt.ylabel("RMSE (µSv/h)"); plt.title("P2 validation trade-off by radiation regime")
        finish_figure("p2_rmse_by_model_and_regime.png")

        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        sns.heatmap(comparison.sort_values("all"), annot=True, fmt=".3f", cmap="viridis_r",
                    cbar_kws={"label": "RMSE (µSv/h)"}, ax=axes[0])
        axes[0].set_title("P2 RMSE matrix")
        skill_plot = validation_results.copy()
        sns.barplot(data=skill_plot, x="RMSE_skill_vs_NAIRAS_percent", y="model",
                    hue="regime", ax=axes[1])
        axes[1].axvline(0, color="black", linewidth=1)
        axes[1].set_title("RMSE skill relative to NAIRAS V3")
        tradeoff = comparison.reset_index()
        axes[2].scatter(tradeoff["all"], tradeoff["A > 15"], s=80)
        for _, row in tradeoff.iterrows():
            axes[2].annotate(row.model, (row["all"], row["A > 15"]),
                             xytext=(4, 3), textcoords="offset points", fontsize=7)
        axes[2].set(xlabel="Overall RMSE", ylabel="High-dose RMSE",
                    title="Overall versus high-dose trade-off")
        finish_figure("p2_validation_performance_dashboard.png", fig)

        if "complexity_table" in globals():
            accuracy_efficiency = (validation_results[validation_results.regime == "all"]
                                   .merge(complexity_table, on="model", how="inner"))
            display(accuracy_efficiency[["model", "RMSE", "MAE", "trainable_parameters",
                                         "approx_FLOPs_per_observation",
                                         "latency_ms_per_observation"]].sort_values("RMSE").round(4))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(accuracy_efficiency.trainable_parameters, accuracy_efficiency.RMSE,
                       s=100, c=accuracy_efficiency.latency_ms_per_observation, cmap="viridis")
            for _, row in accuracy_efficiency.iterrows():
                ax.annotate(row.model, (row.trainable_parameters, row.RMSE),
                            xytext=(5, 4), textcoords="offset points", fontsize=8)
            ax.set_xscale("log")
            ax.set(xlabel="Trainable parameters (log scale)", ylabel="P2 RMSE (µSv/h)",
                   title="Accuracy-complexity trade-off (color = latency)")
            finish_figure("p2_accuracy_complexity_tradeoff.png", fig)
            accuracy_efficiency.to_csv(OUTPUT_DIR / "accuracy_efficiency_tradeoff.csv", index=False)
    """),
    md("""
    ### 7.2 Optional validation-only architecture and hyperparameter study

    Set `RUN_HYPERPARAMETER_STUDY=True` after the baseline run. The curated trials each change one interpretable design choice: capacity, depth, temporal window, pooling, optimization, or regularization. This is preferable to an enormous grid because it gives a defensible ablation story within a realistic Colab budget.

    **Protocol:** train on P1, compare on P2, keep P3 locked, choose one configuration, and only then evaluate P3. Report both overall and high-dose RMSE; do not select from training loss alone. The 1-, 6-, and 12-step variants directly test whether an hour of history adds information beyond the current measurement.
    """),
    code(r"""
    # USER ACTION: this is automatically ON for development/full and OFF for smoke.
    # Keep it on until the P2 configuration decision is frozen; never rerun it to chase P3.
    RUN_HYPERPARAMETER_STUDY = RUN_MODE != "smoke"
    TUNING_EPOCHS = {"smoke": 3, "development": 20, "full": 60}[RUN_MODE]
    base = STANDARD_CONFIGS[-1]
    TUNING_CONFIGS = [
        replace(base, name="PGT small", d_model=32, n_heads=4, n_layers=1,
                feedforward_dim=64, epochs=TUNING_EPOCHS),
        replace(base, name="PGT wide", d_model=128, n_heads=8, n_layers=2,
                feedforward_dim=256, dropout=.15, epochs=TUNING_EPOCHS),
        replace(base, name="PGT deep", d_model=64, n_heads=4, n_layers=3,
                feedforward_dim=192, dropout=.15, epochs=TUNING_EPOCHS),
        replace(base, name="PGT mean pooling", pooling="mean", epochs=TUNING_EPOCHS),
        replace(base, name="PGT 30-minute window", sequence_steps=6, epochs=TUNING_EPOCHS),
        replace(base, name="PGT current step only", sequence_steps=1, epochs=TUNING_EPOCHS),
        replace(base, name="PGT faster LR", learning_rate=1e-3, epochs=TUNING_EPOCHS),
        replace(base, name="PGT stronger regularization", dropout=.25, weight_decay=1e-3,
                epochs=TUNING_EPOCHS),
    ]

    tuning_models, tuning_histories, tuning_predictions, tuning_rows = {}, {}, {}, []
    if RUN_HYPERPARAMETER_STUDY:
        if RUN_MODE == "smoke":
            print("Smoke-mode tuning checks code only; use development/full for conclusions.")
        for config in TUNING_CONFIGS:
            print(f"\n===== TUNING: {config.name} =====")
            model, history = train_model(build_model(config), data["P1"], data["P2"], prep, config)
            prediction = reconstruct_armas(
                predict_standardized(model, data["P2"], config), data["P2"], prep, config)
            tuning_models[config.name] = (model, config)
            tuning_histories[config.name] = history
            tuning_predictions[config.name] = prediction
            rows = evaluate_regimes(data["P2"].y_raw, prediction)
            for row in rows:
                tuning_rows.append({
                    "model": config.name, "d_model": config.d_model,
                    "n_heads": config.n_heads, "n_layers": config.n_layers,
                    "feedforward_dim": config.feedforward_dim, "dropout": config.dropout,
                    "pooling": config.pooling, "sequence_steps": config.sequence_steps,
                    "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
                    "best_epoch": int(history.loc[history.val_RMSE.idxmin(), "epoch"]), **row})

        tuning_results = pd.DataFrame(tuning_rows)
        trained.update(tuning_models)
        histories.update(tuning_histories)
        validation_predictions.update(tuning_predictions)
        tuning_results.to_csv(OUTPUT_DIR / "p2_hyperparameter_study.csv", index=False)
        tuning_pivot = tuning_results.pivot(index="model", columns="regime", values="RMSE")
        display(tuning_pivot.sort_values("all").round(4))
        best_tuned_name = tuning_pivot["all"].idxmin()
        print(f"Best tuned configuration by prespecified overall P2 RMSE: {best_tuned_name}")
        print("This validation-selected configuration will be included in the downstream P2 diagnostics. "
              "Confirm the trade-off before unlocking P3.")
        plot_learning_curves(tuning_histories, label="hyperparameter_study")

        plot_data = tuning_results[tuning_results.regime.isin(["all", "A > 15"])]
        plt.figure(figsize=(11, 5))
        sns.barplot(data=plot_data, x="RMSE", y="model", hue="regime")
        plt.title("P2 tuning trade-off: overall versus high-dose error")
        finish_figure("p2_hyperparameter_overall_high_tradeoff.png")

        tuning_overall = tuning_results[tuning_results.regime == "all"].copy()
        tuning_overall["parameters"] = [
            sum(p.numel() for p in build_model(tuning_models[name][1]).parameters() if p.requires_grad)
            for name in tuning_overall.model]
        tuning_high = (tuning_results[tuning_results.regime == "A > 15"]
                       [["model", "RMSE"]].rename(columns={"RMSE": "high_dose_RMSE"}))
        tuning_visual = tuning_overall.merge(tuning_high, on="model")
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        sns.scatterplot(data=tuning_visual, x="parameters", y="RMSE", hue="sequence_steps",
                        size="d_model", sizes=(60, 220), ax=axes[0, 0])
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_title("Tuned-model accuracy versus parameter count")
        sns.scatterplot(data=tuning_visual, x="RMSE", y="high_dose_RMSE",
                        hue="dropout", size="n_layers", sizes=(70, 200), ax=axes[0, 1])
        for _, row in tuning_visual.iterrows():
            axes[0, 1].annotate(row.model, (row.RMSE, row.high_dose_RMSE),
                                xytext=(3, 3), textcoords="offset points", fontsize=7)
        axes[0, 1].set_title("Overall versus high-dose tuning trade-off")
        sns.scatterplot(data=tuning_visual, x="best_epoch", y="RMSE", hue="learning_rate",
                        style="pooling", s=110, ax=axes[1, 0])
        axes[1, 0].set_title("Convergence epoch versus P2 RMSE")
        normalized_columns = ["d_model", "n_layers", "feedforward_dim", "dropout",
                              "sequence_steps", "learning_rate", "weight_decay", "RMSE"]
        parallel = tuning_visual.set_index("model")[normalized_columns].copy()
        parallel = (parallel - parallel.min()) / (parallel.max() - parallel.min()).replace(0, 1)
        sns.heatmap(parallel, annot=True, fmt=".2f", cmap="mako", ax=axes[1, 1])
        axes[1, 1].set_title("Normalized hyperparameters and RMSE")
        finish_figure("p2_hyperparameter_visual_dashboard.png", fig)

        sequence_window = pd.DataFrame([
            {"window": "current only", "steps": 1,
             "RMSE": tuning_pivot.loc["PGT current step only", "all"]},
            {"window": "30 minutes", "steps": 6,
             "RMSE": tuning_pivot.loc["PGT 30-minute window", "all"]},
            {"window": "60 minutes", "steps": 12,
             "RMSE": result_value(validation_results, "Physics-guided Transformer")
                     if "result_value" in globals() else float(validation_results[
                         (validation_results.model == "Physics-guided Transformer") &
                         (validation_results.regime == "all")].RMSE.iloc[0])},
        ])
        plt.figure(figsize=(7, 4))
        sns.lineplot(data=sequence_window, x="steps", y="RMSE", marker="o")
        for _, row in sequence_window.iterrows():
            plt.annotate(row.window, (row.steps, row.RMSE), xytext=(4, 4),
                         textcoords="offset points")
        plt.xticks([1, 6, 12])
        plt.title("Does a longer temporal window improve P2 RMSE?")
        finish_figure("p2_sequence_window_ablation.png")
    else:
        print("Hyperparameter study is off. Set RUN_HYPERPARAMETER_STUDY=True to run it.")
    """),
    md("""
    ### 7.3 Optional random-seed stability

    A single neural-network run can be unusually good or bad. After choosing an architecture on P2, run it with several seeds and report mean ± standard deviation. If the performance difference between two models is smaller than seed-to-seed variation, do not claim a decisive improvement.

    **What you need to do:** leave this off while changing architectures. Turn it on once for the chosen configuration before writing the final conclusion.
    """),
    code(r"""
    # USER ACTION: change to True only after architecture/loss selection is finished.
    RUN_SEED_STABILITY = False
    STABILITY_SEEDS = [7, 42, 2026, 8851, 12345]
    STABILITY_BASE_CONFIG = STANDARD_CONFIGS[-1]
    seed_rows, seed_histories = [], {}

    if RUN_SEED_STABILITY:
        for seed in STABILITY_SEEDS:
            config = replace(STABILITY_BASE_CONFIG, name=f"{STABILITY_BASE_CONFIG.name} seed={seed}",
                             seed=seed)
            print(f"\n===== SEED STABILITY: {seed} =====")
            model, history = train_model(build_model(config), data["P1"], data["P2"], prep, config)
            prediction = reconstruct_armas(
                predict_standardized(model, data["P2"], config), data["P2"], prep, config)
            seed_histories[config.name] = history
            seed_rows += [{"seed": seed, **row}
                          for row in evaluate_regimes(data["P2"].y_raw, prediction)]
        seed_results = pd.DataFrame(seed_rows)
        seed_summary = seed_results.groupby("regime")[["MAE", "RMSE", "R2", "bias"]].agg(["mean", "std"])
        display(seed_summary.round(4))
        seed_results.to_csv(OUTPUT_DIR / "p2_seed_stability.csv", index=False)
        plot_learning_curves(seed_histories, label="seed_stability")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.pointplot(data=seed_results, x="seed", y="RMSE", hue="regime", ax=axes[0])
        axes[0].set_title("RMSE variation across random seeds")
        sns.boxplot(data=seed_results, x="regime", y="RMSE", ax=axes[1])
        sns.stripplot(data=seed_results, x="regime", y="RMSE", color="black", ax=axes[1])
        axes[1].set_title("Distribution of seed-level RMSE")
        finish_figure("p2_seed_stability.png", fig)
    else:
        print("Seed stability is off. Enable it after selecting an architecture.")
    """),
    md("""
    Decision questions:

    - Does the direct Transformer beat the current-time MLP and hybrid CNN–MLP?
    - Does residual learning beat both the direct Transformer and uncorrected NAIRAS V3?
    - Which Huber weight improves `A > 15` RMSE without an unacceptable increase for `A <= 15`?
    - Do not choose using P3.
    """),
    md("""
    ## 8. Freeze configurations and unlock P3 once

    The cell below applies a transparent P2-only selection rule for weighted Huber and carries forward the best optional tuning trial. Review the printed list, freeze it, and then set `EVALUATE_P3=True` once. Do not revise it after viewing P3.
    """),
    code(r"""
    # Prespecified validation rule: minimize high-dose RMSE among weighted-Huber models
    # whose lower-dose RMSE is no more than 5% worse than the standard residual model.
    standard_low_rmse = result_value(validation_results, "Physics-guided Transformer", "A <= 15") \
        if "result_value" in globals() else float(validation_results[
            (validation_results.model == "Physics-guided Transformer") &
            (validation_results.regime == "A <= 15")].RMSE.iloc[0])
    weighted_table = validation_results[validation_results.model.str.contains("Huber")]
    weighted_wide = weighted_table.pivot(index="model", columns="regime", values="RMSE")
    eligible_weighted = weighted_wide[weighted_wide["A <= 15"] <= 1.05 * standard_low_rmse]
    if eligible_weighted.empty:
        eligible_weighted = weighted_wide
        print("No weighted model met the 5% lower-dose constraint; selecting by high-dose RMSE.")
    SELECTED_WEIGHTED_NAME = eligible_weighted["A > 15"].idxmin()

    FINAL_CONFIG_NAMES = [
        "MLP", "Hybrid CNN-MLP", "Direct Transformer",
        "Physics-guided Transformer", SELECTED_WEIGHTED_NAME,
    ]
    if RUN_HYPERPARAMETER_STUDY and "best_tuned_name" in globals():
        FINAL_CONFIG_NAMES.append(best_tuned_name)
    print("Configurations carried forward:", FINAL_CONFIG_NAMES)
    # ============================= USER ACTION =============================
    # Keep False while you are still comparing architectures or losses on P2.
    # Change to True exactly once after the printed configuration list is frozen.
    # After it becomes True, do NOT return to tuning based on the P3 results.
    # =======================================================================
    EVALUATE_P3 = True
    test_predictions, test_rows = {}, []

    if EVALUATE_P3:
        for name in FINAL_CONFIG_NAMES:
            if name not in trained:
                raise ValueError(f"{name} was not trained in the validation section")
            model, config = trained[name]
            output = predict_standardized(model, data["P3"], config)
            prediction = reconstruct_armas(output, data["P3"], prep, config)
            test_predictions[name] = prediction
            test_rows += [{"model": name, **row} for row in evaluate_regimes(data["P3"].y_raw, prediction)]
        test_predictions["NAIRAS V3"] = data["P3"].nairas_raw
        test_rows += [{"model": "NAIRAS V3", **row}
                      for row in evaluate_regimes(data["P3"].y_raw, data["P3"].nairas_raw)]
        p3_results = pd.DataFrame(test_rows)
        p3_baseline_rmse = p3_results[p3_results.model == "NAIRAS V3"].set_index("regime")["RMSE"]
        p3_results["RMSE_skill_vs_NAIRAS_percent"] = p3_results.apply(
            lambda row: 100 * (p3_baseline_rmse[row.regime] - row.RMSE) /
                        p3_baseline_rmse[row.regime], axis=1)
        print("LOCKED TEST — overall metrics first")
        display(p3_results[p3_results.regime == "all"].sort_values("RMSE").round(4))
        print("LOCKED TEST — below/above-threshold metrics second")
        display(p3_results[p3_results.regime != "all"].sort_values(["regime", "RMSE"]).round(4))
        p3_results.to_csv(OUTPUT_DIR / "p3_locked_test_results.csv", index=False)
    else:
        print("P3 test evaluation is disabled; enable it only after configurations are selected on P2.")

    # Downstream diagnostic cells always produce output. Before the final test they use
    # P2 and are explicitly labeled validation/exploratory; after unlocking, they use P3.
    if EVALUATE_P3:
        ANALYSIS_PARTITION = "P3"
        analysis_data = data["P3"]
        analysis_predictions = test_predictions
        analysis_label = "locked test"
    else:
        ANALYSIS_PARTITION = "P2"
        analysis_data = data["P2"]
        analysis_predictions = {name: validation_predictions[name]
                                for name in FINAL_CONFIG_NAMES if name in validation_predictions}
        analysis_predictions["NAIRAS V3"] = data["P2"].nairas_raw
        analysis_label = "validation/exploratory"
    print(f"Sections 9-12 will analyze {ANALYSIS_PARTITION} ({analysis_label}).")
    """),
    md("""
    ## 9. Required prediction and residual figures

    These figures use P2 as validation when P3 testing is disabled. With `EVALUATE_P3=True`, the same cells use the final P3 test predictions.
    """),
    code(r"""
    if analysis_predictions:
        figure_models = ["NAIRAS V3", "Hybrid CNN-MLP", "Direct Transformer", FINAL_CONFIG_NAMES[-1]]
        figure_models = [name for name in figure_models if name in analysis_predictions]
        fig, axes = plt.subplots(len(figure_models), 2, figsize=(12, 4 * len(figure_models)))
        axes = np.atleast_2d(axes)
        y = analysis_data.y_raw
        for row, name in enumerate(figure_models):
            prediction = analysis_predictions[name]
            residual = prediction - y
            axes[row, 0].hexbin(y, prediction, gridsize=45, mincnt=1, cmap="viridis")
            limit = max(y.max(), prediction.max())
            axes[row, 0].plot([0, limit], [0, limit], "r--")
            axes[row, 0].axvline(15, color="white", linestyle=":")
            axes[row, 0].set(title=f"{name} — {ANALYSIS_PARTITION} {analysis_label}",
                             xlabel="Measured ARMAS", ylabel="Predicted ARMAS")
            sns.histplot(residual, bins=70, ax=axes[row, 1])
            axes[row, 1].axvline(0, color="red", linestyle="--")
            axes[row, 1].set(title=f"{name}: residual", xlabel="Prediction − ARMAS")
        finish_figure(f"{ANALYSIS_PARTITION.lower()}_prediction_and_residual_panels.png", fig)

        # VISUAL: compare full absolute-error distributions and systematic bias.
        error_rows = []
        for name in figure_models:
            prediction = analysis_predictions[name]
            error_rows.append(pd.DataFrame({"model": name, "observed": y,
                                            "predicted": prediction,
                                            "residual": prediction - y,
                                            "absolute_error": np.abs(prediction - y)}))
        error_frame = pd.concat(error_rows, ignore_index=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.ecdfplot(data=error_frame, x="absolute_error", hue="model", ax=axes[0])
        axes[0].set(title="Absolute-error cumulative distribution", xlabel="Absolute error (µSv/h)")
        sns.boxenplot(data=error_frame, x="residual", y="model", ax=axes[1])
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_title("Residual spread and bias")
        best_name = min(figure_models, key=lambda name: np.sqrt(
            np.mean((analysis_predictions[name] - y) ** 2)))
        best_prediction = analysis_predictions[best_name]
        mean_pair = (best_prediction + y) / 2
        difference = best_prediction - y
        sample = np.random.default_rng(42).choice(len(y), min(12000, len(y)), replace=False)
        axes[2].scatter(mean_pair[sample], difference[sample], alpha=.18, s=9)
        axes[2].axhline(difference.mean(), color="red", label="mean bias")
        axes[2].axhline(difference.mean() + 1.96 * difference.std(), color="black", linestyle=":")
        axes[2].axhline(difference.mean() - 1.96 * difference.std(), color="black", linestyle=":")
        axes[2].set(title=f"Bland–Altman view: {best_name}",
                    xlabel="Mean of prediction and observation", ylabel="Prediction − observation")
        finish_figure(f"{ANALYSIS_PARTITION.lower()}_error_distribution_dashboard.png", fig)
    """),
    md("""
    ### 9.1 Error anatomy and uncertainty

    Aggregate RMSE can hide where a model fails. The next analysis reports error across measured-dose bins, calibration by predicted-dose decile, and paired bootstrap confidence intervals for RMSE improvement over NAIRAS V3. Negative `RMSE_difference_vs_NAIRAS` favors the learned model. The row bootstrap is descriptive because observations within flights may be correlated; the three partition rotations remain the stronger robustness check.

    **What you need to do:** report the dose bins where improvement occurs, not only the overall average. Treat an improvement as uncertain when its bootstrap interval crosses zero. If calibration bends away from the diagonal at high values, explicitly describe high-dose under- or over-prediction.
    """),
    code(r"""
    def dose_bin_metrics(y, predictions):
        bins = [-np.inf, 5, 10, 15, 20, np.inf]
        labels = ["<=5", "5-10", "10-15", "15-20", ">20"]
        dose_bin = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
        rows = []
        for model_name, prediction in predictions.items():
            for label in labels:
                mask = np.asarray(dose_bin == label)
                if mask.sum() == 0: continue
                rows.append({"model": model_name, "dose_bin": label,
                             **metric_row(y[mask], prediction[mask])})
        return pd.DataFrame(rows)

    def calibration_table(y, predictions, quantiles=10):
        rows = []
        for model_name, prediction in predictions.items():
            groups = pd.qcut(prediction, q=quantiles, duplicates="drop")
            frame = pd.DataFrame({"observed": y, "predicted": prediction, "group": groups})
            summary = frame.groupby("group", observed=True).agg(
                n=("observed", "size"), mean_observed=("observed", "mean"),
                mean_predicted=("predicted", "mean")).reset_index(drop=True)
            summary["model"] = model_name
            rows.append(summary)
        return pd.concat(rows, ignore_index=True)

    def paired_bootstrap_rmse(y, candidate, baseline, samples=1000, seed=42):
        rng = np.random.default_rng(seed)
        differences = np.empty(samples)
        for i in range(samples):
            index = rng.integers(0, len(y), len(y))
            candidate_rmse = np.sqrt(np.mean((candidate[index] - y[index]) ** 2))
            baseline_rmse = np.sqrt(np.mean((baseline[index] - y[index]) ** 2))
            differences[i] = candidate_rmse - baseline_rmse
        return np.quantile(differences, [.025, .5, .975])

    if analysis_predictions:
        y = analysis_data.y_raw
        dose_diagnostics = dose_bin_metrics(y, analysis_predictions)
        calibration = calibration_table(y, analysis_predictions)
        display(dose_diagnostics.pivot(index="model", columns="dose_bin", values="RMSE").round(4))

        bootstrap_rows = []
        baseline = analysis_predictions["NAIRAS V3"]
        for name, prediction in analysis_predictions.items():
            if name == "NAIRAS V3": continue
            for regime, mask in {"all": np.ones(len(y), bool), "A <= 15": y <= 15,
                                 "A > 15": y > 15}.items():
                low, median, high = paired_bootstrap_rmse(y[mask], prediction[mask], baseline[mask])
                bootstrap_rows.append({"model": name, "regime": regime,
                                       "difference_median": median, "CI_2.5%": low, "CI_97.5%": high})
        bootstrap_results = pd.DataFrame(bootstrap_rows)
        display(bootstrap_results.round(4))
        output_prefix = ANALYSIS_PARTITION.lower()
        dose_diagnostics.to_csv(OUTPUT_DIR / f"{output_prefix}_error_by_dose_bin.csv", index=False)
        calibration.to_csv(OUTPUT_DIR / f"{output_prefix}_calibration.csv", index=False)
        bootstrap_results.to_csv(OUTPUT_DIR / f"{output_prefix}_bootstrap_vs_nairas.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.lineplot(data=dose_diagnostics, x="dose_bin", y="RMSE", hue="model", marker="o", ax=axes[0])
        axes[0].set_title(f"{ANALYSIS_PARTITION} RMSE by measured-dose bin ({analysis_label})")
        for name, group in calibration.groupby("model"):
            axes[1].plot(group.mean_predicted, group.mean_observed, marker="o", label=name)
        limit = max(calibration.mean_predicted.max(), calibration.mean_observed.max())
        axes[1].plot([0, limit], [0, limit], "k--", label="perfect calibration")
        axes[1].set(xlabel="Mean predicted ARMAS", ylabel="Mean observed ARMAS",
                    title=f"{ANALYSIS_PARTITION} calibration by prediction decile ({analysis_label})")
        axes[1].legend(fontsize=8)
        finish_figure(f"{ANALYSIS_PARTITION.lower()}_dose_bin_and_calibration.png", fig)
    """),
    md("""
    ## 10. Attention maps

    Attention shows which time positions exchange information inside the Transformer. It is useful context but is not, by itself, a complete feature attribution method.

    **What you need to do:** look for concentrated rows/columns or emphasis on recent versus earlier measurements. Compare the pattern with integrated gradients and permutation importance. Do not write that attention proves a variable caused the prediction.
    """),
    code(r"""
    def mean_attention_map(model, part, config, sample_size=512):
        model.eval()
        rng = np.random.default_rng(42)
        indices = rng.choice(len(part.y_raw), min(sample_size, len(part.y_raw)), replace=False)
        sequence = torch.from_numpy(part.x_sequence[indices, -config.sequence_steps:]).to(DEVICE)
        current = torch.from_numpy(part.x_current[indices]).to(DEVICE)
        static = torch.from_numpy(part.x_static[indices]).to(DEVICE)
        nairas = torch.from_numpy(part.nairas_scaled[indices]).to(DEVICE)
        with torch.no_grad():
            _, maps = model(current, sequence, static, nairas, return_attention=True)
        # Average final layer across samples and heads.
        return maps[-1].mean(dim=(0, 1)).cpu().numpy()

    if analysis_predictions:
        interpretation_name = FINAL_CONFIG_NAMES[-1]
        interpretation_model, interpretation_config = trained[interpretation_name]
        attention = mean_attention_map(interpretation_model, analysis_data, interpretation_config)
        labels = [f"-{5*(attention.shape[0]-1-i)}m" if i < attention.shape[0]-1 else "current"
                  for i in range(attention.shape[0])]
        plt.figure(figsize=(7, 6))
        sns.heatmap(attention, xticklabels=labels, yticklabels=labels, cmap="mako")
        plt.xlabel("Attended-to time"); plt.ylabel("Query time")
        plt.title(f"Mean final-layer attention on {ANALYSIS_PARTITION} ({analysis_label})")
        finish_figure(f"{ANALYSIS_PARTITION.lower()}_attention_map.png")
    """),
    md("""
    ## 11. Integrated gradients and variable-by-lag heatmap

    Integrated gradients (IG) asks: **how did each input move this prediction away from a reference prediction?** It integrates gradients along the straight path from a baseline input to the observed input:

    `IG_i(x) = (x_i - x'_i) × integral from 0 to 1 of ∂F(x' + α(x - x')) / ∂x_i dα`.

    Here, zero is a meaningful baseline because every input was standardized using P1 statistics: zero approximately represents the training-set mean (with median imputation before scaling). For the physics-guided model, IG explains the model's **predicted standardized correction**, not the whole NAIRAS value.

    How to read the outputs:

    - **Absolute IG magnitude** measures influence strength and is used for ranking and heatmaps.
    - **Signed IG** indicates whether a feature moved the predicted correction upward or downward relative to the baseline.
    - IG is local to the selected observations; compare a representative overall sample with a high-dose sample.
    - Similar importance does not prove a physical cause. Correlated inputs can share or redistribute attribution.
    - The completeness check verifies that summed attributions approximately equal `model(x) - model(baseline)`.
    """),
    code(r"""
    def integrated_gradients(model, part, config, indices, steps=32):
        if steps < 2:
            raise ValueError("Integrated gradients requires at least two integration steps")
        model.eval()
        device_inputs = [
            torch.from_numpy(part.x_current[indices]).to(DEVICE),
            torch.from_numpy(part.x_sequence[indices, -config.sequence_steps:]).to(DEVICE),
            torch.from_numpy(part.x_static[indices]).to(DEVICE),
            torch.from_numpy(part.nairas_scaled[indices]).to(DEVICE),
        ]
        baselines = [torch.zeros_like(x) for x in device_inputs]
        gradient_sums = [torch.zeros_like(x) for x in device_inputs]
        alphas = torch.linspace(0, 1, steps, device=DEVICE)
        for step_index, alpha in enumerate(alphas):
            interpolated = [(b + alpha * (x - b)).detach().requires_grad_(True)
                            for x, b in zip(device_inputs, baselines)]
            output = model(*interpolated).sum()
            gradients = torch.autograd.grad(output, interpolated, allow_unused=True)
            trapezoid_weight = 0.5 if step_index in (0, steps - 1) else 1.0
            gradient_sums = [total + trapezoid_weight * (grad if grad is not None else torch.zeros_like(value))
                             for total, grad, value in zip(gradient_sums, gradients, interpolated)]
        return [((x - b) * total / (steps - 1)).detach().cpu().numpy()
                for x, b, total in zip(device_inputs, baselines, gradient_sums)]

    def ig_completeness(model, part, config, indices, attributions):
        inputs = [torch.from_numpy(part.x_current[indices]).to(DEVICE),
                  torch.from_numpy(part.x_sequence[indices, -config.sequence_steps:]).to(DEVICE),
                  torch.from_numpy(part.x_static[indices]).to(DEVICE),
                  torch.from_numpy(part.nairas_scaled[indices]).to(DEVICE)]
        baselines = [torch.zeros_like(x) for x in inputs]
        with torch.no_grad():
            output_difference = (model(*inputs) - model(*baselines)).cpu().numpy()
        attribution_sum = sum(a.reshape(len(indices), -1).sum(axis=1) for a in attributions)
        absolute_error = np.abs(attribution_sum - output_difference)
        return pd.DataFrame({"output_difference": output_difference,
                             "summed_attribution": attribution_sum,
                             "absolute_completeness_error": absolute_error})

    def temporal_ig_table(sequence_attribution, config, cohort):
        magnitude = np.abs(sequence_attribution).mean(axis=0)
        signed = sequence_attribution.mean(axis=0)
        lag_labels = [f"-{5*(config.sequence_steps-1-i)}m"
                      if i < config.sequence_steps-1 else "current"
                      for i in range(config.sequence_steps)]
        rows = []
        for lag_index, lag in enumerate(lag_labels):
            for feature_index, feature in enumerate(TEMPORAL_FEATURES):
                rows.append({"cohort": cohort, "feature": feature, "lag": lag,
                             "mean_absolute_IG": magnitude[lag_index, feature_index],
                             "mean_signed_IG": signed[lag_index, feature_index]})
        return pd.DataFrame(rows)

    if analysis_predictions:
        rng = np.random.default_rng(42)
        overall_indices = rng.choice(len(analysis_data.y_raw), min(256, len(analysis_data.y_raw)), replace=False)
        high_indices = np.flatnonzero(analysis_data.y_raw > 15)
        high_sample = rng.choice(high_indices, min(256, len(high_indices)), replace=False)

        overall_ig = integrated_gradients(
            interpretation_model, analysis_data, interpretation_config, overall_indices, steps=64)
        high_ig = integrated_gradients(
            interpretation_model, analysis_data, interpretation_config, high_sample, steps=64)
        overall_table = temporal_ig_table(overall_ig[1], interpretation_config, "overall")
        high_table = temporal_ig_table(high_ig[1], interpretation_config, "A > 15")
        integrated_gradient_ranking = pd.concat([overall_table, high_table], ignore_index=True)
        display(integrated_gradient_ranking.sort_values(
            ["cohort", "mean_absolute_IG"], ascending=[True, False]).groupby("cohort").head(20))

        completeness = pd.concat([
            ig_completeness(interpretation_model, analysis_data, interpretation_config,
                            overall_indices, overall_ig).assign(cohort="overall"),
            ig_completeness(interpretation_model, analysis_data, interpretation_config,
                            high_sample, high_ig).assign(cohort="A > 15"),
        ], ignore_index=True)
        display(completeness.groupby("cohort").absolute_completeness_error
                            .agg(["mean", "median", "max"]).round(6))

        overall_map = np.abs(overall_ig[1]).mean(axis=0).T
        high_map = np.abs(high_ig[1]).mean(axis=0).T
        lag_labels = [f"-{5*(interpretation_config.sequence_steps-1-i)}m"
                      if i < interpretation_config.sequence_steps-1 else "current"
                      for i in range(interpretation_config.sequence_steps)]
        common_max = max(np.quantile(overall_map, .99), np.quantile(high_map, .99))
        fig, axes = plt.subplots(1, 2, figsize=(20, 12), sharey=True)
        sns.heatmap(overall_map, yticklabels=TEMPORAL_FEATURES, xticklabels=lag_labels,
                    cmap="rocket", vmax=common_max, ax=axes[0])
        sns.heatmap(high_map, yticklabels=TEMPORAL_FEATURES, xticklabels=lag_labels,
                    cmap="rocket", vmax=common_max, ax=axes[1])
        axes[0].set_title(f"{ANALYSIS_PARTITION} overall sample: mean absolute IG")
        axes[1].set_title(f"{ANALYSIS_PARTITION} high-dose sample: mean absolute IG")
        for ax in axes:
            ax.set_xlabel("Time relative to observation")
        axes[0].set_ylabel("Environmental variable")
        finish_figure(f"{ANALYSIS_PARTITION.lower()}_integrated_gradients_heatmaps.png", fig)

        component_rows = []
        for cohort, attrs in [("overall", overall_ig), ("A > 15", high_ig)]:
            for component, values in zip(["current vector", "temporal sequence", "static vector", "NAIRAS V3"], attrs):
                component_rows.append({"cohort": cohort, "component": component,
                                       "mean_absolute_IG_per_element": np.abs(values).mean(),
                                       "mean_total_absolute_IG_per_observation":
                                           np.abs(values).reshape(len(values), -1).sum(axis=1).mean()})
        ig_component_summary = pd.DataFrame(component_rows)
        display(ig_component_summary.round(6))

        feature_ig = (integrated_gradient_ranking.groupby(["cohort", "feature"])
                      .agg(mean_absolute_IG=("mean_absolute_IG", "mean"),
                           mean_signed_IG=("mean_signed_IG", "mean")).reset_index())
        top_features = (feature_ig.sort_values(["cohort", "mean_absolute_IG"], ascending=[True, False])
                        .groupby("cohort").head(12))
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        sns.barplot(data=top_features, x="mean_absolute_IG", y="feature", hue="cohort", ax=axes[0])
        axes[0].set_title("Top temporal features by absolute IG")
        signed_top = top_features.copy()
        sns.barplot(data=signed_top, x="mean_signed_IG", y="feature", hue="cohort", ax=axes[1])
        axes[1].axvline(0, color="black", linewidth=1)
        axes[1].set_title("Direction of top-feature attribution")
        sns.barplot(data=ig_component_summary, x="mean_total_absolute_IG_per_observation",
                    y="component", hue="cohort", ax=axes[2])
        axes[2].set_title("Attribution by input component")
        finish_figure(f"{ANALYSIS_PARTITION.lower()}_integrated_gradients_rankings.png", fig)

        plt.figure(figsize=(8, 4))
        sns.ecdfplot(data=completeness, x="absolute_completeness_error", hue="cohort")
        plt.title("Integrated-gradients completeness error")
        plt.xlabel("|sum(IG) - output difference|")
        finish_figure(f"{ANALYSIS_PARTITION.lower()}_ig_completeness_ecdf.png")
        output_prefix = ANALYSIS_PARTITION.lower()
        integrated_gradient_ranking.to_csv(
            OUTPUT_DIR / f"{output_prefix}_integrated_gradients_variable_lag.csv", index=False)
        completeness.to_csv(
            OUTPUT_DIR / f"{output_prefix}_integrated_gradients_completeness.csv", index=False)
        ig_component_summary.to_csv(
            OUTPUT_DIR / f"{output_prefix}_integrated_gradients_components.csv", index=False)
    """),
    md("""
    ## 12. Feature-group permutation

    Each environmental group is permuted across observations while its complete trajectory remains intact. The RMSE increase measures predictive reliance, not causality.

    **What you need to do:** focus on groups with a clearly positive RMSE increase. A value near zero means the fitted model barely relied on that group for this partition; a negative value can occur from noise or harmful reliance and should not be described as importance.
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

    def permuted_partition(part, indices, seed=42):
        order = np.random.default_rng(seed).permutation(len(part.y_raw))
        changed = copy.copy(part)
        changed.x_sequence = part.x_sequence.copy()
        changed.x_sequence[:, :, indices] = part.x_sequence[order][:, :, indices]
        return changed

    if analysis_predictions:
        baseline = analysis_predictions[interpretation_name]
        baseline_rmse = metric_row(analysis_data.y_raw, baseline)["RMSE"]
        importance_rows = []
        for group, indices in TEMPORAL_GROUPS.items():
            changed = permuted_partition(analysis_data, indices)
            output = predict_standardized(interpretation_model, changed, interpretation_config)
            prediction = reconstruct_armas(output, changed, prep, interpretation_config)
            importance_rows.append({"group": group,
                                    "delta_RMSE": metric_row(changed.y_raw, prediction)["RMSE"] - baseline_rmse})
        permutation_importance = pd.DataFrame(importance_rows).sort_values("delta_RMSE", ascending=False)
        display(permutation_importance)
        permutation_importance.to_csv(
            OUTPUT_DIR / f"{ANALYSIS_PARTITION.lower()}_feature_group_permutation.csv", index=False)
        sns.barplot(data=permutation_importance, x="delta_RMSE", y="group", color="#457b9d")
        plt.axvline(0, color="black")
        plt.title(f"Feature-group permutation on {ANALYSIS_PARTITION} ({analysis_label})")
        finish_figure(f"{ANALYSIS_PARTITION.lower()}_feature_group_permutation.png")
    """),
    md("""
    ## 13. Train/validation/test RMSE for all six partition assignments

    The proposal's primary analysis uses the three cyclic assignments `123`, `231`, and `312`, so each partition serves once in every role. The expanded study below also runs `132`, `213`, and `321`, giving all six possible ordered assignments of P1/P2/P3.

    Every assignment fits fresh imputers, scalers, model weights, and an early-stopping checkpoint using only its designated training and validation partitions. The frozen model configurations are not retuned. Metrics are then calculated separately on the training, validation, and test partitions. This exposes both generalization gaps and sensitivity to which partition is used for validation.

    Six assignments multiplied by all selected models is computationally expensive. They run automatically in `RUN_MODE="full"`; set `RUN_ALL_PARTITION_PERMUTATIONS=True` manually if you intentionally want them in another mode.

    **What you need to do:** run this only after `EVALUATE_P3=True` and after freezing the model list. Use the train/validation/test heatmaps to identify overfitting and partition sensitivity. Use the primary `123`, `231`, `312` summary for direct proposal compliance, and label the six-assignment table as the expanded robustness analysis.
    """),
    code(r"""
    PRIMARY_CYCLIC_ROTATIONS = [("P1", "P2", "P3"), ("P2", "P3", "P1"), ("P3", "P1", "P2")]
    ALL_PARTITION_ASSIGNMENTS = [
        ("P1", "P2", "P3"),  # 123
        ("P1", "P3", "P2"),  # 132
        ("P2", "P1", "P3"),  # 213
        ("P2", "P3", "P1"),  # 231
        ("P3", "P1", "P2"),  # 312
        ("P3", "P2", "P1"),  # 321
    ]
    # USER ACTION: this becomes True automatically only for the final unlocked full run.
    # Do not force it on while you are still using P2 to make modeling decisions.
    RUN_ALL_PARTITION_PERMUTATIONS = RUN_MODE == "full" and EVALUATE_P3

    def assignment_id(train_name, validation_name, test_name):
        return "".join(name[-1] for name in (train_name, validation_name, test_name))

    def evaluate_fitted_model(model, config, partition, prep_r, assignment, role):
        output = predict_standardized(model, partition, config)
        prediction = reconstruct_armas(output, partition, prep_r, config)
        return [{"assignment": assignment, "is_primary_cycle": assignment in {"123", "231", "312"},
                 "train_partition": f"P{assignment[0]}", "validation_partition": f"P{assignment[1]}",
                 "test_partition": f"P{assignment[2]}", "evaluation_role": role,
                 "evaluation_partition": partition.name, "model": config.name, **row}
                for row in evaluate_regimes(partition.y_raw, prediction)]

    def run_partition_assignment(config, train_name, validation_name, test_name):
        assignment = assignment_id(train_name, validation_name, test_name)
        prep_r = fit_preprocessors(raw[train_name])
        scaled = {name: transform(part, prep_r) for name, part in raw.items()}
        model, history = train_model(
            build_model(config), scaled[train_name], scaled[validation_name], prep_r, config)
        rows = []
        for role, partition_name in [("train", train_name), ("validation", validation_name),
                                     ("test", test_name)]:
            rows += evaluate_fitted_model(
                model, config, scaled[partition_name], prep_r, assignment, role)
        for row in rows:
            row["epochs_run"] = len(history)
            row["best_epoch"] = int(history.loc[history.val_RMSE.idxmin(), "epoch"])
        return rows

    def nairas_assignment_rows(train_name, validation_name, test_name):
        assignment = assignment_id(train_name, validation_name, test_name)
        rows = []
        for role, partition_name in [("train", train_name), ("validation", validation_name),
                                     ("test", test_name)]:
            part = raw[partition_name]
            rows += [{"assignment": assignment,
                      "is_primary_cycle": assignment in {"123", "231", "312"},
                      "train_partition": train_name, "validation_partition": validation_name,
                      "test_partition": test_name, "evaluation_role": role,
                      "evaluation_partition": partition_name, "model": "NAIRAS V3",
                      "epochs_run": 0, "best_epoch": np.nan, **row}
                     for row in evaluate_regimes(part.y, part.nairas_v3)]
        return rows

    if RUN_ALL_PARTITION_PERMUTATIONS:
        if QUICK_RUN:
            raise ValueError("Use development/full mode before running partition permutations.")
        rotation_config_pool = VALIDATION_CONFIGS + (TUNING_CONFIGS if RUN_HYPERPARAMETER_STUDY else [])
        config_lookup = {c.name: c for c in rotation_config_pool}
        frozen_configs = [config_lookup[name] for name in FINAL_CONFIG_NAMES]
        permutation_rows = []
        for roles in ALL_PARTITION_ASSIGNMENTS:
            for config in frozen_configs:
                print(f"\nASSIGNMENT {assignment_id(*roles)} | train={roles[0]} | "
                      f"validation={roles[1]} | test={roles[2]} | {config.name}")
                permutation_rows += run_partition_assignment(config, *roles)
            permutation_rows += nairas_assignment_rows(*roles)

        partition_permutation_results = pd.DataFrame(permutation_rows)
        partition_permutation_results.to_csv(
            OUTPUT_DIR / "all_six_partition_assignments_long.csv", index=False)

        overall_rmse = (partition_permutation_results[partition_permutation_results.regime == "all"]
                        .pivot_table(index=["assignment", "model"], columns="evaluation_role",
                                     values="RMSE", aggfunc="first")
                        .rename(columns={"train": "train_RMSE", "validation": "validation_RMSE",
                                         "test": "test_RMSE"}).reset_index())
        overall_rmse["validation_minus_train_RMSE"] = (
            overall_rmse.validation_RMSE - overall_rmse.train_RMSE)
        overall_rmse["test_minus_train_RMSE"] = overall_rmse.test_RMSE - overall_rmse.train_RMSE
        overall_rmse = overall_rmse.sort_values(["assignment", "test_RMSE"])
        print("OVERALL TRAIN / VALIDATION / TEST RMSE FOR ALL SIX ASSIGNMENTS")
        display(overall_rmse.round(4))
        overall_rmse.to_csv(OUTPUT_DIR / "all_six_assignments_train_val_test_rmse.csv", index=False)

        rmse_long = overall_rmse.melt(
            id_vars=["assignment", "model"],
            value_vars=["train_RMSE", "validation_RMSE", "test_RMSE"],
            var_name="split", value_name="RMSE")
        grid = sns.catplot(data=rmse_long, x="split", y="RMSE", hue="model",
                           col="assignment", col_wrap=3, kind="point",
                           height=4, aspect=1.15, sharey=True)
        grid.fig.suptitle("Train, validation, and test RMSE for all partition assignments", y=1.03)
        grid.savefig(OUTPUT_DIR / "all_six_assignments_train_val_test_rmse.png",
                     dpi=180, bbox_inches="tight")
        plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(21, max(5, .6 * overall_rmse.model.nunique())))
        for ax, metric, title in zip(
                axes, ["train_RMSE", "validation_RMSE", "test_RMSE"],
                ["Training RMSE", "Validation RMSE", "Test RMSE"]):
            matrix = overall_rmse.pivot(index="model", columns="assignment", values=metric)
            sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis_r", ax=ax,
                        cbar_kws={"label": "RMSE"})
            ax.set_title(title); ax.set_xlabel("train-validation-test assignment")
        finish_figure("all_six_assignments_rmse_heatmaps.png", fig)

        fig, axes = plt.subplots(1, 2, figsize=(15, max(5, .6 * overall_rmse.model.nunique())))
        for ax, metric, title in zip(
                axes, ["validation_minus_train_RMSE", "test_minus_train_RMSE"],
                ["Validation − train RMSE", "Test − train RMSE"]):
            matrix = overall_rmse.pivot(index="model", columns="assignment", values=metric)
            limit = np.nanmax(np.abs(matrix.to_numpy()))
            sns.heatmap(matrix, annot=True, fmt="+.3f", center=0, vmin=-limit, vmax=limit,
                        cmap="coolwarm", ax=ax)
            ax.set_title(title); ax.set_xlabel("train-validation-test assignment")
        finish_figure("all_six_assignments_generalization_gaps.png", fig)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.boxplot(data=rmse_long, x="split", y="RMSE", hue="model", ax=axes[0])
        axes[0].set_title("RMSE variability across assignments")
        test_only = overall_rmse.copy()
        test_only["test_rank"] = test_only.groupby("assignment").test_RMSE.rank(method="min")
        rank_matrix = test_only.pivot(index="model", columns="assignment", values="test_rank")
        sns.heatmap(rank_matrix, annot=True, fmt=".0f", cmap="YlGn_r", ax=axes[1],
                    cbar_kws={"label": "rank (1 = best)"})
        axes[1].set_title("Test-performance rank stability")
        finish_figure("all_six_assignments_variability_and_ranks.png", fig)

        regime_rmse = (partition_permutation_results
                       .pivot_table(index=["assignment", "model", "regime"],
                                    columns="evaluation_role", values="RMSE", aggfunc="first")
                       .rename(columns={"train": "train_RMSE", "validation": "validation_RMSE",
                                        "test": "test_RMSE"}).reset_index())
        print("TRAIN / VALIDATION / TEST RMSE, INCLUDING <=15 AND >15 REGIMES")
        display(regime_rmse.sort_values(["assignment", "regime", "test_RMSE"]).round(4))
        regime_rmse.to_csv(OUTPUT_DIR / "all_six_assignments_rmse_by_regime.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, .6 * regime_rmse.model.nunique())))
        for ax, regime in zip(axes, ["A <= 15", "A > 15"]):
            matrix = (regime_rmse[regime_rmse.regime == regime]
                      .pivot(index="model", columns="assignment", values="test_RMSE"))
            sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis_r", ax=ax)
            ax.set_title(f"Test RMSE: {regime}")
        finish_figure("all_six_assignments_test_rmse_by_regime.png", fig)

        permutation_summary = (partition_permutation_results
                               .groupby(["model", "evaluation_role", "regime"])
                               [["MAE", "RMSE", "R2", "bias"]].agg(["mean", "std"]))
        display(permutation_summary.round(4))
        permutation_summary.to_csv(OUTPUT_DIR / "all_six_assignments_summary.csv")

        primary_rotation_results = partition_permutation_results[
            partition_permutation_results.is_primary_cycle].copy()
        primary_rotation_results.to_csv(OUTPUT_DIR / "primary_three_rotation_results.csv", index=False)
        primary_summary = (primary_rotation_results
                           .groupby(["model", "evaluation_role", "regime"])
                           [["MAE", "RMSE", "R2", "bias"]].agg(["mean", "std"]))
        primary_summary.to_csv(OUTPUT_DIR / "primary_three_rotation_summary.csv")
    else:
        print("Six-partition study not run. Use RUN_MODE='full' or set "
              "RUN_ALL_PARTITION_PERMUTATIONS=True when you are ready for the expensive final study.")
    """),
    md("""
    ## 14. Automatically answer the research questions

    This section converts result tables into evidence-backed answers you can use as a first draft. Keep the numbers, but rewrite the prose in your own voice and include uncertainty or limitations. A strong result paragraph follows **claim → numerical evidence → interpretation → limitation**.
    """),
    code(r"""
    def result_value(frame, model, regime="all", metric="RMSE"):
        match = frame[(frame.model == model) & (frame.regime == regime)]
        return float(match.iloc[0][metric]) if len(match) else np.nan

    answer_rows = []
    if validation_rows:
        overall_models = ["MLP", "Hybrid CNN-MLP", "Direct Transformer"]
        overall_scores = {name: result_value(validation_results, name) for name in overall_models}
        temporal_best = min(overall_scores, key=overall_scores.get)
        delta_vs_mlp = overall_scores[temporal_best] - overall_scores["MLP"]
        answer_rows.append({
            "research_question": "Does temporal modeling improve over current-time inputs?",
            "evidence": "; ".join(f"{k}: {v:.4f}" for k, v in overall_scores.items()),
            "answer": (f"{temporal_best} had the lowest P2 RMSE; its difference from MLP was "
                       f"{delta_vs_mlp:+.4f} µSv/h. Confirm with the sequence-length ablation before "
                       "attributing the difference specifically to history.")})

        physics_scores = {name: result_value(validation_results, name) for name in
                          ["NAIRAS V3", "Direct Transformer", "Physics-guided Transformer"]}
        physics_best = min(physics_scores, key=physics_scores.get)
        answer_rows.append({
            "research_question": "Does physics-guided residual learning help?",
            "evidence": "; ".join(f"{k}: {v:.4f}" for k, v in physics_scores.items()),
            "answer": (f"{physics_best} had the lowest P2 RMSE. The residual Transformer changed RMSE by "
                       f"{physics_scores['Physics-guided Transformer'] - physics_scores['NAIRAS V3']:+.4f} "
                       "versus uncorrected NAIRAS V3 and by "
                       f"{physics_scores['Physics-guided Transformer'] - physics_scores['Direct Transformer']:+.4f} "
                       "versus direct prediction.")})

        huber_names = [name for name in validation_results.model.unique() if "Huber" in name]
        high_scores = {name: result_value(validation_results, name, "A > 15")
                       for name in ["Physics-guided Transformer"] + huber_names}
        high_best = min(high_scores, key=high_scores.get)
        low_change = (result_value(validation_results, high_best, "A <= 15") -
                      result_value(validation_results, "Physics-guided Transformer", "A <= 15"))
        answer_rows.append({
            "research_question": "Does imbalance-aware loss improve high-dose predictions?",
            "evidence": "; ".join(f"{k}: {v:.4f}" for k, v in high_scores.items()),
            "answer": (f"{high_best} had the lowest high-dose P2 RMSE. Relative to standard residual "
                       f"training, its lower-dose RMSE changed by {low_change:+.4f} µSv/h; use this trade-off "
                       "rather than high-dose performance alone when freezing the final configuration.")})

        convergence_rows = []
        for name, history in histories.items():
            best = history.loc[history.val_RMSE.idxmin()]
            convergence_rows.append({"model": name, "epochs_run": len(history),
                                     "best_epoch": int(best.epoch), "best_P2_RMSE": best.val_RMSE,
                                     "train_loss_at_best": best.train_loss, "val_loss_at_best": best.val_loss,
                                     "total_training_minutes": history.elapsed_seconds.iloc[-1] / 60,
                                     "peak_gpu_memory_MB": history.peak_gpu_memory_MB.max()})
        convergence_summary = pd.DataFrame(convergence_rows).sort_values("best_P2_RMSE")
        display(convergence_summary.round(4))
        convergence_summary.to_csv(OUTPUT_DIR / "convergence_summary.csv", index=False)

    if EVALUATE_P3:
        final_overall = p3_results[p3_results.regime == "all"].sort_values("RMSE")
        best = final_overall.iloc[0]
        answer_rows.append({"research_question": "Which method performed best on the locked P3 test?",
                            "evidence": "; ".join(f"{r.model}: {r.RMSE:.4f}"
                                                  for _, r in final_overall.iterrows()),
                            "answer": f"{best.model} had the lowest locked-test RMSE ({best.RMSE:.4f} µSv/h)."})

    research_answers = pd.DataFrame(answer_rows)
    display(research_answers)
    research_answers.to_csv(OUTPUT_DIR / "research_question_answers.csv", index=False)
    """),
    md("""
    ### Questions you can answer in the report

    - Did one hour of temporal history add value, or was the current step sufficient?
    - Did learning a correction to NAIRAS generalize better than predicting ARMAS from scratch?
    - Which loss best handled the rare `ARMAS > 15` regime, and what did it cost below the threshold?
    - Did a larger/deeper Transformer earn its extra parameters and latency?
    - Were improvements larger than seed variation and bootstrap uncertainty?
    - Which variables and lags were associated with the physics-model discrepancy?
    - Were attention, integrated gradients, and permutation importance consistent with one another?
    - How much did performance change across flight-separated rotations?
    - Where did every model fail: dose range, bias direction, calibration, or distribution shift?

    Do not describe attention or attribution as causal proof. Do not claim generalization from a smoke run. Distinguish validation-selected findings (P2) from the single locked-test confirmation (P3).
    """),
    md("""
    ## 15. Save reproducibility artifacts
    """),
    code(r"""
    overview.to_csv(OUTPUT_DIR / "partition_overview.csv", index=False)
    distribution_audit.to_csv(OUTPUT_DIR / "distribution_audit.csv", index=False)
    missingness.to_csv(OUTPUT_DIR / "missingness_by_feature.csv", index=False)
    partition_shift.to_csv(OUTPUT_DIR / "partition_shift_smd.csv", index=False)
    architecture_table.to_csv(OUTPUT_DIR / "architecture_parameter_counts.csv", index=False)
    lag_corr.to_csv(OUTPUT_DIR / "p1_residual_feature_by_lag_correlations.csv")
    static_direction.to_csv(OUTPUT_DIR / "p1_static_residual_correlations.csv", index=False)
    for name, history in histories.items():
        history.to_csv(OUTPUT_DIR / f"history_{name.lower().replace(' ', '_')}.csv", index=False)
    if EVALUATE_P3:
        for name in FINAL_CONFIG_NAMES:
            model, config = trained[name]
            torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                        "preprocessors": prep, "static_features": STATIC_FEATURES,
                        "temporal_features": TEMPORAL_FEATURES,
                        "train": "P1", "validation": "P2", "test": "P3"},
                       OUTPUT_DIR / f"{name.lower().replace(' ', '_')}.pt")
    print("Artifacts directory:", OUTPUT_DIR.resolve())

    artifact_files = sorted(path for path in OUTPUT_DIR.iterdir() if path.is_file())
    artifact_inventory = pd.DataFrame([
        {"file": path.name, "type": path.suffix.lower(), "size_KB": path.stat().st_size / 1024}
        for path in artifact_files
    ])
    print(f"Saved {len(artifact_inventory)} artifacts. PNG files are report-ready figures; "
          "CSV files contain the exact plotted values.")
    display(artifact_inventory.sort_values(["type", "file"]).round(1))
    """),
    md("""
    ### Which visuals should you place in the report?

    **You do not need to include every generated plot.** Use this selection:

    1. **Data and motivation:** high-dose prevalence plus `nairas_vs_armas_by_partition.png`.
    2. **Training quality:** `learning_curves_loss_phase1_standard.png` and the validation-RMSE curve.
    3. **Main model result:** `p2_validation_performance_dashboard.png`, followed by the locked P3 prediction/residual panel.
    4. **Temporal-value question:** `p2_sequence_window_ablation.png`.
    5. **Complexity question:** `p2_accuracy_complexity_tradeoff.png`.
    6. **High-dose question:** dose-bin/calibration figure and bootstrap table.
    7. **Interpretation question:** integrated-gradients heatmap plus feature-group permutation plot; attention can be supplementary.
    8. **Robustness question:** six-assignment train/validation/test heatmaps and rank-stability plot.

    **Before submitting:** confirm every selected figure states its partition, units, and whether it is validation or locked-test evidence. Do not mix a P2 exploratory figure with a P3 test claim.
    """),
    md("""
    ## Proposal-compliance checklist

    - Physics-guided target is `ARMAS − NAIRAS V3`, reconstructed by adding NAIRAS V3.
    - NAIRAS V3, MLP, hybrid CNN–MLP, direct Transformer, and residual Transformer are compared.
    - P1/P2/P3 cyclic rotations are implemented.
    - All six ordered partition assignments report train, validation, and test RMSE.
    - Preprocessing is training-partition-only in every rotation.
    - Adam, dropout, weight decay, gradient clipping, and validation early stopping are used.
    - Development and full run modes provide 25- and 80-epoch budgets; three epochs are smoke-test only.
    - Train/validation loss, physical-unit validation metrics, learning rate, time, and GPU memory are tracked.
    - Network parameters, storage, approximate FLOPs, latency, and accuracy-efficiency trade-offs are reported.
    - Curated architecture, temporal-window, optimization, regularization, and random-seed studies are available.
    - Standard MSE is compared with validation-selected weighted Huber loss.
    - MAE, RMSE, R², and bias are reported overall, for `A <= 15`, and for `A > 15`.
    - Predictions are compared with NAIRAS V3 on identical held-out observations.
    - Predicted-versus-observed plots and residual distributions are included.
    - Dose-binned error, calibration, and paired bootstrap comparisons are included.
    - Attention maps, integrated gradients, variable-by-lag heatmaps, and feature-group permutation are included.
    - An automatic research-question table connects numerical evidence to report-ready interpretations.
    - Attribution is interpreted as association, not causation.
    """),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "colab": {"provenance": [], "gpuType": "T4", "accelerator": "GPU"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = OUT / "02_pdf_aligned_full_pipeline_colab.ipynb"
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {path}")
