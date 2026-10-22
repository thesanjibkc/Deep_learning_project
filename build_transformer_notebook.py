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


cells = [
    md(r"""
    # ARMAS Transformer experiment lab (Colab GPU)

    This is a self-contained extension to the planned MLP/CNN study. It tests whether self-attention helps model the preceding hour of 5-minute environmental measurements.

    | Model | Inputs | Question |
    |---|---|---|
    | Temporal Transformer | 12 × 35 environmental sequence | Can attention summarize the hour without flight variables? |
    | Hybrid Transformer–MLP | sequence + 8 static flight/geomagnetic variables | Does combining temporal context and flight conditions work best? |
    | NAIRAS V3 | physics estimate, benchmark only | Does the learned model improve on the external physics baseline? |

    Experimental discipline:

    - P1 trains parameters and preprocessing.
    - P2 selects Transformer settings.
    - P3 is the final test partition and is evaluated after one configuration is frozen.
    - `Vehicle_ID`, `NAIRASV2`, and `NAIRASV3` are never predictors.
    """),
    md("""
    ## 0. Colab setup

    Choose **Runtime → Change runtime type → T4 GPU**. Keep `QUICK_RUN=True` while checking the code. A quick run is only a smoke test, not a scientific result.
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
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    sns.set_theme(style="whitegrid")

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
    OUTPUT_DIR = Path("outputs/transformer")

    # COLAB PATHS: comment the two lines above and uncomment these four lines.
    # from google.colab import drive
    # drive.mount('/content/drive')
    # DATA_DIR = Path('/content/drive/MyDrive/ARMASdatasets_COMPLETE')
    # OUTPUT_DIR = Path('/content/drive/MyDrive/ARMAS_project_outputs/transformer')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FILES = {p: DATA_DIR / f"ARMAS_TIMESERIES_5M_20132023_{p}.json" for p in ("P1", "P2", "P3")}
    assert all(path.exists() for path in FILES.values()), "Fix DATA_DIR before continuing."
    FILES
    """),
    md("""
    ## 1. Experiment controls

    These are the main settings to change. Start with the baseline. Later, compare one change at a time so you know what caused the result.

    - `sequence_steps`: 3, 6, or 12 steps correspond to 15, 30, or 60 minutes.
    - `d_model`: embedding width. Larger is more flexible but slower.
    - `n_heads`: number of attention heads; it must divide `d_model`.
    - `n_layers`: number of Transformer encoder blocks.
    - `pooling`: `current` uses the final/current token; `mean` averages all tokens.
    - `loss`: `huber` is less sensitive to extreme residuals; `mse` emphasizes them.
    - `high_dose_weight`: values above 1 emphasize training rows where ARMAS > 15.
    """),
    code(r"""
    @dataclass(frozen=True)
    class ExperimentConfig:
        name: str = "baseline_transformer"
        model_kind: str = "temporal"       # "temporal" or "hybrid"
        sequence_steps: int = 12            # choose 3, 6, or 12
        d_model: int = 64
        n_heads: int = 4
        n_layers: int = 2
        feedforward_dim: int = 128
        dropout: float = 0.10
        pooling: str = "current"            # "current" or "mean"
        loss: str = "huber"                 # "huber" or "mse"
        high_dose_weight: float = 1.0
        learning_rate: float = 3e-4
        weight_decay: float = 1e-4
        batch_size: int = 512
        epochs: int = 40
        patience: int = 8
        seed: int = 42

    QUICK_RUN = True
    BASE_CONFIG = ExperimentConfig(
        epochs=3 if QUICK_RUN else 40,
        patience=2 if QUICK_RUN else 8,
    )
    BASE_CONFIG
    """),
    md("""
    Before running: predict which will matter most—more history, more encoder layers, mean pooling, or adding the static branch. Write down the prediction so the experiment tests an idea instead of only searching for the lowest number.
    """),
    md("""
    ## 2. Load the three partitions

    The source arrays store current time at index 0. The loader reverses them once, producing standard chronological tensors from oldest to current with shape `[rows, 12, 35]`.
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

    def load_partition(name, path):
        static, sequences, target, benchmark = [], [], [], []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                source = np.asarray([row[f] for f in TEMPORAL_FEATURES], dtype=np.float32).T
                if source.shape != (12, 35):
                    raise ValueError(f"{name} line {line_number}: got {source.shape}, expected (12, 35)")
                static.append([row[f] for f in STATIC_FEATURES])
                sequences.append(source[::-1].copy())  # oldest -> current
                target.append(row["ARMAS"])
                benchmark.append(row["NAIRASV3"])
        part = RawPartition(
            name=name,
            x_static=np.asarray(static, dtype=np.float32),
            x_sequence=np.asarray(sequences, dtype=np.float32),
            y=np.asarray(target, dtype=np.float32),
            nairas_v3=np.asarray(benchmark, dtype=np.float32),
        )
        if not all(np.isfinite(x).all() for x in (part.x_static, part.x_sequence, part.y, part.nairas_v3)):
            raise ValueError(f"{name}: non-finite values found")
        return part

    raw = {name: load_partition(name, path) for name, path in FILES.items()}
    pd.DataFrame([
        {"partition": name, "rows": len(part.y), "static_shape": str(part.x_static.shape),
         "sequence_shape": str(part.x_sequence.shape), "high_dose_n": int((part.y > 15).sum())}
        for name, part in raw.items()
    ])
    """),
    md("""
    ## 3. Standardize using P1 only

    Transformers are sensitive to feature scale. The 35 environmental features have very different physical units, so each feature is standardized. Fitting scalers on P2 or P3 would leak information.
    """),
    code(r"""
    @dataclass
    class ScaledPartition:
        name: str
        x_static: np.ndarray
        x_sequence: np.ndarray
        y_scaled: np.ndarray
        y_raw: np.ndarray
        nairas_v3: np.ndarray

    @dataclass
    class Preprocessors:
        static: StandardScaler
        sequence: StandardScaler
        target: StandardScaler

    def fit_preprocessors(train):
        return Preprocessors(
            static=StandardScaler().fit(train.x_static),
            sequence=StandardScaler().fit(train.x_sequence.reshape(-1, 35)),
            target=StandardScaler().fit(train.y.reshape(-1, 1)),
        )

    def transform(part, prep):
        n = len(part.y)
        seq = prep.sequence.transform(part.x_sequence.reshape(-1, 35)).reshape(n, 12, 35)
        return ScaledPartition(
            name=part.name,
            x_static=prep.static.transform(part.x_static).astype(np.float32),
            x_sequence=seq.astype(np.float32),
            y_scaled=prep.target.transform(part.y.reshape(-1, 1)).ravel().astype(np.float32),
            y_raw=part.y,
            nairas_v3=part.nairas_v3,
        )

    prep = fit_preprocessors(raw["P1"])
    data = {name: transform(part, prep) for name, part in raw.items()}
    print("P1 standardized sequence mean:", data["P1"].x_sequence.mean(axis=(0, 1)).mean().round(6))
    print("P1 standardized sequence SD:", data["P1"].x_sequence.std(axis=(0, 1)).mean().round(6))
    """),
    md("""
    ## 4. Understand the Transformer

    For each time step, a linear layer projects 35 measurements into a `d_model`-dimensional token. A learned position embedding tells attention where each measurement lies in the hour. Encoder blocks let each time step interact with every other step. The pooled representation is converted to the dose-rate prediction.

    The hybrid version concatenates that temporal representation with a small MLP representation of the eight current flight/geomagnetic variables.
    """),
    code(r"""
    class TemporalEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            if config.d_model % config.n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.pooling = config.pooling
            self.input_projection = nn.Linear(35, config.d_model)
            self.position = nn.Parameter(torch.zeros(1, 12, config.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
            self.norm = nn.LayerNorm(config.d_model)

        def forward(self, sequence):
            steps = sequence.shape[1]
            tokens = self.input_projection(sequence) + self.position[:, -steps:, :]
            encoded = self.norm(self.encoder(tokens))
            if self.pooling == "current":
                return encoded[:, -1, :]
            if self.pooling == "mean":
                return encoded.mean(dim=1)
            raise ValueError("pooling must be 'current' or 'mean'")

    class TemporalTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.temporal = TemporalEncoder(config)
            self.head = nn.Sequential(
                nn.Linear(config.d_model, 64), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(64, 1)
            )
        def forward(self, sequence, static):
            return self.head(self.temporal(sequence)).squeeze(1)

    class HybridTransformerMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.temporal = TemporalEncoder(config)
            self.static = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Dropout(config.dropout))
            self.head = nn.Sequential(
                nn.Linear(config.d_model + 32, 64), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(64, 1)
            )
        def forward(self, sequence, static):
            joined = torch.cat([self.temporal(sequence), self.static(static)], dim=1)
            return self.head(joined).squeeze(1)

    def build_model(config):
        if config.model_kind == "temporal":
            return TemporalTransformer(config)
        if config.model_kind == "hybrid":
            return HybridTransformerMLP(config)
        raise ValueError("model_kind must be 'temporal' or 'hybrid'")

    sample_model = build_model(BASE_CONFIG)
    print(sample_model)
    print(f"Trainable parameters: {sum(p.numel() for p in sample_model.parameters() if p.requires_grad):,}")
    """),
    md("""
    ## 5. Training and evaluation functions

    Early stopping watches P2 RMSE in physical units (µSv/h). The best validation checkpoint is restored automatically.
    """),
    code(r"""
    def make_loader(part, config, shuffle):
        # Always keep the most recent steps; tensors are ordered oldest -> current.
        sequence = part.x_sequence[:, -config.sequence_steps:, :]
        dataset = TensorDataset(
            torch.from_numpy(sequence), torch.from_numpy(part.x_static),
            torch.from_numpy(part.y_scaled), torch.from_numpy(part.y_raw),
        )
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    def predict_scaled(model, part, config):
        model.eval()
        predictions = []
        with torch.no_grad():
            for sequence, static, _, _ in make_loader(part, config, shuffle=False):
                predictions.append(model(sequence.to(DEVICE), static.to(DEVICE)).cpu().numpy())
        return np.concatenate(predictions)

    def metric_row(y, prediction):
        return {
            "n": len(y),
            "MAE": mean_absolute_error(y, prediction),
            "RMSE": mean_squared_error(y, prediction) ** 0.5,
            "R2": r2_score(y, prediction),
        }

    def evaluate_regimes(y, prediction):
        rows = []
        for regime, mask in {"all": np.ones(len(y), bool), "ARMAS > 15": y > 15}.items():
            rows.append({"regime": regime, **metric_row(y[mask], prediction[mask])})
        return rows

    def train_model(model, train, validation, prep, config):
        seed_everything(config.seed)
        model = model.to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        best_state, best_rmse, stale = None, float("inf"), 0
        history = []

        for epoch in range(1, config.epochs + 1):
            model.train()
            batch_losses = []
            for sequence, static, y_scaled, y_raw in make_loader(train, config, shuffle=True):
                sequence, static = sequence.to(DEVICE), static.to(DEVICE)
                y_scaled, y_raw = y_scaled.to(DEVICE), y_raw.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(sequence, static)
                if config.loss == "huber":
                    per_row_loss = F.smooth_l1_loss(prediction, y_scaled, reduction="none")
                elif config.loss == "mse":
                    per_row_loss = (prediction - y_scaled).square()
                else:
                    raise ValueError("loss must be 'huber' or 'mse'")
                weights = torch.where(y_raw > 15, config.high_dose_weight, 1.0)
                loss = (per_row_loss * weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(loss.item())

            val_scaled = predict_scaled(model, validation, config)
            val_prediction = prep.target.inverse_transform(val_scaled.reshape(-1, 1)).ravel()
            val_rmse = metric_row(validation.y_raw, val_prediction)["RMSE"]
            history.append({"epoch": epoch, "train_loss": np.mean(batch_losses), "val_RMSE": val_rmse})
            print(f"epoch {epoch:03d} | train loss {np.mean(batch_losses):.4f} | P2 RMSE {val_rmse:.4f}")

            if val_rmse < best_rmse - 1e-4:
                best_rmse, stale = val_rmse, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= config.patience:
                    print("Early stopping")
                    break

        model.load_state_dict(best_state)
        return model, pd.DataFrame(history)
    """),
    md("""
    ## 6. Smoke test one configuration on P1 → P2

    This cell does not evaluate P3. First confirm that loss decreases and P2 predictions are sensible.
    """),
    code(r"""
    model = build_model(BASE_CONFIG)
    start = time.time()
    model, history = train_model(model, data["P1"], data["P2"], prep, BASE_CONFIG)
    p2_scaled = predict_scaled(model, data["P2"], BASE_CONFIG)
    p2_prediction = prep.target.inverse_transform(p2_scaled.reshape(-1, 1)).ravel()

    smoke_results = pd.DataFrame([
        {"model": BASE_CONFIG.name, "partition": "P2", **row}
        for row in evaluate_regimes(data["P2"].y_raw, p2_prediction)
    ])
    print(f"Runtime: {(time.time() - start) / 60:.1f} minutes")
    display(smoke_results.round(4))
    """),
    code(r"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o")
    axes[0].set(title="Training loss", xlabel="epoch")
    axes[1].plot(history["epoch"], history["val_RMSE"], marker="o")
    axes[1].set(title="P2 validation RMSE", xlabel="epoch")
    axes[2].hexbin(data["P2"].y_raw, p2_prediction, gridsize=45, mincnt=1, cmap="viridis")
    limit = max(data["P2"].y_raw.max(), p2_prediction.max())
    axes[2].plot([0, limit], [0, limit], "r--")
    axes[2].set(title="P2 predicted vs measured", xlabel="ARMAS", ylabel="prediction")
    plt.tight_layout()
    """),
    md("""
    Check before experimenting:

    - Does training loss decrease?
    - Does P2 RMSE stop improving, justifying early stopping?
    - Are predictions compressed toward the mean at high dose rates?
    - Is the high-dose RMSE much worse than the overall RMSE?
    """),
    md("""
    ## 7. Controlled experiment grid—P1/P2 only

    The example list changes one or two ideas at a time. Add or remove configurations. Avoid a huge blind grid: each experiment should answer a question.
    """),
    code(r"""
    EXPERIMENTS = [
        replace(BASE_CONFIG, name="temporal_60m_current", model_kind="temporal"),
        replace(BASE_CONFIG, name="temporal_60m_mean", model_kind="temporal", pooling="mean"),
        replace(BASE_CONFIG, name="hybrid_60m_current", model_kind="hybrid"),
        replace(BASE_CONFIG, name="hybrid_30m_current", model_kind="hybrid", sequence_steps=6),
        replace(BASE_CONFIG, name="hybrid_highdose_x2", model_kind="hybrid", high_dose_weight=2.0),
        replace(BASE_CONFIG, name="hybrid_wider", model_kind="hybrid", d_model=128,
                n_heads=8, feedforward_dim=256),
    ]

    RUN_EXPERIMENT_GRID = False  # Change to True when the smoke test works.
    experiment_rows, experiment_objects = [], {}

    if RUN_EXPERIMENT_GRID:
        for config in EXPERIMENTS:
            print(f"\n===== {config.name} =====")
            candidate = build_model(config)
            candidate, candidate_history = train_model(candidate, data["P1"], data["P2"], prep, config)
            pred_scaled = predict_scaled(candidate, data["P2"], config)
            prediction = prep.target.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            for row in evaluate_regimes(data["P2"].y_raw, prediction):
                experiment_rows.append({**asdict(config), **row})
            experiment_objects[config.name] = {
                "config": config, "model": candidate, "history": candidate_history,
                "p2_prediction": prediction,
            }
        validation_results = pd.DataFrame(experiment_rows)
        display(validation_results.query("regime == 'all'").sort_values("RMSE").round(4))
        validation_results.to_csv(OUTPUT_DIR / "transformer_validation_experiments.csv", index=False)
    else:
        print("Set RUN_EXPERIMENT_GRID=True after the smoke test.")
    """),
    md("""
    Interpretation prompts:

    - `temporal` vs `hybrid`: how much do flight and geomagnetic variables add?
    - 30 vs 60 minutes: does older history help?
    - current vs mean pooling: is the latest token sufficient?
    - high-dose weighting: does high-dose RMSE improve at an acceptable cost to overall RMSE?
    - wider model: does extra capacity improve validation performance or only slow training?
    """),
    md("""
    ## 8. Freeze one configuration, then unlock P3

    Select the configuration using P2 only. Write its name into `FINAL_EXPERIMENT_NAME`, then set `EVALUATE_P3=True`. Do not repeatedly change the configuration after viewing P3.
    """),
    code(r"""
    FINAL_EXPERIMENT_NAME = "hybrid_60m_current"
    EVALUATE_P3 = True

    if EVALUATE_P3:
        if FINAL_EXPERIMENT_NAME not in experiment_objects:
            raise ValueError("Run the experiment grid and choose an existing experiment name.")
        final_object = experiment_objects[FINAL_EXPERIMENT_NAME]
        final_config = final_object["config"]
        final_model = final_object["model"]
        p3_scaled = predict_scaled(final_model, data["P3"], final_config)
        p3_prediction = prep.target.inverse_transform(p3_scaled.reshape(-1, 1)).ravel()

        p3_rows = [
            {"model": FINAL_EXPERIMENT_NAME, "partition": "P3", **row}
            for row in evaluate_regimes(data["P3"].y_raw, p3_prediction)
        ]
        p3_rows += [
            {"model": "NAIRAS V3", "partition": "P3", **row}
            for row in evaluate_regimes(data["P3"].y_raw, data["P3"].nairas_v3)
        ]
        p3_results = pd.DataFrame(p3_rows)
        display(p3_results.round(4))
        p3_results.to_csv(OUTPUT_DIR / "transformer_final_p3_metrics.csv", index=False)
    else:
        print("P3 test evaluation is disabled. Select the model on P2 first.")
    """),
    md("""
    ## 9. Residual diagnosis on the P3 final test

    Residual = prediction − measurement. Values below zero mean underprediction, which is especially important in the high-radiation regime.
    """),
    code(r"""
    if EVALUATE_P3:
        residual = p3_prediction - data["P3"].y_raw
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].scatter(data["P3"].y_raw, residual, s=7, alpha=0.2)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].axvline(15, color="black", linestyle=":")
        axes[0].set(xlabel="Measured ARMAS", ylabel="Residual", title="Residual versus dose rate")
        sns.histplot(residual, bins=70, ax=axes[1])
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set(title="P3 residual distribution")
        plt.tight_layout()
    """),
    md("""
    ## 10. Optional final cyclic rotations

    The proposal requests three rotations. Run these only after the architecture and configuration are frozen. Every rotation creates new scalers and a new model.
    """),
    code(r"""
    ROTATIONS = [("P1", "P2", "P3"), ("P2", "P3", "P1"), ("P3", "P1", "P2")]

    def run_rotation(config, train_name, validation_name, test_name):
        prep_r = fit_preprocessors(raw[train_name])
        scaled_r = {name: transform(part, prep_r) for name, part in raw.items()}
        candidate = build_model(config)
        candidate, _ = train_model(candidate, scaled_r[train_name], scaled_r[validation_name], prep_r, config)
        pred_scaled = predict_scaled(candidate, scaled_r[test_name], config)
        prediction = prep_r.target.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        rows = []
        for row in evaluate_regimes(scaled_r[test_name].y_raw, prediction):
            rows.append({"train": train_name, "validation": validation_name, "test": test_name,
                         "model": config.name, **row})
        for row in evaluate_regimes(scaled_r[test_name].y_raw, scaled_r[test_name].nairas_v3):
            rows.append({"train": train_name, "validation": validation_name, "test": test_name,
                         "model": "NAIRAS V3", **row})
        return rows

    RUN_ROTATIONS = False
    if RUN_ROTATIONS:
        if QUICK_RUN:
            raise ValueError("Set QUICK_RUN=False before report-quality rotations.")
        frozen = next(c for c in EXPERIMENTS if c.name == FINAL_EXPERIMENT_NAME)
        rotation_rows = []
        for roles in ROTATIONS:
            print("\nROTATION", roles)
            rotation_rows.extend(run_rotation(frozen, *roles))
        rotation_results = pd.DataFrame(rotation_rows)
        rotation_results.to_csv(OUTPUT_DIR / "transformer_rotation_results.csv", index=False)
        display(rotation_results.round(4))
        summary = (rotation_results.groupby(["model", "regime"])[["MAE", "RMSE", "R2"]]
                   .agg(["mean", "std"]).round(4))
        display(summary)
    else:
        print("RUN_ROTATIONS is False.")
    """),
    md("""
    ## 11. Save the chosen model

    Save to Drive so a Colab disconnect does not erase the checkpoint. The checkpoint records features, preprocessing, and experiment settings needed for reproduction.
    """),
    code(r"""
    if EVALUATE_P3:
        torch.save({
            "model_name": FINAL_EXPERIMENT_NAME,
            "state_dict": final_model.state_dict(),
            "config": asdict(final_config),
            "static_features": STATIC_FEATURES,
            "temporal_features": TEMPORAL_FEATURES,
            "source_sequence_order": "current_to_past",
            "model_sequence_order": "past_to_current",
            "preprocessors": prep,
            "train_partition": "P1",
            "validation_partition": "P2",
            "test_partition": "P3",
        }, OUTPUT_DIR / f"{FINAL_EXPERIMENT_NAME}.pt")
        print("Saved checkpoint to", OUTPUT_DIR)
    """),
    md("""
    ## What belongs in the report?

    Treat the Transformer as an extension or ablation because the approved proposal centers on MLP, CNN, and hybrid CNN–MLP. Report it only if the comparison is fair: identical partitions, preprocessing rules, target, metrics, and preferably similar tuning effort. A negative result is still useful—12 steps may be too short for attention to outperform a compact CNN.
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

path = OUT / "03_transformer_experiments_colab.ipynb"
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {path}")
