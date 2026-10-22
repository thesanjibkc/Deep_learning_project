"""Execute the deep-learning experiments specified in the ARMAS proposal.

Experimental contract
---------------------
* P1 fits every imputer, scaler, and neural-network weight.
* P2 is the validation partition used for early stopping and model selection.
* P3 is loaded only for final testing after each checkpoint is selected on P2.

The source JSON arrays have current time at index 0.  ``load_partition``
reverses them once, so all tensors below run from t-55 to t minutes.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from armas_5m_full_study import (
    HIGH_DOSE_THRESHOLD,
    STATIC_FEATURES,
    TEMPORAL_FEATURES,
    current_features,
    load_partition,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "proposal_models_5m"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
BATCH_SIZE = 512
MAX_EPOCHS = 24
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ArrayScaler:
    """Median imputation followed by z-score scaling, fitted on P1 only."""

    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values, axes=(0,)):
        median = np.nanmedian(values, axis=axes, keepdims=True)
        filled = np.where(np.isfinite(values), values, median)
        mean = np.mean(filled, axis=axes, keepdims=True)
        std = np.std(filled, axis=axes, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(median.astype(np.float32), mean.astype(np.float32), std.astype(np.float32))

    def transform(self, values):
        filled = np.where(np.isfinite(values), values, self.median)
        return ((filled - self.mean) / self.std).astype(np.float32)


class SinusoidalPosition(nn.Module):
    """Fixed positional encoding so attention can distinguish the 12 lags."""

    def __init__(self, length, width):
        super().__init__()
        position = torch.arange(length).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        encoding = torch.zeros(length, width)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, : x.shape[1]]


class CurrentJMLP(nn.Module):
    """Joint MLP (JMLP): eight static + 35 current environmental inputs."""

    def __init__(self, n_inputs=43):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 1),
        )

    def forward(self, sequence, current):
        return self.network(current).squeeze(1)


class HybridCNNMLP(nn.Module):
    """One-dimensional temporal CNN joined to a static/current MLP branch."""

    def __init__(self, current_width=43):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(35, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.current = nn.Sequential(nn.Linear(current_width, 64), nn.GELU(), nn.Dropout(0.10))
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 1))

    def forward(self, sequence, current):
        temporal = self.cnn(sequence.transpose(1, 2))
        return self.head(torch.cat([temporal, self.current(current)], dim=1)).squeeze(1)


class TemporalTransformer(nn.Module):
    """Compact Transformer with a learned summary token and current-input branch."""

    def __init__(self, current_width=43, d_model=48, heads=4, layers=2):
        super().__init__()
        self.projection = nn.Linear(35, d_model)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = SinusoidalPosition(13, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=128, dropout=0.12,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, norm=nn.LayerNorm(d_model))
        self.current = nn.Sequential(
            nn.Linear(current_width, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(96, 48), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + 48, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 1)
        )
        nn.init.normal_(self.summary_token, std=0.02)

    def forward(self, sequence, current):
        tokens = self.projection(sequence)
        summary = self.summary_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(self.position(torch.cat([summary, tokens], dim=1)))
        joined = torch.cat([encoded[:, 0], self.current(current)], dim=1)
        return self.head(joined).squeeze(1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_loaders(train_data, validation_data):
    def dataset(data):
        sequence, current, target, high_mask = data
        return TensorDataset(
            torch.from_numpy(sequence),
            torch.from_numpy(current),
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(high_mask.astype(np.float32)),
        )

    train_dataset = dataset(train_data)
    validation_dataset = dataset(validation_data)
    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(validation_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0),
    )


def epoch_loss(model, loader, optimizer=None, high_weight=1.0):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for sequence, current, target, high in loader:
        sequence, current, target, high = [x.to(DEVICE) for x in (sequence, current, target, high)]
        with torch.set_grad_enabled(training):
            prediction = model(sequence, current)
            element_loss = nn.functional.huber_loss(prediction, target, delta=1.0, reduction="none")
            weights = 1.0 + (high_weight - 1.0) * high
            loss = (element_loss * weights).sum() / weights.sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item() * len(target)
        count += len(target)
    return total / count


def fit_model(name, model, loaders, high_weight=1.0):
    """AdamW training with cosine scheduling, gradient clipping, and early stopping."""

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    best_state, best_loss, stale = None, float("inf"), 0
    rows = []
    started = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = epoch_loss(model, loaders[0], optimizer, high_weight)
        validation_loss = epoch_loss(model, loaders[1], None, high_weight)
        rows.append({"Model": name, "Epoch": epoch, "Train loss": train_loss,
                     "P2 validation loss": validation_loss, "Learning rate": scheduler.get_last_lr()[0]})
        print(f"  {name:38s} epoch {epoch:02d}  train={train_loss:.4f}  val={validation_loss:.4f}", flush=True)
        if validation_loss < best_loss - 1e-4:
            best_loss, best_state, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        scheduler.step()
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    elapsed = time.time() - started
    return model, rows, elapsed


@torch.no_grad()
def predict_scaled(model, sequence, current):
    model.eval()
    outputs = []
    dataset = TensorDataset(torch.from_numpy(sequence), torch.from_numpy(current))
    for seq, cur in DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False):
        outputs.append(model(seq.to(DEVICE), cur.to(DEVICE)).cpu().numpy())
    return np.concatenate(outputs)


def metric_rows(y, prediction, model_name):
    rows = []
    for regime, mask in [
        ("All", np.ones(len(y), dtype=bool)),
        ("ARMAS <= 15", y <= HIGH_DOSE_THRESHOLD),
        ("ARMAS > 15", y > HIGH_DOSE_THRESHOLD),
    ]:
        rows.append({
            "Model": model_name, "Regime": regime, "N": int(mask.sum()),
            "MAE": mean_absolute_error(y[mask], prediction[mask]),
            "RMSE": np.sqrt(mean_squared_error(y[mask], prediction[mask])),
            "R2": r2_score(y[mask], prediction[mask]),
            "Bias": np.mean(prediction[mask] - y[mask]),
        })
    return rows


def plot_results(validation_metrics, test_metrics, test_predictions, p3, history):
    plt.style.use("seaborn-v0_8-whitegrid")
    overall = test_metrics[test_metrics.Regime == "All"].sort_values("RMSE")
    high = test_metrics[test_metrics.Regime == "ARMAS > 15"].set_index("Model").loc[overall.Model]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].barh(overall.Model, overall.RMSE, color="#2e5eaa")
    axes[0].invert_yaxis(); axes[0].set_xlabel("P3 test RMSE (µSv/h)"); axes[0].set_title("Overall")
    axes[1].barh(high.index, high.RMSE, color="#9b2226")
    axes[1].invert_yaxis(); axes[1].set_xlabel("P3 test RMSE (µSv/h)"); axes[1].set_title("High dose: ARMAS > 15")
    fig.tight_layout(); fig.savefig(OUT / "01_proposal_model_rmse.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for name, group in history.groupby("Model"):
        ax.plot(group.Epoch, group["P2 validation loss"], marker="o", ms=2.5, label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("P2 validation weighted Huber loss")
    ax.set_title("Learning curves (early stopping uses P2)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "02_learning_curves.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    validation_overall = validation_metrics[validation_metrics.Regime == "All"].sort_values("RMSE")
    best_name = validation_overall.iloc[0].Model
    best = test_predictions[best_name]
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(p3.y), min(5000, len(p3.y)), replace=False)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.hexbin(p3.y[chosen], best[chosen], gridsize=48, mincnt=1, cmap="viridis")
    limits = [0, np.nanpercentile(np.r_[p3.y, best], 99.7)]
    ax.plot(limits, limits, "r--", lw=1.4, label="Ideal")
    ax.set(xlabel="Observed ARMAS (µSv/h)", ylabel="Predicted (µSv/h)",
           title=f"P3 test: P2-selected model ({best_name})")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "03_best_observed_vs_predicted.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def occlusion_study(model, sequence, current, y, nairas, target_mean, target_std):
    """Measure associations by replacing one standardized group with its P1 mean (zero)."""

    base = nairas + (predict_scaled(model, sequence, current) * target_std + target_mean)
    base_rmse = np.sqrt(mean_squared_error(y, base))
    feature_rows = []
    for j, name in enumerate(TEMPORAL_FEATURES):
        changed = sequence.copy(); changed[:, :, j] = 0.0
        pred = nairas + (predict_scaled(model, changed, current) * target_std + target_mean)
        feature_rows.append({"Feature": name, "RMSE increase": np.sqrt(mean_squared_error(y, pred)) - base_rmse})
    lag_rows = []
    for step in range(12):
        changed = sequence.copy(); changed[:, step, :] = 0.0
        pred = nairas + (predict_scaled(model, changed, current) * target_std + target_mean)
        lag_rows.append({"Lag minutes": 55 - 5 * step, "RMSE increase": np.sqrt(mean_squared_error(y, pred)) - base_rmse})
    features = pd.DataFrame(feature_rows).sort_values("RMSE increase", ascending=False)
    lags = pd.DataFrame(lag_rows)
    features.to_csv(OUT / "p2_residual_transformer_feature_occlusion.csv", index=False)
    lags.to_csv(OUT / "p2_residual_transformer_lag_occlusion.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    top = features.head(12).sort_values("RMSE increase")
    axes[0].barh(top.Feature, top["RMSE increase"], color="#2e5eaa")
    axes[0].set_xlabel("Increase in P2 RMSE (µSv/h)"); axes[0].set_title("Temporal-feature occlusion")
    axes[1].plot(lags["Lag minutes"], lags["RMSE increase"], marker="o", color="#9b2226")
    axes[1].invert_xaxis(); axes[1].set_xlabel("Minutes before current observation")
    axes[1].set_ylabel("Increase in P2 RMSE (µSv/h)"); axes[1].set_title("Time-step occlusion")
    fig.tight_layout(); fig.savefig(OUT / "04_transformer_occlusion.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def main():
    seed_everything()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    print(f"Device: {DEVICE}; loading P1 train, P2 validation, and P3 test ...", flush=True)
    p1, p2, p3 = load_partition("P1"), load_partition("P2"), load_partition("P3")

    seq_scaler = ArrayScaler.fit(p1.sequence, axes=(0, 1))
    seq1, seq2, seq3 = (
        seq_scaler.transform(p1.sequence),
        seq_scaler.transform(p2.sequence),
        seq_scaler.transform(p3.sequence),
    )
    cur1_raw, cur2_raw, cur3_raw = (
        current_features(p1).astype(np.float32),
        current_features(p2).astype(np.float32),
        current_features(p3).astype(np.float32),
    )
    current_scaler = ArrayScaler.fit(cur1_raw, axes=(0,))
    cur1, cur2, cur3 = (
        current_scaler.transform(cur1_raw),
        current_scaler.transform(cur2_raw),
        current_scaler.transform(cur3_raw),
    )
    nairas_scaler = ArrayScaler.fit(p1.nairas[:, None].astype(np.float32), axes=(0,))
    n1, n2, n3 = (
        nairas_scaler.transform(p1.nairas[:, None]),
        nairas_scaler.transform(p2.nairas[:, None]),
        nairas_scaler.transform(p3.nairas[:, None]),
    )
    residual_cur1 = np.column_stack([cur1, n1])
    residual_cur2 = np.column_stack([cur2, n2])
    residual_cur3 = np.column_stack([cur3, n3])

    high1, high2 = p1.y > HIGH_DOSE_THRESHOLD, p2.y > HIGH_DOSE_THRESHOLD
    direct_mean, direct_std = float(p1.y.mean()), float(p1.y.std())
    residual1, residual2 = p1.y - p1.nairas, p2.y - p2.nairas
    residual_mean, residual_std = float(residual1.mean()), float(residual1.std())
    direct_target1 = ((p1.y - direct_mean) / direct_std).astype(np.float32)
    direct_target2 = ((p2.y - direct_mean) / direct_std).astype(np.float32)
    residual_target1 = ((residual1 - residual_mean) / residual_std).astype(np.float32)
    residual_target2 = ((residual2 - residual_mean) / residual_std).astype(np.float32)

    experiments = [
        ("Current-time JMLP direct", CurrentJMLP(43), cur1, cur2, cur3, direct_target1, direct_target2, "direct", 1.0),
        ("Hybrid CNN-MLP direct", HybridCNNMLP(43), cur1, cur2, cur3, direct_target1, direct_target2, "direct", 1.0),
        ("Direct Temporal Transformer", TemporalTransformer(43), cur1, cur2, cur3, direct_target1, direct_target2, "direct", 1.0),
        ("Residual Transformer", TemporalTransformer(44), residual_cur1, residual_cur2, residual_cur3, residual_target1, residual_target2, "residual", 1.0),
        ("Residual Transformer + weighted Huber (w=2)", TemporalTransformer(44), residual_cur1, residual_cur2, residual_cur3, residual_target1, residual_target2, "residual", 2.0),
        ("Residual Transformer + weighted Huber (w=4)", TemporalTransformer(44), residual_cur1, residual_cur2, residual_cur3, residual_target1, residual_target2, "residual", 4.0),
    ]

    validation_metrics = metric_rows(p2.y, p2.nairas, "NAIRAS V3")
    test_metrics = metric_rows(p3.y, p3.nairas, "NAIRAS V3")
    histories = []
    validation_predictions = {"NAIRAS V3": p2.nairas.copy()}
    test_predictions = {"NAIRAS V3": p3.nairas.copy()}
    model_rows = []
    residual_model = None
    for number, (name, model, cur_train, cur_valid, cur_test, target_train, target_valid,
                 target_kind, weight) in enumerate(experiments):
        seed_everything(SEED + number)
        print(f"\nTraining {name} ({parameter_count(model):,} parameters) ...", flush=True)
        loaders = make_loaders(
            (seq1, cur_train, target_train, high1),
            (seq2, cur_valid, target_valid, high2),
        )
        model, rows, seconds = fit_model(name, model, loaders, weight)
        validation_scaled = predict_scaled(model, seq2, cur_valid)
        test_scaled = predict_scaled(model, seq3, cur_test)
        if target_kind == "direct":
            validation_prediction = validation_scaled * direct_std + direct_mean
            test_prediction = test_scaled * direct_std + direct_mean
        else:
            validation_prediction = p2.nairas + validation_scaled * residual_std + residual_mean
            test_prediction = p3.nairas + test_scaled * residual_std + residual_mean
        validation_predictions[name] = validation_prediction
        test_predictions[name] = test_prediction
        validation_metrics.extend(metric_rows(p2.y, validation_prediction, name))
        test_metrics.extend(metric_rows(p3.y, test_prediction, name))
        histories.extend(rows)
        model_rows.append({"Model": name, "Trainable parameters": parameter_count(model),
                           "Epochs run": len(rows), "Training seconds": seconds,
                           "Target": target_kind, "High-dose weight": weight})
        torch.save(model.state_dict(), OUT / (f"model_{number+1}.pt"))
        if name == "Residual Transformer":
            residual_model = model

    validation_metrics_frame = pd.DataFrame(validation_metrics)
    test_metrics_frame = pd.DataFrame(test_metrics)
    history_frame = pd.DataFrame(histories)
    validation_metrics_frame.to_csv(OUT / "p2_validation_metrics.csv", index=False)
    test_metrics_frame.to_csv(OUT / "p3_test_metrics.csv", index=False)
    history_frame.to_csv(OUT / "training_history.csv", index=False)
    pd.DataFrame(model_rows).to_csv(OUT / "model_parameter_counts.csv", index=False)
    pd.DataFrame({"ARMAS": p2.y, **validation_predictions}).to_csv(OUT / "p2_validation_predictions.csv", index=False)
    pd.DataFrame({"ARMAS": p3.y, **test_predictions}).to_csv(OUT / "p3_test_predictions.csv", index=False)
    plot_results(validation_metrics_frame, test_metrics_frame, test_predictions, p3, history_frame)

    print("\nRunning proposal temporal-interpretation experiment ...", flush=True)
    occlusion_study(residual_model, seq2, residual_cur2, p2.y, p2.nairas, residual_mean, residual_std)

    run = {
        "seed": SEED, "device": str(DEVICE), "torch_version": torch.__version__,
        "partition_roles": {"P1": "train", "P2": "validation", "P3": "test"},
        "p1_rows": len(p1.y), "p2_rows": len(p2.y), "p3_rows": len(p3.y),
        "temporal_shape": [12, 35], "source_order_correction": "reversed index 0=current to oldest->current",
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    for label, frame in [("P2 validation", validation_metrics_frame), ("P3 test", test_metrics_frame)]:
        overall = frame[frame.Regime == "All"].sort_values("RMSE")
        high = frame[frame.Regime == "ARMAS > 15"].sort_values("RMSE")
        print(f"\n{label} overall\n", overall[["Model", "RMSE", "MAE", "R2", "Bias"]].to_string(index=False))
        print(f"\n{label} high dose\n", high[["Model", "RMSE", "MAE", "R2", "Bias"]].to_string(index=False))
    print(f"\nCompleted P1/P2/P3 train-validation-test run. Outputs: {OUT}")


if __name__ == "__main__":
    main()
