"""Build the direct-ARMAS, NAIRAS-free companion research notebook."""

import copy
import json
from pathlib import Path

import build_pdf_aligned_pipeline as residual_pipeline


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks" / "final"
OUT.mkdir(exist_ok=True)
notebook = copy.deepcopy(residual_pipeline.notebook)
cells = notebook["cells"]


def source(index):
    return "".join(cells[index]["source"])


def set_source(index, text):
    cells[index]["source"] = text.strip().splitlines(True)


# Title and scientific scope.
set_source(0, r"""
# Direct ARMAS prediction without NAIRAS inputs — complete research pipeline

This companion notebook predicts measured **ARMAS effective dose rate directly**. Neither `NAIRASV2` nor `NAIRASV3` is supplied to any neural network during training, validation, testing, attribution, or permutation analysis.

Required comparison:

| Entry | Prediction target | Model inputs |
|---|---|---|
| NAIRAS V3 | none — external benchmark only | supplied NAIRAS V3 estimate |
| Current-time MLP | ARMAS directly | 8 static + current values of 35 environmental variables |
| Hybrid CNN–MLP | ARMAS directly | 12×35 environmental sequence + 8 static variables |
| Direct Temporal Transformer | ARMAS directly | 12×35 sequence + 43 current-time variables |
| Sequence-static Transformer | ARMAS directly | 12×35 sequence + 8 static variables |

`NAIRASV3` is retained only after prediction so its MAE, RMSE, R², bias, calibration, and regime performance can be compared with the learned models on identical observations. `NAIRASV2` and `Vehicle_ID` are never loaded into model tensors.

P1 trains, P2 validates, and P3 is the final test partition. The final study includes all six ordered train/validation/test assignments while preserving `123`, `231`, and `312` as the proposal-aligned primary cycles.
""")

set_source(4, r"""
## 1. Configuration panel

Every learned model in this notebook targets standardized ARMAS directly. Weighted Huber remains an imbalance experiment, but it also optimizes direct ARMAS error. NAIRAS is an external benchmark only.
""")

config = source(5)
config = config.replace(
    "model_kind: str                 # mlp, cnn_hybrid, transformer_direct, transformer_residual",
    "model_kind: str                 # mlp, cnn_hybrid, transformer_direct, transformer_static_direct",
)
config = config.replace('target_mode: str = "direct"     # direct or residual',
                        'target_mode: str = "direct"     # direct only in this notebook')
config = config.replace(
    'ExperimentConfig("Physics-guided Transformer", "transformer_residual", target_mode="residual",\n'
    '                         epochs=EPOCHS, patience=PATIENCE),',
    'ExperimentConfig("Sequence-static Transformer", "transformer_static_direct",\n'
    '                         epochs=EPOCHS, patience=PATIENCE),',
)
config = config.replace('name=f"Physics-guided Huber w={weight:g}"',
                        'name=f"Sequence-static Huber w={weight:g}"')
config = config.replace(
    "# Validation-only imbalance experiments for the proposed residual model.",
    "# Validation-only imbalance experiments for the direct sequence-static model.",
)
config += "\nassert all(c.target_mode == 'direct' for c in STANDARD_CONFIGS + WEIGHTED_HUBER_CONFIGS)\n"
set_source(5, config)

set_source(6, r"""
### Research questions and decision rules

1. **Can environmental and flight variables predict ARMAS without either NAIRAS version as an input?**
2. **Does one hour of temporal history improve over the 43-variable current-time MLP?**
3. **Which direct architecture works best: MLP, CNN–MLP, current-fusion Transformer, or sequence-static Transformer?**
4. **Does weighted Huber improve `ARMAS > 15` accuracy without unacceptable lower-dose degradation?**
5. **Do any learned models beat the external NAIRAS V3 benchmark on identical held-out observations?**
6. **Is additional network complexity justified by RMSE, latency, and robustness?**

Select configurations using P2 overall RMSE, with high-dose RMSE as a prespecified secondary metric. Keep P3 locked until every decision is frozen.
""")

# Separate output directory so the two studies never overwrite each other.
paths = source(3).replace('outputs/physics_guided_residual',
                          'outputs/direct_armas_no_nairas')
paths = paths.replace('ARMAS_project_outputs/physics_guided_residual',
                      'ARMAS_project_outputs/direct_armas_no_nairas')
paths = paths.replace('STUDY_OUTPUT_SUBFOLDER = "physics_guided_residual"',
                      'STUDY_OUTPUT_SUBFOLDER = "direct_armas_no_nairas"')
paths = paths.replace('Physics-guided residual-study outputs:',
                      'Direct-ARMAS no-NAIRAS study outputs:')
paths = paths.replace('the direct-ARMAS study.',
                      'the physics-guided residual study.')
set_source(3, paths)

set_source(21, r"""
## 4. Train-only preprocessing with no NAIRAS preprocessing

Static, temporal, and current-time inputs are median-imputed and standardized using the training partition only. The direct ARMAS target has its own training-only scaler. No scaler is fitted to either NAIRAS version because neither is a model input or training target.
""")

set_source(22, r"""
@dataclass
class Preprocessors:
    static: object
    sequence: object
    current: object
    direct_target: StandardScaler

@dataclass
class ScaledPartition:
    name: str
    x_static: np.ndarray
    x_sequence: np.ndarray
    x_current: np.ndarray
    nairas_raw: np.ndarray          # benchmark only; never passed to a model
    direct_scaled: np.ndarray
    y_raw: np.ndarray
    residual_raw: np.ndarray        # descriptive ARMAS - NAIRAS benchmark discrepancy

def fit_preprocessors(train):
    # USER CHECK: only the designated training partition is fitted here.
    def input_pipeline(values):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(values)
    return Preprocessors(
        static=input_pipeline(train.x_static),
        sequence=input_pipeline(train.x_sequence.reshape(-1, 35)),
        current=input_pipeline(train.x_current),
        direct_target=StandardScaler().fit(train.y.reshape(-1, 1)),
    )

def transform(part, prep):
    n = len(part.y)
    sequence = prep.sequence.transform(part.x_sequence.reshape(-1, 35)).reshape(n, 12, 35)
    return ScaledPartition(
        name=part.name,
        x_static=prep.static.transform(part.x_static).astype(np.float32),
        x_sequence=sequence.astype(np.float32),
        x_current=prep.current.transform(part.x_current).astype(np.float32),
        nairas_raw=part.nairas_v3,
        direct_scaled=prep.direct_target.transform(part.y.reshape(-1, 1)).ravel().astype(np.float32),
        y_raw=part.y,
        residual_raw=part.residual,
    )
""")

# Replace the physics-guided class with a NAIRAS-free sequence/static Transformer.
models = source(24)
old_class = r'''class PhysicsGuidedTransformer(nn.Module):
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
        return (output, maps) if return_attention else output'''
new_class = r'''class SequenceStaticTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temporal = TemporalEncoder(config)
        self.static = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Dropout(config.dropout))
        self.head = nn.Sequential(nn.Linear(config.d_model + 32, 64), nn.GELU(),
                                  nn.Dropout(config.dropout), nn.Linear(64, 1))
    def forward(self, current, sequence, static, unused_benchmark, return_attention=False):
        temporal, maps = self.temporal(sequence, return_attention)
        joined = torch.cat([temporal, self.static(static)], dim=1)
        output = self.head(joined).squeeze(1)
        return (output, maps) if return_attention else output'''
if old_class not in models:
    raise RuntimeError("Could not locate the residual Transformer class in the source notebook")
models = models.replace(old_class, new_class)
models = models.replace('"transformer_residual": PhysicsGuidedTransformer',
                        '"transformer_static_direct": SequenceStaticTransformer')
set_source(24, models)

set_source(25, r"""
# VISUAL AND LEAKAGE AUDIT: 1 means the model is allowed to use that input.
model_design = pd.DataFrame({
    "43 current inputs": [1, 0, 1, 0],
    "12×35 sequence": [0, 1, 1, 1],
    "8 static inputs": [1, 1, 1, 1],
    "NAIRAS V2 input": [0, 0, 0, 0],
    "NAIRAS V3 input": [0, 0, 0, 0],
    "predicts ARMAS": [1, 1, 1, 1],
}, index=["MLP", "Hybrid CNN-MLP", "Direct Transformer", "Sequence-static Transformer"])
assert model_design[["NAIRAS V2 input", "NAIRAS V3 input"]].to_numpy().sum() == 0

# Runtime leakage test: changing the placeholder must not change any prediction.
leakage_rows = []
for config in STANDARD_CONFIGS:
    model = build_model(config).eval()
    current = torch.randn(8, 43)
    sequence = torch.randn(8, config.sequence_steps, 35)
    static = torch.randn(8, 8)
    zeros = torch.zeros(8, 1)
    random_values = torch.randn(8, 1) * 1000
    with torch.no_grad():
        output_zero = model(current, sequence, static, zeros)
        output_random = model(current, sequence, static, random_values)
    maximum_change = float((output_zero - output_random).abs().max())
    leakage_rows.append({"model": config.name,
                         "max_prediction_change_when_placeholder_changes": maximum_change})
    assert maximum_change == 0.0, f"Unexpected benchmark leakage in {config.name}"
leakage_audit = pd.DataFrame(leakage_rows)
display(leakage_audit)

fig, axes = plt.subplots(1, 2, figsize=(17, 5))
sns.heatmap(model_design, annot=True, fmt="d", cmap="Blues", cbar=False,
            linewidths=.5, ax=axes[0])
axes[0].set_title("Direct-model inputs and target (1 = used)")
sns.barplot(data=architecture_table.sort_values("parameters"), x="parameters", y="model",
            color="#457b9d", ax=axes[1])
axes[1].set_xscale("log")
axes[1].set_title("Trainable parameters")
finish_figure("direct_model_design_and_parameter_counts.png", fig)
""")

set_source(28, r"""
## 6. Direct-ARMAS training, reconstruction, and metrics

Every learned model optimizes standardized ARMAS directly. Predictions are inverse-transformed to µSv/h before scoring. The zero placeholder in the loader is never derived from NAIRAS; it only preserves a shared forward signature. NAIRAS V3 enters later as an external benchmark prediction.
""")

training = source(29)
training = training.replace(
    'target = part.residual_scaled if config.target_mode == "residual" else part.direct_scaled',
    'target = part.direct_scaled\n    # Deliberately pass zeros: neither NAIRAS version enters any model tensor.\n'
    '    unused_benchmark = np.zeros((len(part.y_raw), 1), dtype=np.float32)',
)
training = training.replace(
    'torch.from_numpy(part.x_static), torch.from_numpy(part.nairas_scaled),\n'
    '                            torch.from_numpy(target), torch.from_numpy(part.y_raw))',
    'torch.from_numpy(part.x_static), torch.from_numpy(unused_benchmark),\n'
    '                            torch.from_numpy(target), torch.from_numpy(part.y_raw))',
)
old_reconstruct = r'''def reconstruct_armas(standardized_output, part, prep, config):
    if config.target_mode == "residual":
        residual = prep.residual_target.inverse_transform(standardized_output.reshape(-1, 1)).ravel()
        return part.nairas_raw + residual
    return prep.direct_target.inverse_transform(standardized_output.reshape(-1, 1)).ravel()'''
new_reconstruct = r'''def reconstruct_armas(standardized_output, part, prep, config):
    # Direct ARMAS prediction only; no NAIRAS value is added or otherwise used.
    return prep.direct_target.inverse_transform(standardized_output.reshape(-1, 1)).ravel()'''
if old_reconstruct not in training:
    raise RuntimeError("Could not locate reconstruction logic")
training = training.replace(old_reconstruct, new_reconstruct)
set_source(29, training)

normalization = source(33).replace(
    ',\n            "residual target": part.residual_scaled',
    '',
)
set_source(33, normalization)

# Global naming changes for experiment, tuning, selection, and report text.
for i, cell in enumerate(cells):
    text = source(i)
    text = text.replace("Physics-guided Transformer", "Sequence-static Transformer")
    text = text.replace("Physics-guided Huber", "Sequence-static Huber")
    text = text.replace("PGT ", "SST ")
    text = text.replace("PGT_", "SST_")
    text = text.replace('target_mode="residual"', 'target_mode="direct"')
    text = text.replace('"transformer_residual"', '"transformer_static_direct"')
    text = text.replace("physics-guided residual model", "direct sequence-static model")
    text = text.replace("residual Transformer", "sequence-static Transformer")
    text = text.replace("Physics-guided target", "Direct ARMAS target")
    set_source(i, text)

# Fix tuning model-kind inherited from the last standard configuration; names are now SST.
tuning = source(38)
tuning = tuning.replace('base = STANDARD_CONFIGS[-1]',
                        'base = STANDARD_CONFIGS[-1]  # direct ARMAS, sequence + static, no NAIRAS')
set_source(38, tuning)

# Validation-only selection uses the direct sequence-static reference.
selection = source(43)
selection = selection.replace(
    '# Prespecified validation rule: minimize high-dose RMSE among weighted-Huber models\n'
    '# whose lower-dose RMSE is no more than 5% worse than the standard residual model.',
    '# Prespecified validation rule for direct ARMAS training: minimize high-dose RMSE\n'
    '# while limiting lower-dose degradation to 5% versus standard sequence-static training.',
)
set_source(43, selection)

# Attention and integrated gradients receive a zero placeholder, never NAIRAS.
attention = source(49).replace(
    'nairas = torch.from_numpy(part.nairas_scaled[indices]).to(DEVICE)',
    'nairas = torch.zeros((len(indices), 1), dtype=torch.float32, device=DEVICE)  # unused placeholder',
)
set_source(49, attention)

ig = source(51)
ig = ig.replace(
    'torch.from_numpy(part.nairas_scaled[indices]).to(DEVICE),',
    'torch.zeros((len(indices), 1), dtype=torch.float32, device=DEVICE),',
)
ig = ig.replace(
    'torch.from_numpy(part.nairas_scaled[indices]).to(DEVICE)]',
    'torch.zeros((len(indices), 1), dtype=torch.float32, device=DEVICE)]',
)
ig = ig.replace('["current vector", "temporal sequence", "static vector", "NAIRAS V3"]',
                '["current vector", "temporal sequence", "static vector", "unused zero placeholder"]')
set_source(51, ig)

# Direct-study explanations and conclusions.
set_source(50, source(50).replace(
    "For the physics-guided model, IG explains the model's **predicted standardized correction**, not the whole NAIRAS value.",
    "IG explains the model's **standardized direct ARMAS prediction**. The fourth input is an unused zero placeholder, so its attribution must remain zero.",
).replace("moved the predicted correction upward or downward",
          "moved the direct ARMAS prediction upward or downward"))

answers = source(57)
answers = answers.replace("Does physics-guided residual learning help?",
                          "Can direct ARMAS learning without NAIRAS inputs beat the external benchmark?")
answers = answers.replace("The sequence-static Transformer changed RMSE by",
                          "The NAIRAS-free sequence-static Transformer changed RMSE by")
answers = answers.replace("versus uncorrected NAIRAS V3", "versus external NAIRAS V3")
set_source(57, answers)

set_source(58, r"""
### Questions you can answer in the report

- Can a neural model predict ARMAS directly without either NAIRAS version as an input?
- Which direct architecture has the lowest validation and locked-test RMSE?
- Does one hour of history improve over the current-time MLP and current-only ablation?
- Does the sequence-static Transformer beat the external NAIRAS V3 benchmark?
- Which weighted-Huber configuration improves `ARMAS > 15`, and what is the lower-dose cost?
- Does greater depth/width justify its parameters, FLOPs, latency, and instability?
- Which variables and lags influence direct ARMAS predictions?
- Are attention, integrated gradients, and permutation importance mutually consistent?
- How much does performance change across all six partition assignments?

Do not claim that NAIRAS was an input: it is used only as a post-prediction benchmark. Do not describe attribution as causal proof. Distinguish P2 selection evidence from P3 locked-test confirmation.
""")

set_source(62, r"""
## Direct-study compliance checklist

- Every learned model predicts ARMAS directly.
- `Vehicle_ID`, `NAIRASV2`, and `NAIRASV3` are absent from every model input tensor.
- NAIRAS V3 is retained only as an external benchmark on identical observations.
- MLP, hybrid CNN–MLP, direct current-fusion Transformer, and sequence-static Transformer are compared.
- Standard MSE is compared with validation-selected weighted Huber.
- Missing-value handling and normalization use training-partition statistics only.
- Overall, `A <= 15`, and `A > 15` MAE, RMSE, R², bias, and skill versus NAIRAS are reported.
- P3 is used only for final testing after P1/P2 development.
- The final analysis includes all six ordered partition assignments and primary cyclic summaries.
- Loss curves, complexity, calibration, error anatomy, attention, integrated gradients, permutation, and robustness visuals are included.
- Attribution is interpreted as association, not causation.
""")

# Final safety checks against accidental NAIRAS model input or residual targets.
all_code = "\n".join(source(i) for i, cell in enumerate(cells) if cell["cell_type"] == "code")
for forbidden in ("part.nairas_scaled", "prep.nairas", "part.residual_scaled", "prep.residual_target",
                  'target_mode="residual"',
                  '"transformer_residual"', "PhysicsGuidedTransformer"):
    if forbidden in all_code:
        raise RuntimeError(f"Forbidden direct-study model dependency remains: {forbidden}")

notebook["metadata"]["colab"]["name"] = "03_direct_armas_no_nairas_full_pipeline_colab.ipynb"
path = OUT / "03_direct_armas_no_nairas_full_pipeline_colab.ipynb"
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {path}")
