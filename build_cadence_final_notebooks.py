"""Build matched 5-minute and 1-hour ARMAS Colab research notebooks.

The source of truth is the PDF-aligned final pipeline.  This script changes only
cadence-dependent data paths, tensor lengths, lag labels, temporal ablations,
and output directories, then builds a small cross-cadence results notebook.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "notebooks" / "final" / "02_pdf_aligned_full_pipeline_colab.ipynb"
FINAL_DIR = ROOT / "notebooks" / "final"


def cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def set_cell_text(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def replace_required(text: str, old: str, new: str, count: int = 1) -> str:
    found = text.count(old)
    if found != count:
        raise RuntimeError(f"Expected {count} occurrence(s) of {old!r}; found {found}")
    return text.replace(old, new)


def cadence_label(minutes: int, steps: int) -> str:
    total = minutes * steps
    if total % (24 * 60) == 0:
        return f"{total // (24 * 60)}-day"
    if total % 60 == 0:
        return f"{total // 60}-hour"
    return f"{total}-minute"


def build_cadence_notebook(
    template: dict,
    *,
    cadence_code: str,
    cadence_minutes: int,
    sequence_length: int,
    output_slug: str,
    filename: str,
) -> Path:
    notebook = copy.deepcopy(template)
    cells = notebook["cells"]
    half_steps = sequence_length // 2
    full_window = cadence_label(cadence_minutes, sequence_length)
    half_window = cadence_label(cadence_minutes, half_steps)

    title = f"# ARMAS physics-guided temporal modeling — {cadence_code} comparison pipeline\n\n"
    title += (
        "This is a cadence-specific copy of the current PDF-aligned project pipeline. "
        f"It reads `ARMAS_TIMESERIES_{cadence_code}_20132023_P1/P2/P3.json`. "
        f"Each observation contains {sequence_length} chronological temporal steps at "
        f"{cadence_minutes}-minute spacing after the loader reverses the source arrays.\n\n"
    )
    title += f"**Comparison tag:** `{cadence_code}`. Outputs are isolated in `{output_slug}`.\n\n"
    base_intro = cell_text(cells[0])
    base_intro = base_intro[base_intro.index("Required comparison:") :]
    base_intro = base_intro.replace("12×35 sequence", f"{sequence_length}×35 sequence")
    set_cell_text(cells[0], title + base_intro)

    paths = f'''# Cadence-specific paths and shape controls.
CADENCE_CODE = "{cadence_code}"
CADENCE_MINUTES = {cadence_minutes}
SEQUENCE_LENGTH = {sequence_length}
STUDY_OUTPUT_SUBFOLDER = "{output_slug}"

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
    DATA_DIR = Path('/content/drive/MyDrive/ARMASdatasets_COMPLETE')
    OUTPUT_DIR = Path('/content/drive/MyDrive/ARMAS_project_outputs') / STUDY_OUTPUT_SUBFOLDER
else:
    DATA_DIR = Path("ARMASdatasets_COMPLETE-20261006T110557Z-1-001/ARMASdatasets_COMPLETE")
    OUTPUT_DIR = Path("outputs") / STUDY_OUTPUT_SUBFOLDER

if OUTPUT_DIR.name != STUDY_OUTPUT_SUBFOLDER:
    raise ValueError(f"OUTPUT_DIR must end with {{STUDY_OUTPUT_SUBFOLDER!r}}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FILES = {{p: DATA_DIR / f"ARMAS_TIMESERIES_{{CADENCE_CODE}}_20132023_{{p}}.json"
         for p in ("P1", "P2", "P3")}}
missing_files = [str(path) for path in FILES.values() if not path.exists()]
if missing_files:
    raise FileNotFoundError("Fix DATA_DIR; missing:\\n" + "\\n".join(missing_files))

def cadence_lag_labels(steps):
    labels = []
    for index in range(steps):
        lag_minutes = CADENCE_MINUTES * (steps - 1 - index)
        if lag_minutes == 0:
            labels.append("current")
        elif lag_minutes % 60 == 0:
            labels.append(f"-{{lag_minutes // 60}}h")
        else:
            labels.append(f"-{{lag_minutes}}m")
    return labels

print(f"Cadence: {{CADENCE_CODE}} | sequence shape: {{SEQUENCE_LENGTH}}×35")
print("Outputs:", OUTPUT_DIR)
'''
    set_cell_text(cells[3], paths)

    config = cell_text(cells[5])
    config = replace_required(config, "sequence_steps: int = 12", f"sequence_steps: int = {sequence_length}")
    set_cell_text(cells[5], config)

    history_minutes = cadence_minutes * (sequence_length - 1)
    history_text = (
        "## 2. Load and validate the data\n\n"
        "The source sequences store current time at index 0. The loader reverses each sequence once so tensors are chronological, "
        f"from {history_minutes // 60} hours before the observation to current time. "
        f"The {cadence_code} files must contain exactly {sequence_length} values for every temporal variable.\n"
    )
    set_cell_text(cells[8], history_text)

    loader = cell_text(cells[9])
    loader = replace_required(loader, "if source.shape != (12, 35):", "if source.shape != (SEQUENCE_LENGTH, 35):")
    loader = replace_required(
        loader,
        'raise ValueError(f"{name} line {line_number}: expected (12, 35), got {source.shape}")',
        'raise ValueError(f"{name} line {line_number}: expected ({SEQUENCE_LENGTH}, 35), got {source.shape}")',
    )
    set_cell_text(cells[9], loader)

    lag_corr = cell_text(cells[16])
    lag_corr = replace_required(lag_corr, "matrix = np.empty((35, 12), dtype=float)",
                                "matrix = np.empty((35, SEQUENCE_LENGTH), dtype=float)")
    lag_corr = replace_required(lag_corr, "for lag in range(12):", "for lag in range(SEQUENCE_LENGTH):")
    lag_corr = replace_required(
        lag_corr,
        'columns=[f"-{55 - 5*i}m" if i < 11 else "current" for i in range(12)])',
        "columns=cadence_lag_labels(SEQUENCE_LENGTH))",
    )
    set_cell_text(cells[16], lag_corr)

    preprocessing = cell_text(cells[22])
    preprocessing = replace_required(
        preprocessing,
        ".reshape(n, 12, 35)",
        ".reshape(n, SEQUENCE_LENGTH, 35)",
    )
    set_cell_text(cells[22], preprocessing)

    models = cell_text(cells[24])
    models = replace_required(
        models,
        "self.position = nn.Parameter(torch.zeros(1, 12, config.d_model))",
        "self.position = nn.Parameter(torch.zeros(1, SEQUENCE_LENGTH, config.d_model))",
    )
    set_cell_text(cells[24], models)

    tuning_note = cell_text(cells[37])
    tuning_note = tuning_note.replace(
        "The 1-, 6-, and 12-step variants directly test whether an hour of history adds information beyond the current measurement.",
        f"The 1-, {half_steps}-, and {sequence_length}-step variants test whether the {full_window} record adds information beyond the current measurement.",
    )
    set_cell_text(cells[37], tuning_note)

    tuning = cell_text(cells[38])
    tuning = replace_required(
        tuning,
        'replace(base, name="PGT 30-minute window", sequence_steps=6, epochs=TUNING_EPOCHS)',
        f'replace(base, name="PGT {half_window} window", sequence_steps={half_steps}, epochs=TUNING_EPOCHS)',
    )
    set_cell_text(cells[38], tuning)

    validation = cell_text(cells[31])
    validation = replace_required(
        validation,
        '    validation_results.to_csv(OUTPUT_DIR / "p2_validation_results.csv", index=False)',
        '''    validation_results.to_csv(OUTPUT_DIR / "p2_validation_results.csv", index=False)
    p2_prediction_table = pd.DataFrame({
        "row_index": np.arange(len(data["P2"].y_raw)),
        "ARMAS": data["P2"].y_raw,
        "NAIRAS V3": data["P2"].nairas_raw,
        **validation_predictions,
    })
    p2_prediction_table.to_csv(OUTPUT_DIR / "p2_predictions_by_row.csv", index=False)''',
    )
    set_cell_text(cells[31], validation)

    locked_test = cell_text(cells[43])
    locked_test = replace_required(
        locked_test,
        '    p3_results.to_csv(OUTPUT_DIR / "p3_locked_test_results.csv", index=False)',
        '''    p3_results.to_csv(OUTPUT_DIR / "p3_locked_test_results.csv", index=False)
    p3_prediction_table = pd.DataFrame({
        "row_index": np.arange(len(data["P3"].y_raw)),
        "ARMAS": data["P3"].y_raw,
        **test_predictions,
    })
    p3_prediction_table.to_csv(OUTPUT_DIR / "p3_predictions_by_row.csv", index=False)''',
    )
    set_cell_text(cells[43], locked_test)

    attention = cell_text(cells[49])
    old_attention_labels = '''labels = [f"-{5*(attention.shape[0]-1-i)}m" if i < attention.shape[0]-1 else "current"
              for i in range(attention.shape[0])]'''
    attention = replace_required(
        attention,
        old_attention_labels,
        "labels = cadence_lag_labels(attention.shape[0])",
    )
    set_cell_text(cells[49], attention)

    gradients = cell_text(cells[51])
    old_config_labels = '''lag_labels = [f"-{5*(config.sequence_steps-1-i)}m"
                  if i < config.sequence_steps-1 else "current"
                  for i in range(config.sequence_steps)]'''
    gradients = replace_required(
        gradients,
        old_config_labels,
        "lag_labels = cadence_lag_labels(config.sequence_steps)",
    )
    old_interpretation_labels = '''lag_labels = [f"-{5*(interpretation_config.sequence_steps-1-i)}m"
                  if i < interpretation_config.sequence_steps-1 else "current"
                  for i in range(interpretation_config.sequence_steps)]'''
    gradients = replace_required(
        gradients,
        old_interpretation_labels,
        "lag_labels = cadence_lag_labels(interpretation_config.sequence_steps)",
    )
    set_cell_text(cells[51], gradients)

    questions = cell_text(cells[56]).replace(
        "Did one hour of temporal history add value, or was the current step sufficient?",
        f"Did the {full_window} temporal record add value over the current step for the {cadence_code} data?",
    )
    questions = questions.replace(
        "Do not claim generalization from a smoke run.",
        "Do not claim generalization from a smoke run. Compare cadences only after both notebooks use the same run mode and frozen configuration protocol.",
    )
    set_cell_text(cells[56], questions)

    audit = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Cadence-comparison controls\n",
            "\n",
            f"- Dataset family: `{cadence_code}`\n",
            f"- Temporal tensor: `{sequence_length} × 35`\n",
            f"- Sampling interval: `{cadence_minutes} minutes`\n",
            f"- Full temporal record: `{full_window}`\n",
            "- Identical ARMAS targets, static variables, model families, split protocol, metrics, and radiation regimes are retained.\n",
            "- Use the same `RUN_MODE`, validation-selected configurations, and P3 gate in both cadence notebooks before comparing results.\n",
        ],
    }
    cells.insert(4, audit)

    notebook.setdefault("metadata", {}).setdefault("colab", {})["name"] = filename
    notebook["metadata"]["armas_cadence"] = {
        "code": cadence_code,
        "minutes": cadence_minutes,
        "sequence_length": sequence_length,
        "output_subfolder": output_slug,
        "source_notebook": SOURCE.name,
    }
    destination = FINAL_DIR / filename
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def build_comparison_notebook() -> Path:
    cells = [
        markdown_cell(
            "# ARMAS 5M versus 1H final-results comparison\n\n"
            "Run the matched 5M and 1H final pipelines first. This notebook reads their saved artifacts, checks that comparable protocols were used, and creates report-ready cadence comparisons without retraining either model.\n"
        ),
        code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "sns.set_theme(style='whitegrid', context='notebook')\n"
            "try:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "    ROOT = Path('/content/drive/MyDrive/ARMAS_project_outputs')\n"
            "except ImportError:\n"
            "    ROOT = Path('outputs')\n\n"
            "OUTPUT_DIRS = {\n"
            "    '5M': ROOT / 'physics_guided_residual_5m',\n"
            "    '1H': ROOT / 'physics_guided_residual_1h',\n"
            "}\n"
            "COMPARISON_DIR = ROOT / 'cadence_comparison_5m_vs_1h'\n"
            "COMPARISON_DIR.mkdir(parents=True, exist_ok=True)\n"
            "for cadence, folder in OUTPUT_DIRS.items():\n"
            "    if not folder.exists():\n"
            "        raise FileNotFoundError(f'Run the {cadence} notebook first; missing {folder}')\n"
            "print('Comparison outputs:', COMPARISON_DIR)\n"
        ),
        markdown_cell(
            "## 1. Load matching artifacts\n\n"
            "P2 remains the validation comparison. P3 is the final test and is shown when both runs have completed test evaluation. The six-assignment comparison is likewise optional until both full runs finish.\n"
        ),
        code_cell(
            "def load_pair(filename, required=True):\n"
            "    frames = []\n"
            "    for cadence, folder in OUTPUT_DIRS.items():\n"
            "        path = folder / filename\n"
            "        if not path.exists():\n"
            "            if required:\n"
            "                raise FileNotFoundError(f'Missing {path}')\n"
            "            return None\n"
            "        frames.append(pd.read_csv(path).assign(cadence=cadence))\n"
            "    return pd.concat(frames, ignore_index=True)\n\n"
            "p2 = load_pair('p2_validation_results.csv')\n"
            "p3 = load_pair('p3_locked_test_results.csv', required=False)\n"
            "rotations = load_pair('all_six_partition_assignments_long.csv', required=False)\n"
            "complexity = load_pair('network_complexity.csv', required=False)\n"
            "print('P2 rows:', len(p2))\n"
            "print('P3 available:', p3 is not None)\n"
            "print('Six assignments available:', rotations is not None)\n"
            "display(p2.head())\n"
        ),
        markdown_cell(
            "## 2. Validation comparison\n\n"
            "Use P2 to compare cadences and choose configurations. Keep P3 locked until the choices for both cadence experiments are frozen.\n"
        ),
        code_cell(
            "required = {'model', 'regime', 'RMSE', 'cadence'}\n"
            "missing = required - set(p2.columns)\n"
            "if missing:\n"
            "    raise KeyError(f'P2 tables are missing columns: {sorted(missing)}')\n"
            "p2_table = p2.pivot_table(index=['model', 'regime'], columns='cadence', values='RMSE')\n"
            "if {'5M', '1H'}.issubset(p2_table.columns):\n"
            "    p2_table['1H_minus_5M_RMSE'] = p2_table['1H'] - p2_table['5M']\n"
            "display(p2_table.sort_values(['regime', '5M']).round(4))\n\n"
            "plot = p2[p2.regime.isin(['all', 'A > 15'])].copy()\n"
            "g = sns.catplot(data=plot, x='RMSE', y='model', hue='cadence', col='regime',\n"
            "                kind='bar', sharex=False, height=5, aspect=1.05)\n"
            "g.fig.suptitle('P2 validation: 5M versus 1H', y=1.03)\n"
            "plt.show()\n"
            "p2.to_csv(COMPARISON_DIR / 'p2_cadence_comparison_long.csv', index=False)\n"
            "p2_table.to_csv(COMPARISON_DIR / 'p2_cadence_comparison_rmse.csv')\n"
        ),
        markdown_cell(
            "### Physics and architecture contrasts on validation data\n\n"
            "These contrasts isolate three questions: whether temporal context beats the current-time MLP, whether learning a correction to NAIRAS beats direct prediction, and whether the longer 1H context beats the short 5M context. Positive gain means the first named method reduced RMSE.\n"
        ),
        code_cell(
            "def rmse_lookup(frame, cadence, model, regime):\n"
            "    match = frame[(frame.cadence == cadence) & (frame.model == model) & (frame.regime == regime)]\n"
            "    return float(match.RMSE.iloc[0]) if len(match) else np.nan\n\n"
            "contrast_rows = []\n"
            "for cadence in ['5M', '1H']:\n"
            "    for regime in ['all', 'A <= 15', 'A > 15']:\n"
            "        mlp = rmse_lookup(p2, cadence, 'MLP', regime)\n"
            "        cnn = rmse_lookup(p2, cadence, 'Hybrid CNN-MLP', regime)\n"
            "        direct = rmse_lookup(p2, cadence, 'Direct Transformer', regime)\n"
            "        physics = rmse_lookup(p2, cadence, 'Physics-guided Transformer', regime)\n"
            "        nairas = rmse_lookup(p2, cadence, 'NAIRAS V3', regime)\n"
            "        contrast_rows += [\n"
            "            {'cadence': cadence, 'regime': regime, 'question': 'CNN temporal gain over MLP', 'RMSE_gain': mlp - cnn},\n"
            "            {'cadence': cadence, 'regime': regime, 'question': 'Direct Transformer gain over MLP', 'RMSE_gain': mlp - direct},\n"
            "            {'cadence': cadence, 'regime': regime, 'question': 'Physics-guided gain over direct Transformer', 'RMSE_gain': direct - physics},\n"
            "            {'cadence': cadence, 'regime': regime, 'question': 'Physics-guided gain over NAIRAS V3', 'RMSE_gain': nairas - physics},\n"
            "        ]\n"
            "physics_contrasts = pd.DataFrame(contrast_rows)\n"
            "display(physics_contrasts.pivot_table(index=['question', 'regime'], columns='cadence', values='RMSE_gain').round(4))\n"
            "sns.catplot(data=physics_contrasts[physics_contrasts.regime.isin(['all', 'A > 15'])],\n"
            "            x='RMSE_gain', y='question', hue='cadence', col='regime',\n"
            "            kind='bar', sharex=False, height=5, aspect=1.1)\n"
            "plt.axvline(0, color='black', linewidth=.8)\n"
            "plt.show()\n"
            "physics_contrasts.to_csv(COMPARISON_DIR / 'p2_physics_architecture_contrasts.csv', index=False)\n"
        ),
        markdown_cell(
            "### Feature-by-lag physics exploration\n\n"
            "The saved P1 Spearman matrices describe association between each environmental input and `ARMAS − NAIRAS V3`. Comparing their strongest lags tests whether discrepancies are associated with fast particle/solar-wind variation or longer geomagnetic and solar context. This is exploratory association, not causal evidence.\n"
        ),
        code_cell(
            "lag_frames = []\n"
            "for cadence, folder in OUTPUT_DIRS.items():\n"
            "    path = folder / 'p1_residual_feature_by_lag_correlations.csv'\n"
            "    if not path.exists():\n"
            "        raise FileNotFoundError(f'Missing {path}; finish the exploration section in both notebooks.')\n"
            "    wide = pd.read_csv(path, index_col=0)\n"
            "    long = wide.rename_axis('feature').reset_index().melt(\n"
            "        id_vars='feature', var_name='lag', value_name='Spearman')\n"
            "    long['cadence'] = cadence\n"
            "    long['absolute_Spearman'] = long.Spearman.abs()\n"
            "    lag_frames.append(long)\n"
            "lag_comparison = pd.concat(lag_frames, ignore_index=True)\n"
            "peak_lags = (lag_comparison.sort_values('absolute_Spearman', ascending=False)\n"
            "             .groupby(['cadence', 'feature'], as_index=False).first())\n"
            "display(peak_lags.sort_values(['cadence', 'absolute_Spearman'], ascending=[True, False])\n"
            "        .groupby('cadence').head(15).round(4))\n"
            "top_features = (peak_lags.groupby('feature').absolute_Spearman.max()\n"
            "                .nlargest(15).index)\n"
            "sns.catplot(data=peak_lags[peak_lags.feature.isin(top_features)],\n"
            "            x='absolute_Spearman', y='feature', hue='cadence', kind='bar',\n"
            "            height=7, aspect=1.25)\n"
            "plt.show()\n"
            "lag_comparison.to_csv(COMPARISON_DIR / 'p1_feature_lag_correlations_by_cadence.csv', index=False)\n"
            "peak_lags.to_csv(COMPARISON_DIR / 'p1_peak_feature_lags_by_cadence.csv', index=False)\n"
        ),
        markdown_cell(
            "## 3. Locked P3 and radiation-regime comparison\n\n"
            "This section runs only after both cadence notebooks have evaluated P3. Report overall and high-dose results together; a cadence can improve average error while worsening `ARMAS > 15`.\n"
        ),
        code_cell(
            "if p3 is None:\n"
            "    print('P3 comparison is locked or incomplete. Finish both cadence runs first.')\n"
            "else:\n"
            "    p3_table = p3.pivot_table(index=['model', 'regime'], columns='cadence', values='RMSE')\n"
            "    p3_table['1H_minus_5M_RMSE'] = p3_table['1H'] - p3_table['5M']\n"
            "    display(p3_table.round(4))\n"
            "    sns.catplot(data=p3[p3.regime.isin(['all', 'A <= 15', 'A > 15'])],\n"
            "                x='RMSE', y='model', hue='cadence', col='regime',\n"
            "                kind='bar', sharex=False, height=5, aspect=.9)\n"
            "    plt.show()\n"
            "    p3.to_csv(COMPARISON_DIR / 'p3_cadence_comparison_long.csv', index=False)\n"
            "    p3_table.to_csv(COMPARISON_DIR / 'p3_cadence_comparison_rmse.csv')\n"
        ),
        markdown_cell(
            "### Paired cadence uncertainty on identical P3 rows\n\n"
            "When both prediction tables exist, this paired bootstrap compares 1H and 5M errors on the same ARMAS observations. A negative `1H_minus_5M_RMSE` favors 1H; a positive value favors 5M. Because rows within flights can be correlated, use partition rotations as the stronger robustness result.\n"
        ),
        code_cell(
            "prediction_paths = {c: folder / 'p3_predictions_by_row.csv' for c, folder in OUTPUT_DIRS.items()}\n"
            "if not all(path.exists() for path in prediction_paths.values()):\n"
            "    print('Paired P3 comparison remains locked until both prediction tables exist.')\n"
            "else:\n"
            "    pred_5m = pd.read_csv(prediction_paths['5M']).sort_values('row_index').reset_index(drop=True)\n"
            "    pred_1h = pd.read_csv(prediction_paths['1H']).sort_values('row_index').reset_index(drop=True)\n"
            "    if len(pred_5m) != len(pred_1h) or not np.allclose(pred_5m.ARMAS, pred_1h.ARMAS):\n"
            "        raise ValueError('The cadence prediction tables do not contain identical ordered ARMAS targets.')\n"
            "    common_models = sorted((set(pred_5m.columns) & set(pred_1h.columns)) - {'row_index', 'ARMAS', 'NAIRAS V3'})\n"
            "    y = pred_5m.ARMAS.to_numpy()\n"
            "    rng = np.random.default_rng(42)\n"
            "    paired_rows = []\n"
            "    for model in common_models:\n"
            "        p5, p1 = pred_5m[model].to_numpy(), pred_1h[model].to_numpy()\n"
            "        for regime, mask in {'all': np.ones(len(y), bool), 'A <= 15': y <= 15, 'A > 15': y > 15}.items():\n"
            "            yy, a, b = y[mask], p5[mask], p1[mask]\n"
            "            differences = []\n"
            "            for _ in range(1000):\n"
            "                index = rng.integers(0, len(yy), len(yy))\n"
            "                rmse_1h = np.sqrt(np.mean((b[index] - yy[index]) ** 2))\n"
            "                rmse_5m = np.sqrt(np.mean((a[index] - yy[index]) ** 2))\n"
            "                differences.append(rmse_1h - rmse_5m)\n"
            "            low, median, high = np.quantile(differences, [.025, .5, .975])\n"
            "            paired_rows.append({'model': model, 'regime': regime,\n"
            "                '1H_minus_5M_RMSE_median': median, 'CI_2.5%': low, 'CI_97.5%': high})\n"
            "    paired_cadence_bootstrap = pd.DataFrame(paired_rows)\n"
            "    display(paired_cadence_bootstrap.round(4))\n"
            "    paired_cadence_bootstrap.to_csv(COMPARISON_DIR / 'p3_paired_1h_vs_5m_bootstrap.csv', index=False)\n"
        ),
        markdown_cell("## 4. Partition robustness and efficiency\n"),
        code_cell(
            "if rotations is not None:\n"
            "    metric_columns = [c for c in ['RMSE', 'MAE', 'R2', 'bias'] if c in rotations.columns]\n"
            "    grouping = [c for c in ['cadence', 'model', 'regime', 'split'] if c in rotations.columns]\n"
            "    rotation_summary = rotations.groupby(grouping)[metric_columns].agg(['mean', 'std'])\n"
            "    display(rotation_summary.round(4))\n"
            "    rotations.to_csv(COMPARISON_DIR / 'all_assignments_cadence_comparison.csv', index=False)\n"
            "else:\n"
            "    print('Run all six assignments in both notebooks to compare partition robustness.')\n\n"
            "if complexity is not None:\n"
            "    keep = [c for c in ['cadence', 'model', 'trainable_parameters',\n"
            "          'parameter_and_buffer_MB', 'approx_FLOPs_per_observation',\n"
            "          'latency_ms_per_observation'] if c in complexity.columns]\n"
            "    display(complexity[keep].sort_values(['model', 'cadence']).round(4))\n"
            "    complexity.to_csv(COMPARISON_DIR / 'network_complexity_by_cadence.csv', index=False)\n"
        ),
        markdown_cell(
            "## 5. Cross-cadence attribution consistency\n\n"
            "On the P3 final test, compare integrated-gradient strength by physical variable group. Agreement across cadences strengthens an association claim; disagreement can indicate scale sensitivity or correlated predictors sharing attribution.\n"
        ),
        code_cell(
            "ig_paths = {c: folder / 'p3_integrated_gradients_variable_lag.csv' for c, folder in OUTPUT_DIRS.items()}\n"
            "if not all(path.exists() for path in ig_paths.values()):\n"
            "    print('Integrated-gradient comparison is unavailable until both P3 attribution sections run.')\n"
            "else:\n"
            "    def physics_group(feature):\n"
            "        if feature.startswith('NM_'): return 'neutron monitors'\n"
            "        if feature.startswith('SXR_'): return 'solar X-ray'\n"
            "        if feature.startswith('Particles_'): return 'energetic particles'\n"
            "        if feature.startswith('SW_'): return 'solar wind'\n"
            "        if feature.startswith('Index_'): return 'geomagnetic indices'\n"
            "        if feature.startswith('Solar_'): return 'solar activity'\n"
            "        return 'other'\n"
            "    ig_frames = []\n"
            "    for cadence, path in ig_paths.items():\n"
            "        frame = pd.read_csv(path)\n"
            "        frame['cadence'] = cadence\n"
            "        frame['physics_group'] = frame.feature.map(physics_group)\n"
            "        ig_frames.append(frame)\n"
            "    ig = pd.concat(ig_frames, ignore_index=True)\n"
            "    ig_group = (ig.groupby(['cadence', 'cohort', 'physics_group'], as_index=False)\n"
            "                .mean(numeric_only=True))\n"
            "    display(ig_group.sort_values(['cohort', 'cadence', 'mean_absolute_IG'],\n"
            "                                  ascending=[True, True, False]).round(6))\n"
            "    sns.catplot(data=ig_group, x='mean_absolute_IG', y='physics_group',\n"
            "                hue='cadence', col='cohort', kind='bar', sharex=False,\n"
            "                height=5, aspect=1.05)\n"
            "    plt.show()\n"
            "    ig.to_csv(COMPARISON_DIR / 'integrated_gradients_by_cadence.csv', index=False)\n"
            "    ig_group.to_csv(COMPARISON_DIR / 'integrated_gradients_physics_groups.csv', index=False)\n"
        ),
        markdown_cell(
            "## 6. Report questions\n\n"
            "1. On P2, which cadence has lower overall RMSE for each architecture?\n"
            "2. Does that conclusion hold for `ARMAS > 15`?\n"
            "3. After configurations are frozen, does the same cadence win on locked P3?\n"
            "4. Are cadence differences stable across all six partition assignments?\n"
            "5. Does the 24-hour 1H context improve results enough to justify its compute and latency relative to the one-hour 5M context?\n"
            "6. Do attention and integrated-gradient maps identify consistent variable groups at the two time scales?\n\n"
            "Treat attribution as association. State the cadence, temporal span, run mode, selected loss, and partition role beside every reported result.\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "06_compare_5m_vs_1h_results_colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = FINAL_DIR / "06_compare_5m_vs_1h_results_colab.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def validate_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not notebook.get("cells"):
        raise RuntimeError(f"Invalid notebook structure: {path}")
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell_text(cell)
        try:
            compile(source, f"{path.name}:cell-{index}", "exec")
        except SyntaxError as error:
            raise RuntimeError(f"Python syntax error in {path.name}, cell {index}") from error


def validate_cadence_content(path: Path, cadence: str, steps: int, output_slug: str) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(cell_text(cell) for cell in notebook["cells"])
    required = [
        f'CADENCE_CODE = "{cadence}"',
        f"SEQUENCE_LENGTH = {steps}",
        f'STUDY_OUTPUT_SUBFOLDER = "{output_slug}"',
        "source.shape != (SEQUENCE_LENGTH, 35)",
        ".reshape(n, SEQUENCE_LENGTH, 35)",
        "torch.zeros(1, SEQUENCE_LENGTH, config.d_model)",
        "cadence_lag_labels",
    ]
    missing = [item for item in required if item not in source]
    if missing:
        raise RuntimeError(f"{path.name} is missing cadence controls: {missing}")


def main() -> None:
    template = json.loads(SOURCE.read_text(encoding="utf-8"))
    outputs = [
        build_cadence_notebook(
            template,
            cadence_code="5M",
            cadence_minutes=5,
            sequence_length=12,
            output_slug="physics_guided_residual_5m",
            filename="04_physics_guided_5m_full_pipeline_colab.ipynb",
        ),
        build_cadence_notebook(
            template,
            cadence_code="1H",
            cadence_minutes=60,
            sequence_length=24,
            output_slug="physics_guided_residual_1h",
            filename="05_physics_guided_1h_full_pipeline_colab.ipynb",
        ),
        build_comparison_notebook(),
    ]
    for output in outputs:
        validate_notebook(output)
    validate_cadence_content(outputs[0], "5M", 12, "physics_guided_residual_5m")
    validate_cadence_content(outputs[1], "1H", 24, "physics_guided_residual_1h")
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
