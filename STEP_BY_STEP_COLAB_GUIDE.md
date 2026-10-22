# ARMAS project: step-by-step Colab guide

## What to use

The notebook that matches the current PDF is:

`notebooks/final/02_pdf_aligned_full_pipeline_colab.ipynb`

The three notebooks under `notebooks/exploration/` are retained for preliminary learning and optional experiments. They do not replace the final PDF-aligned notebook.

## Step 1: prepare Google Drive

1. Create a folder such as `MyDrive/ARMAS_Project/`.
2. Upload the complete `ARMASdatasets_COMPLETE` folder.
3. Upload `02_pdf_aligned_full_pipeline_colab.ipynb`.
4. Open the notebook with Google Colab.
5. Select **Runtime → Change runtime type → T4 GPU**.

You do not need to create the output folder manually. After Drive is mounted, the notebook creates `OUTPUT_DIR` and any missing parent folders automatically.

Do not rename individual JSON files. The notebook discovers these exact filenames:

- `ARMAS_TIMESERIES_5M_20132023_P1.json`
- `ARMAS_TIMESERIES_5M_20132023_P2.json`
- `ARMAS_TIMESERIES_5M_20132023_P3.json`

## Step 2: connect the notebook to Drive

In the paths cell:

1. Comment out the local `DATA_DIR` and `OUTPUT_DIR` lines.
2. Uncomment the four Colab/Drive lines.
3. Change the Drive paths if your folder names differ.
4. Run the cell and confirm that no `Fix DATA_DIR` assertion appears.

All tables, histories, checkpoints, and final metrics will be written beneath `OUTPUT_DIR`.

When both final notebooks use the same Google Drive, keep this shared parent with separate subfolders:

```python
# Notebook 02: physics-guided residual study
OUTPUT_DIR = Path('/content/drive/MyDrive/ARMAS_project_outputs/physics_guided_residual')

# Notebook 03: direct ARMAS without NAIRAS inputs
OUTPUT_DIR = Path('/content/drive/MyDrive/ARMAS_project_outputs/direct_armas_no_nairas')
```

Both notebooks check the final folder name and stop with an error if the wrong study folder is selected.

Keep `SAVE_FIGURES = True` to save each report-ready PNG automatically. The notebook ends with an artifact inventory showing every saved CSV, PNG, and checkpoint. Markdown blocks labeled **What you need to do** explain when a flag should change and how each figure should be interpreted.

## Step 3: run a smoke test

Set:

```python
RUN_MODE = "smoke"
```

Run the notebook from the beginning through the P1/P2 validation section. The notebook will use only a few epochs.

Confirm:

- the GPU is detected;
- P1, P2, and P3 have the expected row and tensor counts;
- the feature-by-lag heatmap appears;
- training loss generally decreases;
- validation RMSE is finite;
- all required models train without a shape or memory error.

Do not interpret smoke-test metrics as final results.

## Step 4: understand the required models

The notebook trains four learned models and one external benchmark:

1. **NAIRAS V3:** physics benchmark, with no training.
2. **Current-time MLP:** predicts ARMAS from 8 static and 35 current environmental values.
3. **Hybrid CNN-MLP:** predicts ARMAS from the one-hour environmental sequence and 8 static variables.
4. **Direct Transformer:** predicts ARMAS directly from the sequence and complete 43-variable current-time vector.
5. **Physics-guided Transformer:** predicts `ARMAS - NAIRAS V3`, then reconstructs ARMAS by adding NAIRAS V3.

`Vehicle_ID` and `NAIRASV2` are never predictors. NAIRAS V3 is an input only to the explicitly physics-guided residual model.

## Step 5: run full P1/P2 validation experiments

Restart the runtime to clear smoke-test models, then set:

```python
RUN_MODE = "development"  # 25-epoch budget for iteration
RUN_VALIDATION_EXPERIMENTS = True
RUN_HYPERPARAMETER_STUDY = True
EVALUATE_P3 = False
```

When the code and design choices are stable, use `RUN_MODE = "full"` for the 80-epoch final budget. Early stopping can finish sooner when validation performance stops improving.

Run from the beginning through the validation comparison.

With P3 locked, sections 9-12 now produce exploratory P2 figures instead of empty cells. They switch automatically to P3 after the final-test flag is enabled, and every title/file name identifies which partition was used.

The roles are:

- P1: training;
- P2: validation and configuration selection;
- P3: locked test partition.

The notebook compares standard MSE training with weighted Huber residual models using high-dose weights 1, 2, and 4.

Read the outputs in two phases. First use the unweighted loss curves and the `regime == "all"` table to compare models on every observation. Only then use the `A <= 15` and `A > 15` tables and weighted-Huber runs to study imbalance and trade-offs.

Use P2 to answer:

- Which model has the lowest overall RMSE?
- Does residual learning improve over uncorrected NAIRAS V3?
- Which weight improves `ARMAS > 15` without greatly harming `ARMAS <= 15`?
- Does the one-hour model improve over the current-time MLP?

## Step 6: freeze the final configurations

Edit `FINAL_CONFIG_NAMES` based only on P2 results. Keep these four required standard models:

- `MLP`
- `Hybrid CNN-MLP`
- `Direct Transformer`
- `Physics-guided Transformer`

Add exactly one validation-selected weighted-Huber model, for example:

```python
"Physics-guided Huber w=2"
```

Record why that weight was selected before viewing P3.

## Step 7: evaluate P3 once

After configurations are frozen, change:

```python
EVALUATE_P3 = True
```

Run the P3 and interpretation sections. Do not repeatedly tune the architecture or loss after seeing P3.

Save and inspect:

- overall MAE, RMSE, R², and bias;
- `ARMAS <= 15` metrics;
- `ARMAS > 15` metrics;
- predicted-versus-observed plots;
- residual distributions;
- attention heatmap;
- integrated-gradients overall/high-dose heatmaps, ranked signed/absolute attributions, and completeness check;
- feature-group permutation table and plot.

Attribution indicates association with predictions, not physical causation.

## Step 8: run all six train/validation/test assignments

Keep the selected configurations unchanged and set:

```python
RUN_MODE = "full"
EVALUATE_P3 = True
RUN_ALL_PARTITION_PERMUTATIONS = True
```

The notebook runs every ordered assignment:

1. `123`: P1 train, P2 validation, P3 test;
2. `132`: P1 train, P3 validation, P2 test;
3. `213`: P2 train, P1 validation, P3 test;
4. `231`: P2 train, P3 validation, P1 test;
5. `312`: P3 train, P1 validation, P2 test;
6. `321`: P3 train, P2 validation, P1 test.

Each assignment fits new imputers, scalers, and model weights using only its training partition. It reports train, validation, and test RMSE for every model, along with regime-specific metrics and generalization gaps. The proposal's primary cyclic assignments (`123`, `231`, and `312`) are also saved separately.

## Step 9: collect report material

Use these saved outputs:

- `p2_validation_results.csv`
- `p3_locked_test_results.csv`
- `all_six_partition_assignments_long.csv`
- `all_six_assignments_train_val_test_rmse.csv`
- `all_six_assignments_rmse_by_regime.csv`
- `all_six_assignments_summary.csv`
- `primary_three_rotation_results.csv`
- `primary_three_rotation_summary.csv`
- `architecture_parameter_counts.csv`
- `network_complexity.csv` and `accuracy_efficiency_tradeoff.csv`;
- `convergence_summary.csv` and `research_question_answers.csv`;
- optional hyperparameter and seed-stability tables;
- P3 dose-bin, calibration, bootstrap, and integrated-gradients tables;
- `p1_residual_feature_by_lag_correlations.csv`
- training-history CSV files;
- saved PyTorch checkpoints;
- prediction, residual, attention, integrated-gradient, and permutation figures.

The preliminary/final report should explain:

1. the residual-learning equation;
2. flight-separated partitions and train-only preprocessing;
3. the five-way comparison;
4. standard versus imbalance-aware training;
5. overall and radiation-regime performance;
6. whether one-hour history adds value;
7. which variables/lags are associated with discrepancies;
8. limitations and the non-causal interpretation of attribution.

## Optional exploration notebooks

Use the notebooks in `notebooks/exploration/` when you want additional descriptive plots or earlier direct-prediction experiments. Keep their outputs labeled as exploratory so they are not confused with the final PDF-aligned experiment.

## Optional 5M versus 1H physics comparison

After the main proposal pipeline is stable, upload and run:

1. `04_physics_guided_5m_full_pipeline_colab.ipynb`;
2. `05_physics_guided_1h_full_pipeline_colab.ipynb`;
3. `06_compare_5m_vs_1h_results_colab.ipynb`.

Keep the same run mode and experimental flags in notebooks 04 and 05. Each notebook automatically mounts Drive in Colab, loads its cadence-specific P1/P2/P3 files, validates the expected 12-step or 24-step tensor, and saves to a separate output directory.

Use P2 to answer the exploratory physics questions first:

- Does high-resolution 5M history better capture fast solar-wind, X-ray, or particle variation?
- Does the longer 1H history better capture persistent geomagnetic and solar conditions?
- Does temporal modeling improve over the current-time MLP at each cadence?
- Does physics-guided residual learning improve over direct prediction and uncorrected NAIRAS V3?
- Are gains retained for `ARMAS > 15`, where observations are rarer?

Freeze configurations separately from P2, unlock P3 once in both notebooks, and then use notebook 06 for the paired cadence bootstrap, all-six-assignment robustness comparison, efficiency comparison, and attribution consistency. The 5M input spans about one hour, while the 1H input spans about 24 hours, so conclusions concern the combined effect of resolution and context length.
