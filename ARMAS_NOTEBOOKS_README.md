# ARMAS notebook organization

## Current PDF-aligned workflow

Use this notebook for the current project proposal:

1. `notebooks/final/02_pdf_aligned_full_pipeline_colab.ipynb`

Use this companion notebook for the direct-ARMAS, NAIRAS-free comparison:

2. `notebooks/final/03_direct_armas_no_nairas_full_pipeline_colab.ipynb`

## Cadence comparison: 5M versus 1H

Use these matched Colab notebooks to test temporal scale while preserving the PDF-aligned physics-guided design:

1. `notebooks/final/04_physics_guided_5m_full_pipeline_colab.ipynb`
   - loads the three `ARMAS_TIMESERIES_5M_20132023_P*.json` files;
   - uses 12 temporal values per variable (55 minutes before the observation through current time);
   - writes only to `ARMAS_project_outputs/physics_guided_residual_5m`.
2. `notebooks/final/05_physics_guided_1h_full_pipeline_colab.ipynb`
   - loads the three `ARMAS_TIMESERIES_1H_20132023_P*.json` files;
   - uses 24 temporal values per variable (23 hours before the observation through current time);
   - writes only to `ARMAS_project_outputs/physics_guided_residual_1h`.
3. `notebooks/final/06_compare_5m_vs_1h_results_colab.ipynb`
   - reads the saved results from both cadence runs;
   - compares P2 validation before final P3 testing;
   - compares overall and high-dose performance, temporal gain, residual-learning gain, paired P3 uncertainty, partition robustness, complexity, feature-by-lag associations, and integrated-gradient physics groups.

Run 04 and 05 with the same `RUN_MODE`, validation protocol, frozen model list, and P3 gate. Run 06 after both have produced P2 outputs; its P3 and attribution sections activate only after both locked-test runs exist. The cadence comparison changes two linked quantities by design: sampling interval and temporal span. Interpret it as a comparison between a short, high-resolution one-hour context and a longer, lower-resolution 24-hour context, rather than as an isolated sampling-rate experiment.

Notebook 03 preserves the same partitions, metrics, hyperparameter study, visual diagnostics, attribution tools, and six train-validation-test assignments. All learned models predict ARMAS directly. `NAIRASV2` and `NAIRASV3` are excluded from model inputs; NAIRAS V3 is used only as an external benchmark after prediction.

The notebooks can process the same dataset from the same mounted Google Drive, but their results are isolated:

- Notebook 02: `ARMAS_project_outputs/physics_guided_residual`
- Notebook 03: `ARMAS_project_outputs/direct_armas_no_nairas`

Each notebook checks its required output-folder name before creating files, preventing accidental overwrites.

It implements the current **Physics-Guided Temporal Transformer** proposal, including:

- feature-by-lag exploration of `ARMAS - NAIRAS V3`;
- current-time MLP;
- hybrid CNN-MLP;
- direct Temporal Transformer;
- physics-guided Transformer trained on the residual;
- standard MSE versus validation-selected weighted Huber training;
- overall, `ARMAS <= 15`, and `ARMAS > 15` metrics;
- separate Phase 1 overall analysis followed by Phase 2 radiation-regime analysis;
- train/validation loss curves, learning-rate histories, timing, and memory;
- model/hyperparameter/temporal-window ablations and seed stability;
- parameter, FLOP, storage, latency, and accuracy-efficiency analysis;
- attention maps, integrated gradients, feature-group permutation, and variable-by-lag heatmaps;
- dose-bin diagnostics, calibration, bootstrap comparisons, and automatic research-question answers;
- all six ordered P1/P2/P3 train-validation-test assignments, including separate primary cyclic summaries;
- train, validation, and test RMSE for every model and assignment.
- an expanded visual suite covering data quality, normalization, convergence, hyperparameters, complexity, calibration, error distributions, attribution, and partition robustness;
- explicit **What you need to do** instructions beside each decision point.

## Earlier exploration notebooks

These are retained separately for learning, preliminary exploration, and optional ablations:

1. `notebooks/exploration/01_armas_5m_exploration.ipynb`
2. `notebooks/exploration/02_stepwise_gpu_models_colab.ipynb`
3. `notebooks/exploration/03_transformer_experiments_colab.ipynb`

The exploration notebooks reflect the earlier direct-prediction plan. Do not use them as the sole source of final results for the current PDF; use the final notebook above.

## Verified dataset facts

| Partition | Rows | ARMAS > 15 | Date range |
|---|---:|---:|---|
| P1 | 32,369 | 5,324 | 2013-06-28 to 2023-03-23 |
| P2 | 30,752 | 4,983 | 2013-08-03 to 2023-03-25 |
| P3 | 29,294 | 4,947 | 2013-06-28 to 2023-03-19 |

All 35 temporal variables contain 12 values. Source index 0 is current time; the modeling notebooks reverse the arrays once so tensors are chronological from past to current.

## Colab sequence

1. Upload the dataset folder and desired notebook to Google Drive.
2. Open the notebook in Colab and select a T4 GPU.
3. Uncomment the Drive mount/path cell.
4. Start with `RUN_MODE="smoke"`, then use `"development"` and `"full"`.
5. Select configurations using validation results only.
6. Unlock P3 only after configurations are frozen.
7. Set `RUN_MODE="full"`; the notebook then enables all six partition assignments for final results.
