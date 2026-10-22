# Current PDF requirements and implementation map

The current source is `Deep_Learning_Project_final.pdf`, titled **Physics-Guided Temporal Transformer for Aviation Effective Dose-Rate Estimation**.

| Proposal requirement | Implementation |
|---|---|
| 92,415 observations and three flight-separated partitions | Dataset validation and overview in notebook 02 |
| 8 static variables and 35 variables × 12 five-minute steps | Loader assertions and tensor-shape checks |
| Vehicle ID excluded | Never loaded into model tensors |
| Current-time MLP with 43 inputs | `CurrentMLP` |
| Hybrid CNN–MLP | `HybridCNNMLP` |
| Direct Temporal Transformer predicting ARMAS | `DirectTransformer` |
| Physics-guided Transformer predicting ARMAS − NAIRAS V3 | `PhysicsGuidedTransformer` |
| Reconstruct prediction as NAIRAS V3 + predicted residual | `reconstruct_armas` |
| Standard versus weighted Huber training | `STANDARD_CONFIGS` and `WEIGHTED_HUBER_CONFIGS` |
| Select high-dose weight on validation only | P1/P2 validation grid with P3 gate |
| Three proposal cyclic partition rotations | Primary assignments `123`, `231`, and `312`, saved separately |
| All six ordered partition assignments | `123`, `132`, `213`, `231`, `312`, and `321` |
| Train/validation/test RMSE for every assignment | Wide overall and regime-specific RMSE tables with generalization gaps |
| Train-only missing handling and standardization | `fit_preprocessors` inside each rotation |
| Adam, dropout, weight decay, early stopping | Shared training loop |
| Overall analysis before radiation-regime split | Phase 1 overall table and unweighted loss curves |
| Separate A ≤ 15 and A > 15 trade-off analysis | Phase 2 regime tables and weighted-Huber comparison |
| Meaningful training beyond a three-epoch check | Development (25) and full (80) epoch budgets with LR scheduling |
| MAE, RMSE, R² overall, A ≤ 15, and A > 15 | `evaluate_regimes` |
| Compare NAIRAS V3 on identical held-out observations | Validation, P3, and rotation result tables |
| Predicted-versus-observed and residual plots | Notebook section 9 |
| Attention maps | Notebook section 10 |
| Integrated gradients | Notebook section 11 |
| Integrated-gradients interpretation and validation | Overall/high-dose cohorts, signed and absolute IG, ranked variable-lag table, completeness check |
| Variable-by-lag heatmaps | Exploration and integrated-gradients heatmaps |
| Feature-group permutation | Notebook section 12 |
| Attribution interpreted as association | Notebook interpretation notes |
| Network complexity | Parameters, storage, approximate FLOPs, latency, and accuracy-efficiency analysis |
| Robustness and uncertainty | Seed study, bootstrap RMSE differences, dose-bin errors, calibration, and cyclic rotations |
| Report-ready answers | Automatic research-question evidence table |

Recommended run order:

1. `notebooks/exploration/01_armas_5m_exploration.ipynb` for general data quality and distributions.
2. `notebooks/final/02_pdf_aligned_full_pipeline_colab.ipynb` for the current proposal's complete experimental workflow.

Earlier notebooks under `notebooks/exploration/` are retained for preliminary exploration and optional ablations, but should not be used as the sole source of final results.
