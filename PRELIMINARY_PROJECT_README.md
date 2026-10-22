# ARMAS 5 Minute Preliminary and Final Project Workflow

This project uses only the three `ARMAS_TIMESERIES_5M_20132023_P*.json` files.
P1 is the training partition, P2 is the validation partition, and P3 is the
final test partition.

## Recommended starting point

Open `notebooks/preliminary/ARMAS_5M_proposal_deep_learning_executed.ipynb`.
It contains the exact proposal-aligned PyTorch code in explained sections.

The notebook performs:

- dataset validation and missingness checks;
- target, NAIRAS-residual, and feature-by-lag visualizations;
- the 43-input current-time JMLP;
- hybrid temporal CNN-MLP;
- direct Temporal Transformer;
- physics-guided residual Transformer;
- weighted Huber residual Transformers with weights 2 and 4;
- learning curves, parameter counts, model comparisons, and temporal occlusion.

P1 trains every model, P2 supports model selection and early stopping, and P3
is evaluated only after selection is complete.

## Reproduce the full preliminary analysis

Install the required packages:

```powershell
python -m pip install -r requirements-preliminary.txt
```

Run the executed deep-learning experiment and regenerate its checkpoints,
predictions, CSV tables, and figures:

```powershell
python armas_proposal_models_5m.py
```

The analysis writes to `outputs/proposal_models_5m/`. The separate
`armas_5m_full_study.py` script remains available for the broader EDA and
classical diagnostic baselines.

Build the PDF report after the analysis finishes:

```powershell
python build_deep_learning_preliminary_report.py
```

The report is written to
`output/pdf/Sanjib_KC_preliminary_project_report.pdf`.

The matching standalone LaTeX source is
`latex/Sanjib_KC_preliminary_project_report.tex`. See `latex/README.md` for
local and Overleaf compilation instructions.

## Extended final deep learning code

After the preliminary model and loss choices are frozen, use
`notebooks/final/04_physics_guided_5m_full_pipeline_colab.ipynb` for the final
GPU Transformer study. It implements the current-time MLP, hybrid CNN-MLP,
direct Transformer, physics-guided residual Transformer, weighted Huber loss,
attention, integrated gradients, grouped permutation, and partition rotations.

Run the final notebook in this order:

1. smoke mode to verify shapes and GPU execution;
2. development mode with P1 training and P2 validation;
3. record and freeze architecture and loss choices;
4. unlock P3 once for the final test;
5. run the frozen partition rotations and interpretation analyses.

Before submission, replace the repository URL placeholder and add the author's
email and PantherID. The submitted filename should follow the instructor's
`firstName_lastName_preliminary_project_report.pdf` convention.

