# LaTeX report instructions

The complete report source is `Sanjib_KC_preliminary_project_report.tex`.
It uses a standalone two-column layout and reads the executed neural figures
from `../outputs/proposal_models_5m/` plus the target-distribution figure from
`../outputs/armas_5m_full_study/`.

Before submission:

1. Replace the email, PantherID, and repository URL placeholders.
2. If the instructor provides the official CVPR template, copy the document
   body into that template and retain the same figures, tables, equation, and
   references.
3. Upload `01_proposal_model_rmse.png`, `02_learning_curves.png`,
   `03_best_observed_vs_predicted.png`, `04_transformer_occlusion.png`, and
   `figure01_target_distribution.png` when compiling on Overleaf, then adjust
   `\graphicspath` if they are placed in a different folder.

Compile locally from the `latex` directory with:

```text
pdflatex Sanjib_KC_preliminary_project_report.tex
pdflatex Sanjib_KC_preliminary_project_report.tex
```

The second pass resolves figure, table, and equation references.
