# CLAUDE.md: start here

Physics-guided temporal networks that correct the NAIRAS V3 aviation radiation model against in-flight
ARMAS measurements. CSC 8851 Deep Learning, Fall 2026, final project (individual; poster slot "Team 5").
Deadlines: poster session **Nov 30 / Dec 2, 2026**; final report **Dec 7, 2026**.
Current state and to-do list: [`notes/next_steps.md`](notes/next_steps.md). Results table: [`README.md`](README.md).

## Hard rules
- Work only on branch **`final-project`**; push only there. **Never change or push `main`** (it keeps the October state).
- **NAIRAS is never a network input.** Residual models predict `r = A − N` from position, shielding and space weather,
  and the prediction is `N + r`. Direct models get the same inputs and predict `A`. (Decided 2026-11-27; all models retrained.)
- Only neural networks are trained. The 2025 XGBoost result (K C et al., ICDMW 2025) is quoted, not retrained.
- Every result uses all 6 train/validation/test assignments (rotations 123, 132, 213, 231, 312, 321) × 3 seeds (42, 1, 2).
  Analyses (importance, occlusion, error causes) use seed 42 on all 6 assignments, for both histories.
- Data naming is inverted from intuition: `data/5M` = 5-minute steps = **1-hour history** (12 steps);
  `data/1H` = hourly steps = **24-hour history** (24 steps).
- Cite only verified, real references. The instructor penalizes hallucinated references.

## Report (`latex/Sanjib_KC_final_project_report.tex`)
Instructor checklist (preliminary report got a B; these items cost points):
official CVPR template, camera-ready; author block of 1–2 lines; Introduction goes WHY → current approaches and their gaps → HOW;
Related Work contrasts each approach with ours; detailed self-contained captions (tables: caption above; figures: below);
every figure, table and equation referenced in the text; equations numbered and punctuated; abstract short, no citations;
**main text exactly 4 pages**, Peer Evaluation starts page 5, then References, then appendix.
Figures must look like a CVPR paper (no titles inside images; the caption explains). Do not reuse poster-style PNGs in the report.

- Paper figures: `src/paperstyle.py` (style) + `src/paper_figures.py` → `figures/paper/*.pdf` (`python3 src/paper_figures.py [name ...]`).
- Compile locally (TinyTeX is installed in `~/Library/TinyTeX`):
  ```bash
  export PATH=$PATH:$HOME/Library/TinyTeX/bin/universal-darwin
  cd latex && pdflatex Sanjib_KC_final_project_report && bibtex Sanjib_KC_final_project_report && pdflatex Sanjib_KC_final_project_report && pdflatex Sanjib_KC_final_project_report
  ```
  Check the page count via the `.aux` labels: `sec:conclusions` must be on page 4 and `app:eda` on page 6.
- `*.pdf` is gitignored: commit figures and the report PDF with `git add -f`.

## Pipeline (reproduce everything)
```bash
for s in 42 1 2; do for r in 123 132 213 231 312 321; do python3 train_basic.py --rotation $r --seed $s; done; done
bash scripts/run_final.sh          # temporal models, 1 h + 24 h, + high-dose weighted Transformer (~1 h on an M5 laptop)
bash scripts/run_analysis.sh       # stats, importance, history flattening, occlusion, tracking, all figures, web exports
python3 poster/deck_update_results.py && python3 poster/deck_update_retrain.py   # patch the slides in place
```
Key code: `src/data.py` (loading, splits, scaler), `src/models.py` (Flat MLP, CNN–MLP, TCN, Transformer),
`train_basic.py` (MLPs), `train_temporal.py`, `src/final_stats.py` (seed ensembles + flight bootstrap),
`src/explain_errors.py`, `src/history_importance.py`, `src/variable_importance.py`, `src/occlusion.py` (Transformer, incl.
variable × hour map), `src/occlusion_all.py` (all models), `scripts/track_training.py` (attention per epoch).

## Slides (`poster/Team5_Sanjib_KC_slides.pptx`, 16 slides)
Never rebuild with `poster/build_deck.py` (it overwrites hand edits and has old numbers). Patch instead:
`poster/deck_qr.py`, `poster/deck_update_results.py`, `poster/deck_update_retrain.py` (idempotent).
When changing text, keep each run's formatting: some text boxes have several runs (e.g. slide 8's equation).
PowerPoint automation is blocked for Claude, so the user must view the slides; verify text with python-pptx.

## Interactive pages (Artifacts, private until the user shares them)
| Page | Source | Build | URL |
|---|---|---|---|
| How the Six Models Differ (tutorial, real numbers) | `poster/models_explained/` | `python3 scripts/export_tutorial.py && python3 poster/models_explained/build.py` | https://claude.ai/artifact/H5NE4f65zQJuUqFh45hCSn |
| Watching Attention Learn | `poster/animation/` | `python3 scripts/export_animation.py && python3 poster/animation/build.py` | https://claude.ai/artifact/SoxLwXhTsnEeFPNjU8EjjC |
| Flight Model Zoo (networks run live in JS) | `poster/zoo/` | `python3 scripts/export_model_zoo.py && python3 poster/zoo/build.py` | https://claude.ai/artifact/XTzA1TvvKFi7pJ45pAKZ9B |
| Flight Dose Lab (index.html + data.json) | `poster/web/` | `python3 src/export_web.py results/temporal_final` | https://claude.ai/artifact/X3saE5EHSytJkidqedoo52 |

Republishing from a new session: the Artifact tool requires reading the live version first. The Zoo is 2.6 MB:
read only a few lines of the saved copy, then republish the local build.

## Headline results (no NAIRAS input; 18-run means, test RMSE in µSv/h)
NAIRAS 4.044 · MLP direct 3.842 · MLP 3.740 · best overall **CNN–MLP 24 h 3.667 (−9.3%)** ·
best high dose **weighted Transformer 1 h 5.551 above 15 µSv/h (−19%)** · Flat MLP worst network (3.87–3.91).
Residual vs direct MLP (same inputs): −0.07 overall, −0.29 high dose (significant). 1 h vs 24 h: no difference.
Remaining error: 83% within flights; aircraft 5% (chance 3%) → instrument effect.
Old results (NAIRAS as input) are kept locally in `results_with_nairas_input/` (gitignored) for comparison.

## Working with the user
They learn by active recall: quiz them and keep `LEARNING_LOG.md` (recall cards). They want the report, README, slides
and pages kept in sync after every change, and plain-language explanations with real examples from the data.
