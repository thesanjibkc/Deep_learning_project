# Physics-Guided Temporal Networks for Aviation Effective Dose-Rate Estimation

CSC 8851 Deep Learning, Fall 2026 · Sanjib K C · Team 5

Neural networks that learn a correction to the NAIRAS V3 physics model from aircraft
position, geomagnetic shielding, and the recent history of solar and geomagnetic
conditions, evaluated against in-flight ARMAS measurements.

**Final report:** [`latex/Sanjib_KC_final_project_report.tex`](latex/Sanjib_KC_final_project_report.tex)
(CVPR format, 4 pages + peer evaluation + references + appendix; figures in `figures/paper/`)
**Poster / slides:** [`poster/`](poster/)
**Code (this branch):** https://github.com/thesanjibkc/Deep_learning_project/tree/final-project  
**Learning notes:** [`LEARNING_LOG.md`](LEARNING_LOG.md)

---

## Status (updated 2026-11-27)

| Step | Status |
|---|---|
| Exploration 1–4 (where NAIRAS fails, maps, solar cycle, history) | ✅ in report |
| MLP direct and residual, 6 rotations × 3 seeds | ✅ |
| Regularization tuning (validation of rotation 123 only) | ✅ width 16, dropout 0.3, weight decay 0.01 |
| Temporal models (Flat MLP, CNN–MLP, TCN, Transformer, direct Transformer), 1 h and 24 h, 6 rotations × 3 seeds | ✅ |
| High-dose weighted Transformer (Huber, w = 4) | ✅ |
| Seed ensembles + paired flight-bootstrap significance | ✅ `results/final/` |
| Importance, history flattening, per-variable importance, occlusion (all models), remaining-error analysis, learning curves, for 1 h **and** 24 h | ✅ `results/explain/` |
| Epoch-by-epoch attention tracking | ✅ `results/tracking/` |
| Interactive pages: tutorial, training animation, Flight Model Zoo, Flight Dose Lab | ✅ `poster/models_explained/`, `poster/animation/`, `poster/zoo/`, `poster/web/` |
| Final report | ✅ 4 pages, compiled locally |
| Slides (16) | ✅ |

**Design:** NAIRAS is never a network input. Residual models predict the correction
`r = A − N` from position, shielding and space weather only, and the prediction is `N + r`; direct
models get the same inputs and predict `A`. (Until 2026-11-26 the residual models also received
NAIRAS as an input; all models were retrained without it. The old results are kept locally in
`results_with_nairas_input/`, which is gitignored.)

## Final results (test set, mean ± std over 18 runs = 6 rotations × 3 seeds)

Training time is the mean per run on an Apple M5 laptop GPU (PyTorch 2.8, MPS), including early
stopping and an evaluation of all observations after every epoch.

| Model | History | RMSE all (µSv/h) | RMSE A > 15 | Gain vs NAIRAS (all / high) | Parameters | Train time (s) |
|---|---|---|---|---|---|---|
| NAIRAS V3 | – | 4.044 ± 0.040 | 6.845 ± 0.277 | — | – | – |
| XGBoost (K C et al. 2025, cited, static data) | – | 3.671 ± 0.061 | not reported | 9.2% | – | – |
| MLP, direct | none | 3.842 ± 0.096 | 6.627 ± 0.203 | 5.0% / 3.2% | 13,953 | 2.5 |
| MLP | none | 3.740 ± 0.044 | 6.325 ± 0.194 | 7.5% / 7.6% | 13,953 | 2.0 |
| Flat MLP | 1 h | 3.874 ± 0.032 | 6.619 ± 0.161 | 4.2% / 3.3% | 29,569 | 2.7 |
| Flat MLP | 24 h | 3.909 ± 0.057 | 6.708 ± 0.234 | 3.3% / 2.0% | 56,449 | 1.8 |
| CNN–MLP | 1 h | 3.686 ± 0.028 | 6.144 ± 0.230 | 8.9% / 10.2% | 6,993 | 3.1 |
| **CNN–MLP** | **24 h** | **3.667 ± 0.043** | 6.136 ± 0.266 | **9.3%** / 10.4% | 6,993 | 3.3 |
| TCN | 1 h | 3.678 ± 0.042 | 6.115 ± 0.168 | 9.1% / 10.7% | 8,769 | 6.3 |
| TCN | 24 h | 3.679 ± 0.049 | 6.168 ± 0.187 | 9.0% / 9.9% | 10,337 | 8.4 |
| Transformer, direct | 1 h | 3.683 ± 0.034 | 6.268 ± 0.218 | 8.9% / 8.4% | 8,769 | 19.2 |
| Transformer, direct | 24 h | 3.712 ± 0.054 | 6.417 ± 0.230 | 8.2% / 6.3% | 8,961 | 27.5 |
| Transformer | 1 h | 3.688 ± 0.041 | 6.284 ± 0.150 | 8.8% / 8.2% | 8,769 | 21.5 |
| Transformer | 24 h | 3.683 ± 0.050 | 6.325 ± 0.190 | 8.9% / 7.6% | 8,961 | 28.8 |
| **Transformer, high-dose weight 4** | **1 h** | 3.788 ± 0.080 | **5.551 ± 0.246** | 6.3% / **18.9%** | 8,769 | 21.7 |
| Transformer, high-dose weight 4 | 24 h | 3.773 ± 0.092 | 5.696 ± 0.249 | 6.7% / 16.8% | 8,961 | 34.4 |

Seed ensembles (average of 3 seeds) lower every model by about 0.03 (Transformer 24 h: 3.643).
Significance (`results/final/significance.csv`, paired flight bootstrap, 95% intervals):
residual beats direct for the MLP (−0.07 overall, −0.29 above 15); CNN–MLP, TCN and Transformer
with 24 h beat the no-history MLP (−0.06); the Flat MLP is worse than every temporal model
(−0.25 vs Transformer); 1 h vs 24 h shows no difference; the 1-hour weighted Transformer matches
the MLP overall while cutting high-dose RMSE by 0.82 µSv/h.

**Interactive:** [How the Six Models Differ (tutorial)](https://claude.ai/artifact/H5NE4f65zQJuUqFh45hCSn) ·
[Flight Dose Lab](https://claude.ai/artifact/X3saE5EHSytJkidqedoo52) ·
[Watching Attention Learn](https://claude.ai/artifact/SoxLwXhTsnEeFPNjU8EjjC) ·
[Flight Model Zoo](https://claude.ai/artifact/XTzA1TvvKFi7pJ45pAKZ9B) (runs the trained networks live; private links, share from each page).
Next steps and open items: [`notes/next_steps.md`](notes/next_steps.md).

## Data

92,415 ARMAS observations (2013–2023) in three flight-separated partitions P1, P2, P3.

- **Target:** ARMAS effective dose rate `A` (µSv/h). High-dose regime: `A > 15`.
- **Physics baseline:** NAIRAS V3 `N`. Residual models learn `r = A − N` and predict `N + r`.
- **Static features (8):** latitude, longitude, barometric and GPS altitude,
  geomagnetic latitude and longitude, cutoff rigidity `Rc`, L-shell.
- **Temporal features (35):** neutron monitors (4), solar X-rays (2), energetic
  particles (7), solar wind (11), Kp/ap/Dst (3), solar activity (8).
  - `5M`: 12 steps × 5 min (last 55 min) · `1H`: 24 steps × 1 h (last 23 h).
  - Same observations in both; index 0 in the raw files is the current time
    (the loader reverses to oldest → current).
- Vehicle ID is never used as an input.

Put the two zip files in `data/raw/` and extract (the `data/` folder is not
committed; the first run builds a fast cache in `data/cache_*.npz`):

```bash
mkdir -p data/5M data/1H
unzip data/raw/ARMAS_TIMESERIES_5M_V3.zip -d data/5M
unzip data/raw/ARMAS_TIMESERIES_1H_V3.zip -d data/1H
```

## Setup

Python 3.9+, runs locally (Apple-silicon GPU via MPS, or CPU).

```bash
pip3 install --user -r requirements.txt
```

## Reproduce

```bash
# exploration figures -> figures/explore/
python3 explore/e01_where_nairas_fails.py
python3 explore/e02_flight_map.py
python3 explore/e03_solar_cycle.py
python3 explore/e04_temporal_information.py

# temporal models (both cadences, all rotations) -> results/temporal/<cadence>/
for c in 1H 5M; do for r in 123 132 213 231 312 321; do
  python3 train_temporal.py --cadence $c --rotation $r --seed 42
done; done
python3 src/summarize.py results/temporal/1H

# final temporal runs, tuning, statistics, explanations, figures
bash scripts/tune_temporal.sh          # validation-only tuning (rotation 123)
bash scripts/run_final.sh              # 6 rotations x 3 seeds x 2 cadences (+ weighted)
bash scripts/run_analysis.sh           # everything after training: stats, analyses (1 h and 24 h), figures, web pages
python3 src/final_stats.py             # seed ensembles + flight bootstrap
for r in 123 132 213 231 312 321; do python3 src/explain_errors.py 1H $r 42; done
for r in 123 132 213 231 312 321; do python3 src/history_importance.py 1H $r 42; done   # whose history matters (flatten / shuffle past only)
for c in 1H 5M; do for r in 123 132 213 231 312 321; do python3 src/variable_importance.py $c $r 42; python3 src/occlusion.py $c $r 42; python3 src/occlusion_all.py $c $r 42; done; done   # per-variable importance, occlusion (Transformer, all models)
python3 scripts/track_training.py      # epoch-by-epoch attention
python3 src/plot_results.py results/temporal_final; python3 src/plot_tracking.py; python3 src/plot_explain.py
python3 src/plot_best_vs_nairas.py        # NAIRAS vs best models against ARMAS (report Fig. 2)
python3 src/paper_figures.py              # all report figures, CVPR paper style -> figures/paper/*.pdf
python3 poster/deck_update_results.py     # sync slides 12, 15, 16 with the report
python3 poster/deck_update_retrain.py     # slides 8, 9, 11-14, 16 after retraining without the NAIRAS input
python3 poster/deck_fix_claims.py         # slides 1, 12, 18: claim wording, layout fixes
python3 poster/deck_add_interpretation.py # slides 14-15: inputs used, occlusion (run first on a 16-slide deck)
python3 poster/deck_polish.py             # slides 1, 9, 11, 13: both histories, attention key, ablation row
python3 src/export_web.py results/temporal_final; python3 poster/animation/build.py
python3 scripts/export_model_zoo.py; python3 poster/zoo/build.py   # Flight Model Zoo (real model internals on one unseen flight)

# MLP baselines on all six train/validation/test assignments -> results/basic/
for s in 42 1 2; do for r in 123 132 213 231 312 321; do
  python3 train_basic.py --rotation $r --seed $s
done; done
python3 src/summarize.py results/basic
python3 src/plot_basic.py results/basic/rot123_seed42
```

Rotation `123` means train P1, validate P2, test P3. Preprocessing statistics are
fitted on the training partition only; models are selected on validation (early
stopping); the test partition is only scored.

## Layout

```
README.md                  project overview, status, results, how to reproduce
requirements.txt           Python dependencies
LEARNING_LOG.md            recall cards (spaced repetition)
train_basic.py             MLP direct / residual on the 43 current-time features
src/
  data.py                  loading, caching, splits, train-only scaling
  metrics.py               RMSE / MAE / R2 / bias overall, A<=15, A>15
  summarize.py             mean ± std over rotations
  plot_basic.py            learning curves and RMSE bars
explore/                   exploration scripts e01–e04 (+ load.py helper)
figures/explore/           exploration figures
figures/results/           model result figures
results/<experiment>/      metrics, training histories, checkpoints per run
latex/                     final and preliminary reports (CVPR), references
poster/                    poster / slides (Team 5)
docs/                      course guidelines and the original proposal
data/                      raw zips, extracted JSON, caches (not committed)
```
