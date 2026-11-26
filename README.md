# Physics-Guided Temporal Transformer for Aviation Effective Dose-Rate Estimation

CSC 8851 Deep Learning, Fall 2026 · Sanjib K C · Team 5

Neural networks that learn a correction to the NAIRAS V3 physics model from aircraft
position, geomagnetic shielding, and the recent history of solar and geomagnetic
conditions, evaluated against in-flight ARMAS measurements.

**Final report:** [`latex/Sanjib_KC_final_project_report.tex`](latex/Sanjib_KC_final_project_report.tex)
(CVPR format; red `[TBD]` markers show unfinished parts)
**Poster / slides:** [`poster/`](poster/)
**Code (this branch):** https://github.com/thesanjibkc/Deep_learning_project/tree/final-project  
**Learning notes:** [`LEARNING_LOG.md`](LEARNING_LOG.md)

---

## Status (updated 2026-11-26)

| Step | Status |
|---|---|
| Exploration 1–4 (where NAIRAS fails, maps, solar cycle, history) | ✅ in report |
| MLP direct and residual, 6 rotations × 3 seeds | ✅ |
| Regularization tuning (validation of rotation 123 only) | ✅ width 16, dropout 0.3, weight decay 0.01 |
| Temporal models (Flat MLP, CNN–MLP, TCN, Transformer, direct Transformer), 1 h and 24 h, 6 rotations × 3 seeds | ✅ |
| High-dose weighted Transformer (Huber, w = 4) | ✅ |
| Seed ensembles + paired flight-bootstrap significance | ✅ `results/final/` |
| Permutation importance + remaining-error analysis (6 rotations) | ✅ `results/explain/` |
| Epoch-by-epoch attention tracking | ✅ `results/tracking/` |
| Interactive page, training animation, Flight Model Zoo | ✅ `poster/web/`, `poster/animation/`, `poster/zoo/` |
| Final report | 🟡 complete draft; check page limit on Overleaf |
| Slides (12–16) | 🟡 slide 3 done |

## Final results (test set, mean ± std over 18 runs = 6 rotations × 3 seeds)

| Model | RMSE all (µSv/h) | RMSE A > 15 | Gain vs NAIRAS (all / high) |
|---|---|---|---|
| NAIRAS V3 | 4.044 ± 0.040 | 6.845 ± 0.277 | — |
| XGBoost (K C et al. 2025, cited, static data) | 3.671 ± 0.061 | not reported | 9.3% |
| MLP, direct | 3.842 ± 0.096 | 6.627 ± 0.203 | 5.0% / 3.2% |
| MLP (no history) | 3.737 ± 0.032 | 6.311 ± 0.168 | 7.6% / 7.8% |
| Flat MLP, 24 h | 3.906 ± 0.070 | 6.731 ± 0.264 | 3.4% / 1.7% |
| CNN–MLP, 24 h | 3.661 ± 0.043 | 6.212 ± 0.261 | 9.5% / 9.2% |
| **TCN, 24 h** | **3.659 ± 0.029** | 6.173 ± 0.193 | 9.5% / 9.8% |
| Transformer, direct, 24 h | 3.712 ± 0.054 | 6.417 ± 0.230 | 8.2% / 6.3% |
| Transformer, 24 h | 3.667 ± 0.034 | 6.309 ± 0.148 | 9.3% / 7.8% |
| Transformer, 1 h | 3.663 ± 0.039 | 6.261 ± 0.125 | 9.4% / 8.5% |
| **Transformer, 1 h, high-dose weight 4** | 3.762 ± 0.080 | **5.563 ± 0.233** | 7.0% / **18.7%** |

Seed ensembles (average of 3 seeds) lower every model by about 0.03 (Transformer 1 h: 3.628).
Significance (`results/final/significance.csv`): temporal models beat the no-history MLP overall
in all 6 rotations; residual beats direct; 1 h vs 24 h shows no difference; the weighted
Transformer matches the MLP overall while cutting high-dose RMSE by 0.78 µSv/h.

**Interactive:** [Flight Dose Lab](https://claude.ai/artifact/X3saE5EHSytJkidqedoo52) ·
[Watching Attention Learn](https://claude.ai/artifact/SoxLwXhTsnEeFPNjU8EjjC) (private links; share from the page).

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
python3 src/final_stats.py             # seed ensembles + flight bootstrap
for r in 123 132 213 231 312 321; do python3 src/explain_errors.py 1H $r 42; done
python3 scripts/track_training.py      # epoch-by-epoch attention
python3 src/plot_results.py results/temporal_final; python3 src/plot_tracking.py; python3 src/plot_explain.py
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
