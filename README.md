# Physics-Guided Temporal Transformer for Aviation Effective Dose-Rate Estimation

CSC 8851 Deep Learning, Fall 2026 · Sanjib K C · Team 5

Neural networks that learn a correction to the NAIRAS V3 physics model from aircraft
position, geomagnetic shielding, and the recent history of solar and geomagnetic
conditions, evaluated against in-flight ARMAS measurements.

**Final report:** [`latex/Sanjib_KC_final_project_report.tex`](latex/Sanjib_KC_final_project_report.tex)
(CVPR format; red `[TBD]` markers show unfinished parts)
**Poster / slides:** [`poster/`](poster/)
**Learning notes:** [`LEARNING_LOG.md`](LEARNING_LOG.md)

---

## Status (updated 2026-11-25)

| Step | Status |
|---|---|
| Exploration 1–4 (where NAIRAS fails, maps, solar cycle, history) | ✅ done, in report |
| MLP direct and residual, 6 rotations × 3 seeds | ✅ final, in report (Table 3) |
| Temporal models: 1D-CNN, CNN–MLP, flattened-history MLP, TCN, Transformer (5M and 1H) | ⏳ next |
| Weighted Huber loss for A > 15 | ⏳ |
| Attention maps and integrated gradients | ⏳ |
| 3 seeds per model | ✅ MLP · ⏳ temporal models |
| Poster / slides (12–16) | ⏳ |
| Final report | 🟡 intro, related work, data, exploration done |

## Current results (test set, mean ± std over 6 rotations × 3 seeds)

| Model | RMSE all (µSv/h) | RMSE A > 15 | Gain vs NAIRAS (all / high) |
|---|---|---|---|
| NAIRAS V3 | 4.044 ± 0.040 | 6.845 ± 0.277 | — |
| XGBoost (K C et al. 2025, cited) | 3.671 ± 0.061 | not reported | 9.3% |
| MLP, direct | 3.842 ± 0.096 | 6.627 ± 0.203 | 5.0% / 3.2% |
| MLP, residual | 3.737 ± 0.032 | 6.311 ± 0.168 | 7.6% / 7.8% |

---

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
