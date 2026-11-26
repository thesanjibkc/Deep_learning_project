# Learning log — active recall and spaced repetition

How to use this file: cover the answers, answer each question out loud or in writing,
then check. Review each card on the dates shown (1 day → 3 days → 7 days → 14 days).
Move a card to "Mastered" once you answer it correctly on 3 reviews in a row.

## Cards

| # | Question | Introduced | Next review |
|---|---|---|---|
| 1 | What do ARMAS and NAIRAS V3 each give us, and which one is the "truth"? | 2026-11-24 | 2026-11-25 |
| 2 | Write the residual equation and the reconstruction equation. | 2026-11-24 | 2026-11-25 |
| 3 | Why do we explore only on P1 (training), not P2/P3? | 2026-11-24 | 2026-11-25 |
| 4 | What is the high-dose regime, and what fraction of P1 is in it? | 2026-11-24 | 2026-11-25 |
| 5 | Why does binning the error by the TRUE dose make a model look biased even if it isn't? (regression to the mean) | 2026-11-24 | 2026-11-25 |
| 6 | 5M vs 1H data: how many steps, what time span, and what is identical between them? | 2026-11-24 | 2026-11-25 |
| 7 | How is an IQR band built, and where is each bin's dot drawn? | 2026-11-24 | 2026-11-25 |
| 9 | Why does ARMAS have peaks near 4.5, 9.5 and 15? (hint: Rc) | 2026-11-25 | 2026-11-26 |
| 10 | Why does binning the error by the truth vs by the prediction tell different stories? | 2026-11-25 | 2026-11-26 |
| 11 | Name three factors that set the dose, and which way each pushes it. | 2026-11-25 | 2026-11-26 |
| 12 | Why do neutron-monitor counts go DOWN when the Sun is active? | 2026-11-25 | 2026-11-26 |
| 13 | Split by row vs split by flight: what's the difference, and why does a row split cheat? | 2026-11-25 | 2026-11-26 |
| 14 | NAIRAS error vs solar cycle: too high at solar minimum, too low at higher activity. What does that say about NAIRAS? | 2026-11-25 | 2026-11-26 |
| 15 | What is a lead–lag effect? Give the Kp example from our data. | 2026-11-25 | 2026-11-26 |
| 16 | Kp vs ap: scale, cadence, and why they look identical in our heatmaps. | 2026-11-25 | 2026-11-26 |
| 17 | In our Transformer, what is a "token", and why does a flat history give attention nothing to choose from? | 2026-11-25 | 2026-11-26 |
| 18 | Static vs temporal features: how many of each, and give two examples of each. | 2026-11-25 | 2026-11-26 |
| 19 | What are Solar_NPF/SPF/APF and the "20" versions, and why are they flat even over 24 h? | 2026-11-25 | 2026-11-26 |
| 20 | Sept 2017 example: why can't the 1-hour Transformer see the storm? (use "token" and "window") | 2026-11-25 | 2026-11-26 |
| 21 | Name the model ladder from simplest to the main model, and what question each rung answers. | 2026-11-25 | 2026-11-26 |
| 22 | Train RMSE 2.2 vs test 3.7 (Random Forest): name it and explain what it means. | 2026-11-25 | 2026-11-26 |
| 23 | What is an ablation, and why does Transformer vs MLP NOT prove attention helps? | 2026-11-25 | 2026-11-26 |
| 24 | The Sun's two opposite effects on dose: name both, and which particles each involves. | 2026-11-26 | 2026-11-27 |
| 25 | Permutation importance: how does it work, and what did the correction rely on most? | 2026-11-26 | 2026-11-27 |
| 26 | Where does the remaining error live (within flight / between flights / aircraft), and what does that imply? | 2026-11-26 | 2026-11-27 |
| 8 | Narrow band entirely above zero vs wide band crossing zero: what does each mean for ML? | 2026-11-24 | 2026-11-25 |

## Answers (cover before reviewing)

1. ARMAS = measured dose rate on the aircraft (truth, target). NAIRAS V3 = physics-model estimate (baseline).
2. r = A − N ;  Â = N + f_θ(inputs). The network learns the correction r.
3. To avoid information leakage: anything we decide after looking at validation/test data biases the final evaluation.
4. A > 15 µSv/h; about 16% of P1 (5,324 of 32,369).
5. Selecting cases by high observed A also selects cases where the measurement noise happened to be positive, so A − N is positive on average even for an unbiased model. Binning by the prediction N avoids this selection effect.
9. Peaks follow geomagnetic shielding (cutoff rigidity Rc): ~4.5 → Rc≈4.9 (lower latitude, more shielding), ~9.5 → Rc≈2.4, ~15 → Rc≈1.8 (near-polar). Altitude is similar (~11 km). Also ARMAS repeats values: ~119 rows/day but only ~12 distinct values.
10. Binning by truth selects lucky-high measurements → error looks like it grows (regression to the mean). Binning by prediction asks "when NAIRAS says X, is it right on average?" — nearly yes, but with wide spread. Conclusion: NAIRAS isn't simply "too low"; it can't tell WHICH flights will be high, so its estimates are squeezed toward the middle. The extra inputs must supply that information.
11. Altitude (higher → more dose), cutoff rigidity Rc (lower → more dose; near poles), solar activity (quieter Sun → more cosmic rays → more dose). They interact: near the equator a strong shield (Rc≈12) beats high altitude (Hawaii flights at 13.6 km have the lowest dose).
12. An active Sun has a stronger solar wind and magnetic field that pushes galactic cosmic rays out of the inner solar system (solar modulation), so fewer reach Earth. Corr(sunspots, Oulu NM) = −0.76 in our data.
13. Row split = shuffle all 92,415 rows and deal them randomly into train/test, so minutes from the SAME flight land on both sides; the model sees near-copies (56% of rows repeat the previous value) of its test rows during training → test score too optimistic. Flight split = whole flights go to one partition only → test flights are truly unseen.
14. NAIRAS swings too strongly with the solar cycle (over-reacts to cosmic-ray level): at solar minimum (high cosmic rays) it overshoots, at higher activity it undershoots. Preliminary (43 matched days).
15. A cause shows up in the data before its effect: the "leader" variable changes, and the "lagger" responds hours later. In our data Kp/ap ~15 h before the observation correlate with the NAIRAS error more (ρ≈0.12) than Kp now (ρ≈0.04).
16. Kp: 0–9 quasi-logarithmic in thirds (stored ×10: 27=3−, 30=3o, 33=3+), every 3 h, from ~13 magnetometers. ap: linear version of the same 3-hourly Kp (in units of ~2 nT; Kp 3o → ap 15). Same information, different scale → identical patterns. Daily Ap = average of the 8 ap values.
17. Token = one time step (all 35 variables at that moment, turned into a vector). Attention compares tokens and weights the useful ones; if all tokens are identical (flat 1-h history), every weighting gives the same answer, so attention adds nothing.
18. 8 static (position: lat, lon, 2 altitudes; geomagnetic: geomag lat/lon, Rc, L-shell) — one value each. 35 temporal (neutron monitors, X-rays, particles, solar wind, Kp/ap/Dst, solar activity) — a history of 12 or 24 steps. Total 43 current-time features.
19. Sun's north/south/average polar magnetic field from Wilcox Solar Observatory; "20" = 20 nHz low-pass filtered to remove the yearly viewing-geometry effect. Values are 30-day averages updated every 10 days → constant within any 24-h window; they describe the slow solar-cycle state, not hour-to-hour changes.
20. The storm peaked 13–18 h before the flight, OUTSIDE the 1-hour window. All 12 tokens come from the quiet aftermath, so they look the same and attention has nothing to pick. The 24-token window still contains the storm hours.
21. NAIRAS (physics) → linear/ridge residual (is the correction just linear?) → static tree model from the 2025 paper (prior ML) → current-time MLP (nonlinear, no history) → 1D-CNN / CNN-MLP (local patterns in history) → TCN (longer receptive field) → direct Transformer (attention, predicts A) → residual Transformer (attention + physics baseline). Each rung adds ONE idea, so the gain from each idea is measurable.
22. Overfitting: the model memorizes training data (including noise) instead of learning patterns that generalize, so it is much better on training than on unseen flights. Watch train vs validation curves; use early stopping, dropout, weight decay.
23. Ablation = remove or change ONE component and measure the effect. Transformer vs MLP changes two things at once (history AND attention), so a gain can't be credited to attention. Fix: compare models that get the SAME 24-h history (flattened-history MLP, CNN, TCN vs Transformer) to isolate attention; compare MLP (now) vs MLP (flattened history) to isolate history.
24. (1) Solar modulation: an active Sun pushes away GALACTIC cosmic rays (GCR) → less dose. (2) Eruptions can release SOLAR energetic particles (SEP); only very energetic ones (GLE, hundreds of MeV+) add dose at aviation altitude, and only where the shield is weak (low Rc). Our data: NO SEP-exposed flights — Sept 2017 SEP happened while flights were at Rc 3–10 GV (blocked); Mar 2023 polar flight had a weak event with negligible >100 MeV flux; no GLE. Lesson: 'SEP occurring in space' ≠ 'flight exposed to SEP'.
25. Shuffle one feature group across test rows (breaking its link to the target) and measure how much RMSE rises; bigger rise = model relies on it more. Correction relied on altitude (+0.68) and shielding Rc (+0.62), then NAIRAS input (+0.24) and neutron monitors (+0.21); other space-weather groups ≤0.05. No single hour mattered alone (hours are redundant).
26. After the model: 81% of remaining error variance is within one flight (minute-to-minute noise), 12% between flights, 7% tied to the aircraft — above the 3% chance level, so a real instrument/aircraft effect. The model removed the error tied to Rc, altitude and storms (each now <1%). Implication: remaining error is mostly not fixable from space-weather inputs; part is instrument.
7. Cut x into 2-unit bins (0–2, 2–4, ...); dot at the bin MIDPOINT (1, 3, 5, ...). In each bin, sort the errors: 25th percentile = band bottom, median = dot, 75th = band top. Bins with <30 points skipped. Band = middle 50%.
8. Narrow and one-sided = systematic error, easy to correct. Wide and crossing zero = the binned variable alone can't tell the error direction; other inputs needed (the case for our model, panel d).
6. 5M: 12 steps × 5 min (last 55 min). 1H: 24 steps × 1 h (last 23 h). Same rows, same times, same ARMAS/NAIRAS, same current (index 0) values.

## Mastered

(none yet)
