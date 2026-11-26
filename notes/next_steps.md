# Where we stopped (2026-11-26) and what to do next

## Deadlines
- Poster session: **Nov 30 / Dec 2** (slides, Team 5, printed)
- Final report: **Dec 7** (4 pages + peer evaluation + references, CVPR format)

## Done
- All models trained (6 rotations × 3 seeds, 1 h and 24 h), significance tests, error analysis, attention tracking
- Final report draft (no `\tbd` left), README, 16-slide deck, three interactive pages (see README)
- **Flight Model Zoo** now runs the six real networks live in the browser (ported from PyTorch, matches the saved
  predictions to < 0.01 µSv/h), with a "Meet the models" section, edits (storm, look-back, altitude, Rc), race, game
- QR codes for the Zoo added to slides 8, 9 and 14

## To do (in this order)
1. **Share the three artifacts** from each page's Share menu, otherwise the QR codes only work for the owner:
   Flight Model Zoo, Watching Attention Learn, Flight Dose Lab. Scan a QR with a phone to test. Then test the Zoo on a phone.
2. **Compile the report on Overleaf** (upload `latex/*`, plus PNGs from `figures/explore` and `figures/results` into a `figures/` folder).
   Check the 4-page limit and trim if needed (move the exploration paragraphs and figures to the appendix first).
3. **Review the slides in PowerPoint** (close and reopen `poster/Team5_Sanjib_KC_slides.pptx`, it was patched on disk), print, rehearse (~2 min per slide).
   Optional: on slide 14 add "scaling the storm changes predictions by at most 1 µSv/h".
4. **Verify five older references** on Google Scholar: Tobiska 2016, Mertens 2013, Copeland 2017 (CARI-7A), Vaswani 2017, Huber 1964.
5. Optional: email the ARMAS team about (a) how effective dose is computed from the silicon measurement and the update interval,
   (b) the flight on 2016-06-10 labelled "data" whose remaining error is +8.3 µSv/h.
6. Optional: uncertainty bands from the 3-seed ensemble; integrated gradients as a second check on attention.

## Learning log
`LEARNING_LOG.md` has ~26 recall cards. Open questions: Q23 (first reviewer question), Q24 (why within-flight noise can't be fixed),
and the Sun's two effects (GCR vs SEP; none of our flights saw an SEP).

## Reminders
- Work only on branch `final-project`; never change `main`.
- Only neural-network models are trained (XGBoost is cited from the 2025 paper).
- Regenerate the Zoo: `python3 scripts/export_model_zoo.py && python3 poster/zoo/build.py`.
- Regenerate the deck: `python3 poster/build_deck.py` (overwrites hand edits; patch with `python3 poster/deck_qr.py` instead).
