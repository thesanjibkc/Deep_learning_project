# Where we stopped (2026-11-27) and what to do next

Read `CLAUDE.md` first (rules, pipeline, links). This file is the status and to-do list.

## Deadlines
- Poster session: **Nov 30 / Dec 2** (slides, printed)
- Final report: **Dec 7** (4 pages + peer evaluation + references, CVPR format) → submit `Sanjib_KC_final_project_report.pdf`

## Done (2026-11-27)
- All models retrained **without NAIRAS as an input** (90 runs); statistics, significance, error analysis, attention tracking
- Analyses for **both** 1-hour and 24-hour models: group/variable importance, history flattening and swapping,
  occlusion (all models; Transformer also variable × hour), remaining-error breakdown, learning curves of all models
- Report: paper-style figures, full-width Table 1 with both histories + parameters + training time, 4 pages,
  PantherID in the author line, compiled locally (`latex/Sanjib_KC_final_project_report.pdf`)
- Slides patched (8, 9, 11–16), four interactive pages republished, README updated
- Everything committed and pushed to `origin/final-project` (main untouched)
- Report review (rated ~8/10) → overclaims fixed in report, slides 1/12/16, README, Dose Lab: title now "Temporal Networks";
  weighting "matches the MLP overall" (+0.08 vs unweighted Transformer); residual beats direct clearly for MLP, narrowly for Transformer

## To do (in this order)
1. **Open the slides in PowerPoint** and check slides 8–16 (they were patched with python-pptx; Claude cannot render them).
   Print and rehearse (~2 min per slide).
2. **Share the four pages** from each page's Share menu (tutorial, attention animation, Model Zoo, Dose Lab), otherwise the
   QR codes on slides 8, 9, 14 only work for you. Test one QR with a phone.
3. **Verify five older references** on Google Scholar: Tobiska 2016, Mertens 2013, Copeland 2017 (CARI-7A), Vaswani 2017,
   Huber 1964.
4. Read the final PDF once end to end; submit by Dec 7.
5. Optional: uncertainty bands from the 3-seed ensemble; a pure 1D-CNN baseline; email the ARMAS team about how effective
   dose is computed from the silicon measurement.

## Learning log
`LEARNING_LOG.md` has ~26 recall cards. Good next cards: what occlusion measures; why the Flat MLP is a control;
why NAIRAS is not an input; why 83% of the remaining error lies within flights.
