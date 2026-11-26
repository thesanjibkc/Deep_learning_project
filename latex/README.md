# Report sources

| File | Purpose |
|---|---|
| `Sanjib_KC_final_project_report.tex` | Final report (due Dec 7, 2026), CVPR camera-ready format |
| `Sanjib_KC_preliminary_project_report.tex` | Preliminary report (Oct 21, 2026), revised to the CVPR format |
| `references.bib` | Bibliography for both reports |
| `cvpr.sty`, `cvpr_eso.sty`, `eso-pic.sty`, `ieee_fullname.bst` | Official CVPR template files from the course |

Figures are read from `../figures/explore/` and `../figures/results/` (or a local
`figures/` folder).

## Compile

Locally:

```bash
pdflatex Sanjib_KC_final_project_report && bibtex Sanjib_KC_final_project_report && pdflatex Sanjib_KC_final_project_report && pdflatex Sanjib_KC_final_project_report
```

On Overleaf: upload this folder's files plus the PNGs from `figures/explore/` and
`figures/results/` into a `figures/` folder, and set the main document to the
final report.

Red `[TBD: ...]` markers show unfinished parts; remove the `\tbd` macro before
submission. Page limit: 4 pages of content; Peer Evaluation starts on page 5,
then References, then the appendix.
