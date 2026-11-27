# Report sources

| File | Purpose |
|---|---|
| `Sanjib_KC_final_project_report.tex` | Final report (due Dec 7, 2026), CVPR camera-ready format |
| `Sanjib_KC_preliminary_project_report.tex` | Preliminary report (Oct 21, 2026), revised to the CVPR format |
| `references.bib` | Bibliography for both reports |
| `cvpr.sty`, `cvpr_eso.sty`, `eso-pic.sty`, `ieee_fullname.bst` | Official CVPR template files from the course |

Figures are the paper-style PDFs in `../figures/paper/` (or a local `figures/`
folder), made by `python3 src/paper_figures.py` in CVPR style (no titles inside images,
Times text, drawn at the printed width). Poster/slide figures live in `figures/explore`
and `figures/results` and are not used by the report.

## Compile

Locally:

```bash
pdflatex Sanjib_KC_final_project_report && bibtex Sanjib_KC_final_project_report && pdflatex Sanjib_KC_final_project_report && pdflatex Sanjib_KC_final_project_report
```

On Overleaf: upload this folder's files plus the PDFs from `figures/paper/` into a
`figures/` folder, and set the main document to the final report.

Page limit: 4 pages of content; Peer Evaluation starts on page 5,
then References, then the appendix.
