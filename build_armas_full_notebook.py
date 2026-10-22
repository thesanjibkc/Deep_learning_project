"""Create a beginner-friendly notebook from the tested full-study script."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "armas_5m_full_study.py"
OUT = ROOT / "notebooks" / "preliminary" / "ARMAS_5M_full_preliminary_study.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


source = SOURCE.read_text(encoding="utf-8")
docstring_end = source.find('"""', 3) + 3
source = source[docstring_end:].lstrip()
source = source.replace('\n\nif __name__ == "__main__":\n    main()\n', "\n")

pattern = re.compile(r"\n?# -{20,}\n# ([^\n]+)\n# -{20,}\n")
matches = list(pattern.finditer(source))

descriptions = {
    "1. Paths and feature names": "Define the dataset location, output folder, threshold, and source variables.",
    "2. Feature construction": "Create current-time, full-history, and transparent temporal-summary features.",
    "3. Metrics and model helpers": "Define evaluation and model functions. Preprocessing is fitted on P1 only.",
    "4. Exploratory analysis": "Create data-quality tables and four EDA figures before fitting predictive models.",
    "5. Validation experiments": "Fit baselines, residual models, ablations, nonlinear models, and high-dose weights.",
    "6. Main program": "Run the study. Tree ensembles, neural networks, and permutation importance take several minutes.",
}

cells = [
    markdown(
        """
# ARMAS 5 Minute Full Preliminary Study

This notebook is the readable, reproducible version of the preliminary project analysis.

The experimental contract is fixed:

1. **P1** fits preprocessing and model parameters.
2. **P2** is used for preliminary validation and model selection.
3. **P3** is the final test partition and is scored after P2 model comparison is complete.

The notebook performs complete exploratory analysis, linear and nonlinear baselines,
physics-guided residual learning, temporal-window and feature-group ablations,
high-dose weighting, bootstrap uncertainty, dose-bin analysis, and permutation importance.
No 1-hour files are used.
"""
    ),
    markdown(
        """
## Before running

Place this notebook in the project folder or update `DATA_DIR` in the first code section.
Install the packages listed in `requirements-preliminary.txt`. Start with a CPU runtime;
the final PyTorch Transformer notebook can later be run on a Colab GPU.
"""
    ),
]

prefix = source[: matches[0].start()]
cells.append(markdown("## Imports"))
cells.append(code(prefix))

for index, match in enumerate(matches):
    title = match.group(1)
    start = match.end()
    end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
    section_code = source[start:end]
    cells.append(markdown(f"## {title}\n\n{descriptions.get(title, '')}"))
    cells.append(code(section_code))

cells.append(
    markdown(
        """
## Run the complete analysis

The final cell writes all CSV tables and PNG figures beneath
`outputs/armas_5m_full_study`. Read the printed P2 validation and P3 test summaries,
then inspect the saved figures. Do not revise model choices after viewing P3 results.
"""
    )
)
cells.append(code("main()"))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUT)
