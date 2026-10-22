"""Create a readable notebook containing the exact executed proposal-model code."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "armas_proposal_models_5m.py"
OUT = ROOT / "notebooks" / "preliminary" / "ARMAS_5M_proposal_deep_learning_executed.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(True)}


source = SOURCE.read_text(encoding="utf-8")
source = source[source.find('"""', 3) + 3:].lstrip()
source = source.replace('\n\nif __name__ == "__main__":\n    main()\n', "\n")
cuts = [
    source.index("class SinusoidalPosition"),
    source.index("def parameter_count"),
    source.index("def metric_rows"),
    source.index("def plot_results"),
    source.index("def main"),
]
parts = [source[:cuts[0]]] + [source[cuts[i]:cuts[i+1]] for i in range(len(cuts)-1)] + [source[cuts[-1]:]]

cells = [
    md("""
# ARMAS Proposal Deep-Learning Study

This notebook contains the exact code used for the executed preliminary experiment.
It follows the proposal: **JMLP, CNN-MLP, direct Transformer, residual Transformer,
and weighted residual Transformer**.

P1 is training, P2 is validation for early stopping and model selection, and P3 is the
final test set. The source arrays are reversed once from current-first to chronological order.
"""),
    md("""
## 1. Imports, paths, reproducibility, and P1-only scaling

`ArrayScaler` learns medians, means, and standard deviations only from P1. This prevents
information from P2 or P3 leaking into preprocessing.
"""), code(parts[0]),
    md("""
## 2. Neural architectures

- **JMLP:** current 43-variable baseline.
- **CNN-MLP:** convolutions summarize the 12-step history.
- **Transformer:** summary token, sinusoidal positions, two attention blocks, and a
  separate current-input branch.
"""), code(parts[1]),
    md("""
## 3. Training utilities

Training uses AdamW, cosine decay, Huber loss, gradient clipping, checkpointing, and
P2 validation for early stopping. `high_weight` changes examples with ARMAS above 15 microSv/h.
"""), code(parts[2]),
    md("## 4. Metrics"), code(parts[3]),
    md("""
## 5. Scientific figures and occlusion

Occlusion sets one standardized feature group to zero, its P1 mean, and measures the
change in P2 RMSE. This is sensitivity analysis, not a causal claim.
"""), code(parts[4]),
    md("""
## 6. Complete experimental pipeline

This cell defines the six executed neural experiments, saves checkpoints and tables,
and evaluates the P2-selected checkpoints once on P3.
"""), code(parts[5]),
    md("""
## 7. Run everything

The full CPU run takes several minutes. Outputs are written to
`outputs/proposal_models_5m`. The saved CSVs and figures came from this run.
"""), code("main()"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUT)
