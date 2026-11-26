# Related work and next steps (verified 2026-11-26)

Every paper below was matched to a DOI (Crossref) or an official proceedings / arXiv page.
BibTeX for all of them is in `latex/references.bib`.

| Theme | Papers | How they relate to this project |
|---|---|---|
| Physics + ML hybrids | Karpatne 2017; Willard 2022; Reichstein 2019 | Support the residual design (prediction = NAIRAS + network) |
| Physics-informed nets | Raissi 2019; Karniadakis 2021 | Contrast: physics in the loss, ours is physics in the output |
| Space-weather sequence models | Camporeale 2019; Gruet 2018 (LSTM, Dst); Collado-Villaverde 2021 (CNN+LSTM, SYM-H); Abduallah 2024 (Transformer + uncertainty) | Precedents for feeding solar-wind history to deep sequence models |
| Aviation dose ML | Sadykov 2026 (dataset); K C 2025 (static ML); Wang 2024 (NN surrogate of simulation) | Data source, our static baseline, and a contrast (simulation vs measurements) |
| Interpretability | Lim 2021 (TFT); Jain & Wallace 2019; Wiegreffe & Pinter 2019; Sundararajan 2017 (IG) | Attention needs a second check; use integrated gradients |
| Imbalance and uncertainty | Yang 2021 (DIR); Ribeiro & Moniz 2020 (SERA); Lakshminarayanan 2017 (ensembles); Gal 2016 (MC dropout) | High-dose regime; error bars for a safety application |
| Advanced directions | Li 2021 (Fourier neural operator); Das 2024 (TimesFM); Roy 2025 (Surya, preprint) | Surrogate physics; pretrained time-series or heliophysics backbones |

## Next steps a reviewer would value
1. Deep ensemble (we already have 3 seeds) to give each prediction an uncertainty band; check calibration above 15 µSv/h.
2. Integrated gradients to cross-check attention (the lead–lag claim).
3. Imbalance-aware training (label-distribution smoothing) and SERA for the high-dose tail.
4. Pretrain the history encoder on long OMNI / neutron-monitor archives, then fine-tune on ARMAS.

Not verified, so not cited: a UPB Scientific Bulletin paper on DNNs vs CARI-7A; Chronos (arXiv 2403.07815).
Older references kept from the preliminary report (Tobiska 2016, Mertens 2013, Copeland 2017, Vaswani 2017,
Huber 1964) were not part of this check; confirm them on Google Scholar before submission.
