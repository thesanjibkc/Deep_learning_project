"""Build the proposal-aligned preliminary report from executed PyTorch results."""

from pathlib import Path

import pandas as pd
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, PageBreak

import build_scientific_preliminary_report as layout


ROOT = Path(__file__).resolve().parent
DL = ROOT / "outputs" / "proposal_models_5m"
EDA = ROOT / "outputs" / "armas_5m_full_study"
PDF = ROOT / "output" / "pdf" / "Sanjib_KC_preliminary_project_report.pdf"


def fig(path, caption, height=1.75):
    return KeepTogether([
        Image(str(path), width=3.23 * inch, height=height * inch),
        layout.P(caption, "Caption"),
    ])


def build_story():
    validation_metrics = pd.read_csv(DL / "p2_validation_metrics.csv")
    metrics = pd.read_csv(DL / "p3_test_metrics.csv")
    models = pd.read_csv(DL / "model_parameter_counts.csv")
    features = pd.read_csv(DL / "p2_residual_transformer_feature_occlusion.csv")
    lags = pd.read_csv(DL / "p2_residual_transformer_lag_occlusion.csv")
    summary = pd.read_csv(EDA / "dataset_summary.csv")
    overall = metrics[metrics.Regime == "All"].set_index("Model")
    high = metrics[metrics.Regime == "ARMAS > 15"].set_index("Model")
    low = metrics[metrics.Regime == "ARMAS <= 15"].set_index("Model")
    validation_overall = validation_metrics[validation_metrics.Regime == "All"].set_index("Model")
    nairas, jmlp = overall.loc["NAIRAS V3"], overall.loc["Current-time JMLP direct"]
    w4h = high.loc["Residual Transformer + weighted Huber (w=4)"]
    gain_all = 100 * (nairas.RMSE - jmlp.RMSE) / nairas.RMSE
    gain_high = 100 * (high.loc["NAIRAS V3", "RMSE"] - w4h.RMSE) / high.loc["NAIRAS V3", "RMSE"]

    S = [
        layout.P("Abstract", "Abstract"),
        layout.P(
            "This preliminary study executes the model family specified in the approved proposal for aviation "
            "effective dose-rate estimation. The 5-minute data contain 92,415 observations in three flight-separated "
            "partitions, 35 environmental variables over 12 time steps, and eight current flight/geomagnetic variables. "
            "The completed preliminary analysis uses assignment 123: P1 fits preprocessing and neural weights, P2 "
            "controls validation and checkpoint selection, and P3 is the final test. Six PyTorch experiments compare a 43-input joint multilayer perceptron (JMLP), a "
            "CNN-MLP, a direct Transformer, and three physics-guided residual Transformers. P2 validation selected the "
            f"JMLP overall; on P3 it reached RMSE {jmlp.RMSE:.3f} versus {nairas.RMSE:.3f} microSv/h for NAIRAS V3 "
            f"({gain_all:.1f}% lower). "
            f"Weighted residual learning with weight 4 gives the best high-dose RMSE ({w4h.RMSE:.3f} versus "
            f"{high.loc['NAIRAS V3','RMSE']:.3f}; {gain_high:.1f}% lower), while slightly sacrificing overall error. "
            "Occlusion identifies the Newark, Oulu, and Thule neutron-monitor histories as the most influential temporal groups."
        ),
        layout.NextPageTemplate("Later"),
        layout.H("1", "Introduction"),
        layout.P(
            "ARMAS provides in-flight measurements of aviation effective dose rate, while NAIRAS V3 supplies a "
            "physics-based estimate [1,2]. Earlier static learning reduced error, especially where NAIRAS underpredicts "
            "large ARMAS values [3,4]. This project asks whether a neural temporal model can learn a useful correction "
            "from the preceding 55 minutes, which variables and lags drive that correction, and whether high-dose "
            "weighting improves performance above 15 microSv/h without an excessive cost elsewhere."
        ),
        layout.P(
            "The central hypothesis is tested through residual learning. If A_i is ARMAS and N_i is NAIRAS V3, the "
            "network predicts r_i=A_i-N_i and reconstructs A-hat_i=N_i+r-hat_i. This constrains the model to correct a "
            "physical estimate rather than learn dose rate entirely from scratch."
        ),
        layout.H("2", "Related Work"),
        layout.P(
            "Tobiska et al. introduced the ARMAS measurement system for global, real-time aviation radiation monitoring "
            "[1]. Mertens et al. developed and validated NAIRAS as a physics-based aircraft dose model [2]. The present "
            "study uses the machine-learning-ready dataset described by Sadykov et al., which aligns ARMAS and NAIRAS "
            "with aircraft position, neutron-monitor measurements, solar-wind variables, particle fluxes, and geomagnetic "
            "indices [3]."
        ),
        layout.P(
            "Earlier static machine learning showed that geospace properties can improve aviation-radiation nowcasts "
            "relative to uncorrected NAIRAS [4]. This project extends that work by testing whether the preceding hour adds "
            "predictive information. The temporal models follow the self-attention framework of Vaswani et al. [5], while "
            "the physics-guided variants learn a correction to NAIRAS rather than replacing the physical model."
        ),
        layout.H("3", "Model and Method"),
        layout.P(
            "Each record contains a 12 by 35 environmental tensor and eight static variables. Source arrays place the "
            "current sample at index zero; they are reversed once to chronological order from t-55 to t minutes. "
            "Vehicle ID is excluded. Median imputation and z-score normalization use P1 statistics only. The preliminary "
            "assignment is 123, where the digits give the train-validation-test order: all P1 records drive gradient "
            "fitting, P2 controls early stopping and model selection, and P3 is used only for the final test after the "
            "checkpoint has been selected."
        ),
        layout.P(
            "The residual-learning formulation is"
        ),
        layout.P(
            "r<sub>i</sub> = A<sub>i</sub> - N<sub>i</sub>,&nbsp;&nbsp; "
            "r-hat<sub>i</sub> = f<sub>theta</sub>(X<sub>i</sub>, N<sub>i</sub>),&nbsp;&nbsp; "
            "A-hat<sub>i</sub> = N<sub>i</sub> + r-hat<sub>i</sub>.&nbsp;&nbsp; (1)",
            "Equation"
        ),
    ]
    t = [["Partition", "Rows", "High dose", "Percent"]]
    for _, r in summary.iterrows():
        t.append([r.Partition, f"{int(r.Rows):,}", f"{int(r['ARMAS > 15']):,}", f"{r['High-dose percent']:.1f}"])
    S += [
        layout.table(t, [0.70*inch, 0.72*inch, 0.94*inch, 0.70*inch]),
        layout.P("Table 1. Partition sizes and fixed roles: P1 training, P2 validation, and P3 testing.", "Caption"),
        fig(EDA / "figure01_target_distribution.png", "Figure 1. ARMAS target distribution using the full P1+P2+P3 dataset. The upper tail is rare but scientifically important.", 1.62),
        layout.P(
            "<b>Model architecture.</b> The JMLP joins the eight static variables and 35 current environmental values, then applies 128- and "
            "64-unit GELU layers with normalization and dropout. The CNN-MLP applies two 1-D convolutions to the full "
            "12-step history and joins global temporal features to the current-input branch. The Transformer projects "
            "each 35-variable time step to 48 dimensions, adds sinusoidal position encoding and a learned summary token, "
            "and uses two four-head pre-normalized encoder blocks. A separate MLP embeds current inputs before the "
            "concatenated representation reaches the regression head."
        ),
        layout.P(
            "Direct models predict standardized ARMAS. Residual Transformers receive standardized NAIRAS in the current "
            "branch and predict standardized ARMAS-minus-NAIRAS. All models use AdamW, cosine learning-rate decay, "
            "Huber loss, gradient clipping at 1.0, batch size 512, at most 24 epochs, and patience 5. High-dose residual "
            "models multiply loss above 15 microSv/h by 2 or 4."
        ),
    ]
    arch = [["Model", "Parameters", "Epochs", "Target"]]
    short_names = {
        "Current-time JMLP direct":"JMLP", "Hybrid CNN-MLP direct":"CNN-MLP",
        "Direct Temporal Transformer":"Direct Transformer", "Residual Transformer":"Residual Transformer",
        "Residual Transformer + weighted Huber (w=2)":"Residual Transformer w=2",
        "Residual Transformer + weighted Huber (w=4)":"Residual Transformer w=4",
    }
    for _, r in models.iterrows():
        arch.append([short_names[r.Model], f"{int(r['Trainable parameters']):,}", int(r['Epochs run']), r.Target])
    S += [
        layout.table(arch, [1.46*inch, 0.72*inch, 0.48*inch, 0.54*inch]),
        layout.P("Table 2. Executed PyTorch architectures. Every run completed on CPU with seed 42.", "Caption"),
        fig(DL / "02_learning_curves.png", "Figure 2. P2 validation curves used for early stopping and checkpoint selection.", 1.78),
        layout.H("4", "Experiments and Results"),
    ]
    order = ["NAIRAS V3", "Current-time JMLP direct", "Hybrid CNN-MLP direct", "Direct Temporal Transformer",
             "Residual Transformer", "Residual Transformer + weighted Huber (w=2)",
             "Residual Transformer + weighted Huber (w=4)"]
    result = [["Model", "All RMSE", "High RMSE", "All R2"]]
    for name in order:
        label = short_names.get(name, name)
        result.append([label, f"{overall.loc[name,'RMSE']:.3f}", f"{high.loc[name,'RMSE']:.3f}", f"{overall.loc[name,'R2']:.3f}"])
    S += [
        layout.table(result, [1.38*inch, 0.61*inch, 0.67*inch, 0.52*inch]),
        layout.P("Table 3. P3 test results in microSv/h. Lower RMSE and higher R2 are better.", "Caption"),
        fig(DL / "01_proposal_model_rmse.png", "Figure 3. Overall and high-dose P3 test RMSE. The ranking changes when upper-tail error is emphasized.", 1.52),
        layout.P(
            f"P2 validation selected the JMLP overall (validation RMSE {validation_overall.loc['Current-time JMLP direct','RMSE']:.3f}). "
            f"Its P3 test RMSE is {jmlp.RMSE:.3f}, with MAE {jmlp.MAE:.3f} and R2={jmlp.R2:.3f}. "
            "Its improvement over NAIRAS shows that the current state contains substantial correctable structure. The "
            f"direct Transformer reaches RMSE {overall.loc['Direct Temporal Transformer','RMSE']:.3f}, and the unweighted "
            f"residual Transformer reaches {overall.loc['Residual Transformer','RMSE']:.3f}. Thus the one-hour encoder "
            "does not beat the current-only network overall on P3. The weight-2 residual Transformer has the lowest "
            f"descriptive P3 RMSE ({overall.loc['Residual Transformer + weighted Huber (w=2)','RMSE']:.3f}), but this test-set ranking is not used to revise selection. This negative comparison is retained because "
            "it directly tests the proposal rather than selecting only favorable results."
        ),
        layout.P(
            f"High-dose behavior supports the imbalance-aware hypothesis. Increasing the residual loss weight from 1 "
            f"to 4 lowers high-dose RMSE from {high.loc['Residual Transformer','RMSE']:.3f} to {w4h.RMSE:.3f} and changes "
            f"high-dose bias from {high.loc['Residual Transformer','Bias']:.3f} to {w4h.Bias:.3f} microSv/h. The cost is "
            f"an overall RMSE increase from {overall.loc['Residual Transformer','RMSE']:.3f} to "
            f"{overall.loc['Residual Transformer + weighted Huber (w=4)','RMSE']:.3f}; low-dose RMSE changes from "
            f"{low.loc['Residual Transformer','RMSE']:.3f} to {low.loc['Residual Transformer + weighted Huber (w=4)','RMSE']:.3f}."
        ),
        layout.H("4.1", "Temporal Interpretation"),
        layout.P(
            "Grouped occlusion replaces one standardized temporal variable or one complete lag with its P1 mean and "
            "measures the change in residual-Transformer P2 RMSE. This tests model dependence, not causal importance. "
            f"The largest feature increases are {features.iloc[0].Feature} ({features.iloc[0]['RMSE increase']:.3f}), "
            f"{features.iloc[1].Feature} ({features.iloc[1]['RMSE increase']:.3f}), and "
            f"{features.iloc[2].Feature} ({features.iloc[2]['RMSE increase']:.3f}) microSv/h. The strongest lag is "
            f"{int(lags.sort_values('RMSE increase', ascending=False).iloc[0]['Lag minutes'])} minutes before the current "
            "sample, but every lag effect is small relative to the full RMSE. This agrees with the limited aggregate "
            "advantage of temporal models."
        ),
        fig(DL / "04_transformer_occlusion.png", "Figure 4. Residual-Transformer temporal-feature and lag occlusion. Positive values indicate worse RMSE after removal.", 1.62),
        layout.H("5", "Conclusions"),
        layout.P(
            "The preliminary work shows that learned correction can improve the physics model: "
            "all proposal neural models except the CNN-MLP reduce overall NAIRAS RMSE. Second, temporal complexity is "
            "not automatically useful. Closely spaced environmental histories are redundant, and the 32,369-row P1 "
            "training partition may be too small for a 61,000-parameter attention model to dominate a compact JMLP. "
            "The main lesson is therefore objective-dependent: the compact JMLP is the validation-selected overall model, "
            "whereas weighted residual learning is more useful when reducing high-dose underprediction is the priority."
        ),
        layout.P(
            "These are single-seed, single-split preliminary results. P3 contains correlated measurements and only "
            f"{int(high.loc['NAIRAS V3', 'N']):,} high-dose records; observation-level metrics therefore understate uncertainty from flights or years. "
            "Occlusion can be distorted by correlated variables. Hyperparameters were intentionally compact for CPU "
            "execution. P3 must not be used to revise architecture or hyperparameters after viewing these results."
        ),
        layout.H("6", "Future Work"),
        layout.P(
            "The completed preliminary analysis reports assignment 123 (P1 train, P2 validation, P3 test). The future "
            "robustness study will repeat the frozen model protocol for the five remaining train-validation-test "
            "permutations: 132, 213, 231, 312, and 321. Each rotation will fit preprocessing and weights only on its "
            "designated training partition, use its designated validation partition for selection, and score its test "
            "partition only after selection. JMLP, direct Transformer, residual Transformer, and weighted residual "
            "Transformer will be repeated with at least three seeds. Integrated gradients and attention diagnostics will complement grouped "
            "occlusion, with all interpretations described as associations."
        ),
        layout.H("7", "Reproducibility"),
        layout.P(
            "The executed script armas_proposal_models_5m.py saves checkpoints, per-epoch histories, predictions, "
            "metrics, plots, parameter counts, occlusion tables, and a JSON run manifest. Random seed 42 and software "
            "version are recorded. The final submission will include a documented public repository URL sufficient to "
            "reproduce the reported tables and figures."
        ),
        layout.H("8", "Peer Evaluation"),
        layout.P(
            "This is an individual project. All data preparation, implementation, experiments, analysis, visualization, "
            "and report writing are assigned to Sanjib K C; therefore, there are no team peers to evaluate."
        ),
        PageBreak(),
        layout.H("", "References"),
        layout.P('[1] W. K. Tobiska et al., "Global real-time dose measurements using the Automated Radiation Measurements for Aerospace Safety system," <i>Space Weather</i>, 14, 1053-1080, 2016.', "Reference"),
        layout.P('[2] C. J. Mertens et al., "NAIRAS aircraft radiation model development, dose climatology, and initial validation," <i>Space Weather</i>, 11, 603-635, 2013.', "Reference"),
        layout.P('[3] V. M. Sadykov et al., "Machine Learning-Ready Data Sets for the Analysis and Nowcasting of Atmospheric Radiation at Aviation Altitudes," arXiv:2602.07108, 2026.', "Reference"),
        layout.P('[4] S. K C, V. M. Sadykov, and D. Kempton, "Nowcasting of Aviation Radiation Using Geospace Environment Properties: A Machine Learning Approach," arXiv:2511.15746, 2025.', "Reference"),
        layout.P('[5] A. Vaswani et al., "Attention Is All You Need," <i>NeurIPS</i>, 2017.', "Reference"),
    ]
    return S


if __name__ == "__main__":
    layout.Paper(str(PDF)).build(build_story())
    print(PDF)
