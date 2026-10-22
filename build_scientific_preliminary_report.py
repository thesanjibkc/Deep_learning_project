"""Build the expanded scientific ARMAS preliminary report."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "armas_5m_full_study"
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / "Sanjib_KC_preliminary_project_report.pdf"

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2E5EAA")
PALE = colors.HexColor("#EAF1F8")
GRID = colors.HexColor("#D9D9D9")
TEXT = colors.HexColor("#111111")
RED = colors.HexColor("#9B2226")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawCentredString(letter[0] / 2, 0.33 * inch, str(doc.page))
    canvas.restoreState()


def first_page(canvas, doc):
    footer(canvas, doc)
    canvas.saveState()
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica-Bold", 16.2)
    canvas.drawCentredString(letter[0] / 2, 10.42 * inch,
                             "Physics Guided Temporal Transformer for Aviation")
    canvas.drawCentredString(letter[0] / 2, 10.14 * inch,
                             "Effective Dose Rate Estimation")
    canvas.setFont("Helvetica", 10)
    canvas.drawCentredString(letter[0] / 2, 9.80 * inch, "Sanjib K C")
    canvas.setFont("Helvetica", 8.4)
    canvas.drawCentredString(letter[0] / 2, 9.59 * inch,
                             "Department of Physics and Astronomy  Georgia State University")
    canvas.drawCentredString(letter[0] / 2, 9.40 * inch,
                             "CSC 8851 Deep Learning  Fall 2026  Preliminary Project Report")
    canvas.drawCentredString(letter[0] / 2, 9.21 * inch,
                             "Email: [add email]    PantherID: [add PantherID]")
    canvas.restoreState()


class Paper(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename, pagesize=letter,
            leftMargin=0.55 * inch, rightMargin=0.55 * inch,
            topMargin=0.55 * inch, bottomMargin=0.60 * inch,
            title="Physics Guided Temporal Transformer for Aviation Effective Dose Rate Estimation",
            author="Sanjib K C",
            subject="Expanded CSC 8851 preliminary project report",
        )
        gutter = 0.24 * inch
        width = (letter[0] - self.leftMargin - self.rightMargin - gutter) / 2
        first_frames = [
            Frame(self.leftMargin, self.bottomMargin, width, 8.43 * inch, id="first-left",
                  leftPadding=0, rightPadding=0.04 * inch, topPadding=0, bottomPadding=0),
            Frame(self.leftMargin + width + gutter, self.bottomMargin, width, 8.43 * inch, id="first-right",
                  leftPadding=0.04 * inch, rightPadding=0, topPadding=0, bottomPadding=0),
        ]
        later_frames = [
            Frame(self.leftMargin, self.bottomMargin, width, 9.76 * inch, id="left",
                  leftPadding=0, rightPadding=0.04 * inch, topPadding=0, bottomPadding=0),
            Frame(self.leftMargin + width + gutter, self.bottomMargin, width, 9.76 * inch, id="right",
                  leftPadding=0.04 * inch, rightPadding=0, topPadding=0, bottomPadding=0),
        ]
        self.addPageTemplates([
            PageTemplate(id="First", frames=first_frames, onPage=first_page),
            PageTemplate(id="Later", frames=later_frames, onPage=footer),
        ])


base = getSampleStyleSheet()
base.add(ParagraphStyle(name="Body", parent=base["BodyText"], fontName="Times-Roman",
                        fontSize=8.25, leading=9.85, alignment=TA_JUSTIFY,
                        textColor=TEXT, spaceAfter=4.1))
base.add(ParagraphStyle(name="Heading", parent=base["Heading2"], fontName="Helvetica-Bold",
                        fontSize=10.2, leading=12, textColor=colors.black,
                        spaceBefore=6.5, spaceAfter=2.7, keepWithNext=True))
base.add(ParagraphStyle(name="Subheading", parent=base["Heading3"], fontName="Helvetica-Bold",
                        fontSize=8.6, leading=10.2, textColor=colors.black,
                        spaceBefore=4.5, spaceAfter=2, keepWithNext=True))
base.add(ParagraphStyle(name="Abstract", parent=base["Heading2"], fontName="Helvetica-Bold",
                        fontSize=9.2, leading=11, alignment=TA_CENTER,
                        textColor=colors.black, spaceAfter=3))
base.add(ParagraphStyle(name="Caption", parent=base["BodyText"], fontName="Times-Roman",
                        fontSize=7.25, leading=8.5, alignment=TA_LEFT,
                        textColor=TEXT, spaceBefore=2, spaceAfter=4.5))
base.add(ParagraphStyle(name="Equation", parent=base["BodyText"], fontName="Times-Italic",
                        fontSize=8.8, leading=10.5, alignment=TA_CENTER,
                        spaceBefore=3, spaceAfter=4))
base.add(ParagraphStyle(name="Reference", parent=base["BodyText"], fontName="Times-Roman",
                        fontSize=7.55, leading=9, leftIndent=12, firstLineIndent=-12,
                        alignment=TA_LEFT, spaceAfter=3))
base.add(ParagraphStyle(name="Cell", parent=base["BodyText"], fontName="Helvetica",
                        fontSize=6.8, leading=8, alignment=TA_LEFT, textColor=TEXT))
base.add(ParagraphStyle(name="CellCenter", parent=base["BodyText"], fontName="Helvetica",
                        fontSize=6.8, leading=8, alignment=TA_CENTER, textColor=TEXT))
base.add(ParagraphStyle(name="CellHeader", parent=base["BodyText"], fontName="Helvetica-Bold",
                        fontSize=6.8, leading=8, alignment=TA_CENTER, textColor=colors.white))


def P(text, style="Body"):
    return Paragraph(text, base[style])


def H(number, title):
    prefix = f"{number}  " if number else ""
    return P(prefix + title, "Heading")


def figure(filename, caption, height=1.72):
    return KeepTogether([
        Image(str(DATA / filename), width=3.23 * inch, height=height * inch),
        P(caption, "Caption"),
    ])


def table(data, widths, font_size=6.8):
    converted = []
    for row_index, row in enumerate(data):
        converted.append([
            P(str(value), "CellHeader" if row_index == 0 else ("CellCenter" if column_index > 0 else "Cell"))
            for column_index, value in enumerate(row)
        ])
    result = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ]
    for row in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, row), (-1, row), PALE))
    result.setStyle(TableStyle(style))
    return result


def build_story():
    summary = pd.read_csv(DATA / "dataset_summary.csv")
    metrics = pd.read_csv(DATA / "p3_test_metrics.csv")
    weights = pd.read_csv(DATA / "p2_weight_sweep.csv")
    windows = pd.read_csv(DATA / "p2_temporal_window_ablation.csv")
    ablation = pd.read_csv(DATA / "p2_feature_ablation.csv")
    bootstrap = pd.read_csv(DATA / "paired_bootstrap_vs_nairas.csv").iloc[0]
    importance = pd.read_csv(DATA / "permutation_importance.csv")

    all_metrics = metrics[metrics["Regime"] == "All"].set_index("Model")
    high_metrics = metrics[metrics["Regime"] == "ARMAS > 15"].set_index("Model")
    nairas = all_metrics.loc["NAIRAS V3"]
    static = all_metrics.loc["Static ridge residual"]
    static_high = high_metrics.loc["Static ridge residual"]
    w4 = all_metrics.loc["Weighted residual boosting w=4"]
    w4_high = high_metrics.loc["Weighted residual boosting w=4"]
    overall_gain = 100 * (nairas.RMSE - static.RMSE) / nairas.RMSE
    high_gain = 100 * (high_metrics.loc["NAIRAS V3", "RMSE"] - w4_high.RMSE) / high_metrics.loc["NAIRAS V3", "RMSE"]

    story = [
        P("Abstract", "Abstract"),
        P(
            "This project tests whether one hour of solar-terrestrial history can improve aviation effective dose-rate "
            "estimates. A physics-guided model predicts the residual between in-flight ARMAS measurements and NAIRAS "
            "Version 3. The available 5-minute data contain 92,415 observations in three flight-separated partitions, "
            "35 environmental variables at 12 time steps, and eight current flight and geomagnetic variables. P1 fits "
            "all models, P2 supports validation and selection, and P3 supplies the final test. A static ridge residual correction "
            f"reduced P3 test RMSE from {nairas.RMSE:.3f} to {static.RMSE:.3f} microSv/h ({overall_gain:.1f} percent). "
            f"High-dose weighting reduced RMSE above 15 microSv/h from {high_metrics.loc['NAIRAS V3','RMSE']:.3f} to "
            f"{w4_high.RMSE:.3f} ({high_gain:.1f} percent) but increased lower-dose error. Linear temporal windows did "
            "not improve overall RMSE, which makes temporal value an open hypothesis for the final Transformer rather "
            "than a preliminary conclusion."
        ),
        H("1", "Introduction"),
        P(
            "Aircraft occupants receive more ionizing radiation at cruising altitude because the atmosphere provides "
            "less shielding. ARMAS measures effective dose rate aboard aircraft, whereas NAIRAS estimates it from a "
            "physics-based model [1, 2]. A useful estimator must work across ordinary conditions and the rarer upper tail "
            "that matters most for exposure assessment. In this study, high radiation is defined as ARMAS greater than "
            "15 microSv/h, following the approved proposal."
        ),
        P(
            "The project asks three questions. First, can a learned correction improve NAIRAS V3? Second, does the "
            "preceding hour contain predictive information beyond the current flight state? Third, can imbalance-aware "
            "training reduce high-dose error without unacceptable degradation elsewhere? The preliminary work answers "
            "the first question with linear and nonlinear baselines and defines controlled tests for the remaining two."
        ),
        H("2", "Related Work"),
        P(
            "Tobiska et al. described the ARMAS measurement system [1], and Mertens et al. presented the NAIRAS aircraft "
            "radiation model and validation framework [2]. The machine-learning-ready dataset joins ARMAS, NAIRAS, "
            "aircraft position, and solar-terrestrial variables [3]. Prior static machine-learning work showed that "
            "geospace properties can improve aviation-radiation nowcasts over the uncorrected physics estimate [4]."
        ),
        P(
            "Self-attention can relate every time step in a sequence to every other step [5], making a Transformer a "
            "natural candidate for the 12-sample environmental history. Integrated gradients [6] and permutation tests "
            "can identify inputs associated with predictions. These methods describe model sensitivity; they do not "
            "establish a causal physical mechanism."
        ),
        NextPageTemplate("Later"),
        H("3", "Model and Method"),
        P("3.1  Data and leakage control", "Subheading"),
        P(
            "Only ARMAS_TIMESERIES_5M files P1, P2, and P3 are used. Source arrays store the current value at index zero; "
            "the loader reverses each array once to obtain a chronological sequence from 55 minutes before observation "
            "to current time. P1 fits imputers, scalers, and model parameters. P2 is the validation partition used for "
            "selection. P3 is scored once as the final test after selection is complete."
        ),
    ]

    summary_table = [["Partition", "Rows", "High dose", "Percent"]]
    for _, row in summary.iterrows():
        summary_table.append([row.Partition, f"{int(row.Rows):,}", f"{int(row['ARMAS > 15']):,}",
                              f"{row['High-dose percent']:.1f}"])
    story += [
        table(summary_table, [0.70 * inch, 0.72 * inch, 0.94 * inch, 0.70 * inch]),
        P("Table 1. Partition size and high-dose prevalence for P1 train, P2 validation, and P3 test.", "Caption"),
        P(
            "All current-time inputs and targets are complete in the supplied files. The partitions have similar means, "
            "medians, and high-dose prevalence, although this does not guarantee identical joint feature distributions. "
            "P1 contains three duplicate timestamps; the records are retained because timestamps alone do not identify "
            "a flight observation and the partitions may contain simultaneous aircraft measurements."
        ),
        P("3.2  Residual target and candidate models", "Subheading"),
        P("Let A<sub>i</sub> be ARMAS and N<sub>i</sub> be NAIRAS V3. The physics-guided model learns a correction:") ,
        P("r<sub>i</sub> = A<sub>i</sub> - N<sub>i</sub>,&nbsp;&nbsp; r-hat<sub>i</sub> = f<sub>theta</sub>(X<sub>i</sub>, N<sub>i</sub>),&nbsp;&nbsp; A-hat<sub>i</sub> = N<sub>i</sub> + r-hat<sub>i</sub>.&nbsp;&nbsp; (1)", "Equation"),
        P(
            "The comparison includes the P1 mean, uncorrected NAIRAS V3, direct ridge and multilayer perceptron models, "
            "ridge residual corrections, Extra Trees, and gradient boosting with high-dose weights 1, 2, and 4. Temporal "
            "inputs are represented either by all 12 samples or by per-variable mean, standard deviation, minimum, "
            "maximum, change, and slope. The final model will replace these summaries with a compact Transformer encoder."
        ),
        P("3.3  Evaluation and uncertainty", "Subheading"),
        P(
            "MAE, RMSE, R squared, and bias are reported overall and separately below and above 15 microSv/h. A "
            "600-replicate paired bootstrap resamples the same P2 observations for the selected model and NAIRAS V3. "
            "Temporal context is ablated at 1, 3, 6, and 12 steps. Feature groups are ablated with the same ridge model "
            "so architecture changes do not confound the input comparison."
        ),
        H("4", "Exploratory Data Analysis"),
        figure("figure01_target_distribution.png",
               "Figure 1. ARMAS distributions across flight-separated partitions. Values above 30 microSv/h are clipped only for display.", 1.66),
        P(
            "The multimodal target shape appears in all three partitions. P2 contains 4,983 high-dose observations "
            "(16.2 percent), while only 80 observations exceed 30 microSv/h. This tail imbalance means a good overall "
            "score can coexist with large errors at the most extreme doses."
        ),
        figure("figure02_nairas_relationship.png",
               "Figure 2. P2 relationship between observed ARMAS, NAIRAS V3, and their residual. The residual becomes increasingly positive at high ARMAS.", 1.42),
        P(
            "NAIRAS follows the broad dose-rate pattern but increasingly underestimates the upper tail. In P2, its bias "
            f"above 15 microSv/h is {high_metrics.loc['NAIRAS V3','Bias']:.3f} microSv/h. This structure supports residual "
            "learning and requires a separate high-dose objective."
        ),
        figure("figure04_lag_correlation.png",
               "Figure 3. P1 feature-by-lag correlations with ARMAS minus NAIRAS. Neutron-monitor variables have the strongest marginal associations.", 1.84),
        P(
            "The largest absolute marginal correlation is approximately 0.284 for neutron-monitor measurements. Similar "
            "values occur at many lags, indicating strong temporal redundancy. A Transformer may still detect nonlinear "
            "changes or interactions, but a lag heatmap alone does not demonstrate that an hour of history adds predictive value."
        ),
        H("5", "Experiments and Results"),
        P("5.1  Main P2 comparison", "Subheading"),
    ]

    chosen = [
        "NAIRAS V3", "Static ridge residual", "Current ridge residual",
        "Full-history ridge residual", "Current Extra Trees direct",
        "Weighted residual boosting w=2", "Weighted residual boosting w=4",
        "Current MLP direct", "Temporal-summary MLP residual",
    ]
    result_table = [["Model", "All RMSE", "High RMSE"]]
    short = {
        "NAIRAS V3": "NAIRAS V3", "Static ridge residual": "Static ridge residual",
        "Current ridge residual": "Current ridge residual", "Full-history ridge residual": "Full-history ridge",
        "Current Extra Trees direct": "Current Extra Trees", "Weighted residual boosting w=2": "Weighted boost w=2",
        "Weighted residual boosting w=4": "Weighted boost w=4", "Current MLP direct": "Current MLP",
        "Temporal-summary MLP residual": "Temporal MLP residual",
    }
    for name in chosen:
        result_table.append([short[name], f"{all_metrics.loc[name,'RMSE']:.3f}", f"{high_metrics.loc[name,'RMSE']:.3f}"])
    story += [
        table(result_table, [1.66 * inch, 0.76 * inch, 0.76 * inch]),
        P("Table 2. P2 validation RMSE in microSv/h. Lower is better. No P3 metric has been computed.", "Caption"),
        P(
            f"The static ridge residual model achieved the lowest overall RMSE, {static.RMSE:.3f}, compared with "
            f"{nairas.RMSE:.3f} for NAIRAS V3. The paired bootstrap estimates an RMSE difference of "
            f"{bootstrap['Mean RMSE difference']:.3f} with a 95 percent interval from {bootstrap['CI low']:.3f} to "
            f"{bootstrap['CI high']:.3f} microSv/h. Because the entire interval is below zero, the P2 improvement is "
            "stable under resampling of observations. The interval does not address dependence among observations from "
            "the same flight, which remains a limitation."
        ),
        figure("figure05_model_comparison.png",
               "Figure 4. Overall and high-dose P2 RMSE for the strongest preliminary models. Ranking changes when high-dose performance is emphasized.", 1.63),
        P(
            "The neural baselines reached the 90-iteration cap without convergence and performed worse than the linear "
            "and tree models. These runs diagnose an unfinished optimization problem; they do not reject the proposed "
            "Transformer. Final neural comparisons will use learning curves, validation-based early stopping, repeated "
            "seeds, and a GPU training budget."
        ),
        P("5.2  Feature and temporal ablations", "Subheading"),
    ]

    ab_all = ablation[ablation["Regime"] == "All"].set_index("Ablation")
    window_all = windows[windows["Regime"] == "All"].set_index("Steps")
    ab_table = [
        ["Input", "P2 RMSE"],
        ["Static + NAIRAS", f"{ab_all.loc['Static + NAIRAS','RMSE']:.3f}"],
        ["Environmental current + NAIRAS", f"{ab_all.loc['Environmental current + NAIRAS','RMSE']:.3f}"],
        ["Static + current + NAIRAS", f"{ab_all.loc['Static + current + NAIRAS','RMSE']:.3f}"],
        ["Temporal summaries + NAIRAS", f"{ab_all.loc['Temporal summaries + NAIRAS','RMSE']:.3f}"],
        ["Full 12-step history + NAIRAS", f"{ab_all.loc['Full 12-step history + NAIRAS','RMSE']:.3f}"],
    ]
    story += [
        table(ab_table, [2.34 * inch, 0.84 * inch]),
        P("Table 3. Ridge residual feature ablation. The same preprocessing and regularization are used in every row.", "Caption"),
        P(
            "Flight and geomagnetic variables plus NAIRAS are sufficient for the best linear result. Adding current "
            "environmental values changes RMSE only from 3.705 to 3.712, and the full history increases it to 3.762. "
            "The temporal-window ablation shows the same pattern: overall RMSE increases from 3.712 at the current step "
            "to 3.762 with 55 minutes of context. High-dose RMSE changes little, from 6.362 to 6.340."
        ),
        figure("figure06_temporal_window_ablation.png",
               "Figure 5. Linear residual performance as temporal context grows from the current step to 55 minutes. Added history does not improve overall P2 RMSE.", 1.70),
        P(
            "This result narrows the final research question. The Transformer must outperform a strong static residual "
            "baseline and demonstrate gain beyond redundant lag measurements. A temporal model that merely matches the "
            "static ridge result would not justify its additional complexity."
        ),
        P("5.3  High-dose weighting", "Subheading"),
        figure("figure07_weight_tradeoff.png",
               "Figure 6. High-dose weighting improves upper-tail RMSE while increasing lower-dose and overall error.", 1.70),
        P(
            f"Weight 4 lowers high-dose RMSE to {w4_high.RMSE:.3f} microSv/h and reduces high-dose bias from "
            f"{high_metrics.loc['NAIRAS V3','Bias']:.3f} to {w4_high.Bias:.3f}. Its overall RMSE is {w4.RMSE:.3f}, "
            f"which remains {100*(nairas.RMSE-w4.RMSE)/nairas.RMSE:.1f} percent better than NAIRAS but is "
            f"{100*(w4.RMSE-static.RMSE)/static.RMSE:.1f} percent worse than the static ridge residual. Weight 2 offers "
            "a milder compromise. The final selection should state which operational error matters most rather than "
            "choosing a weight from a single aggregate score."
        ),
        P("5.4  Error structure and interpretation", "Subheading"),
        figure("figure09_dose_bin_rmse.png",
               "Figure 7. P2 RMSE by observed dose-rate bin for NAIRAS V3 and the selected overall model. Error rises sharply in the sparse upper tail.", 1.68),
        P(
            "The static residual model improves RMSE in every displayed dose bin, but errors remain large above 20 "
            "microSv/h. Only 80 P2 observations exceed 30 microSv/h, where RMSE is 17.815 microSv/h. Tail conclusions "
            "therefore have high sampling uncertainty and should be reported with counts and resampling intervals."
        ),
        figure("figure10_permutation_importance.png",
               "Figure 8. P2 permutation importance for the best-overall weighted boosting configuration. Altitude dominates the learned correction.", 1.92),
        P(
            f"Permuting GPS altitude increases residual RMSE by {importance.iloc[0]['RMSE increase']:.3f} microSv/h, "
            f"followed by barometric altitude ({importance.iloc[1]['RMSE increase']:.3f}). Neutron-monitor, solar-polar-field, "
            "latitude, cutoff-rigidity, and L-shell variables also contribute. These are predictive associations and may "
            "reflect correlated physical conditions. Permutation importance does not isolate causal effects."
        ),
        H("6", "Conclusions"),
        P(
            "The preliminary work establishes a reproducible loader, complete EDA, leakage-controlled evaluation, "
            "multiple model families, uncertainty analysis, feature ablations, temporal-window tests, and high-dose "
            "trade-off analysis. Residual learning improves NAIRAS V3 on P2, but the simplest static residual model is "
            "currently strongest overall. High-dose weighting addresses the main systematic failure, underprediction in "
            "the upper tail, at a measurable cost below the threshold."
        ),
        P(
            "The evidence does not yet support the claim that one hour of history improves prediction. The final study "
            "will test whether a Transformer extracts nonlinear temporal structure missed by linear windows and summaries. "
            "Architectures and loss weights will be frozen using P2, trained with multiple seeds, and evaluated once on "
            "P3. Final interpretation will combine attention, integrated gradients, lag ablation, and grouped permutation."
        ),
        H("7", "Limitations"),
        P(
            "P2 is one partition rather than an independent external cohort. The bootstrap resamples observations and "
            "does not yet cluster by flight because a unique flight identifier is not supplied in the time-series files. "
            "Vehicle labels are excluded from models and cannot substitute for flight IDs. The neural baselines are "
            "under-trained, and no P3 result is available by design. Extreme-dose estimates rely on few observations."
        ),
        H("8", "Final Experiment Plan"),
        P(
            "The remaining work follows a fixed decision sequence. Each stage produces a saved configuration and "
            "evaluation record so that P3 cannot influence model design. The Transformer will be retained only if it "
            "improves P2 beyond the static residual baseline across seeds or provides a clearly stated high-dose gain."
        ),
        table(
            [
                ["Stage", "Decision"],
                ["1  Pipeline check", "Run a short GPU smoke test and verify tensor shapes"],
                ["2  P2 selection", "Compare architecture, context, loss, and high-dose weight"],
                ["3  Stability", "Repeat selected models with at least three random seeds"],
                ["4  Freeze", "Record architecture, preprocessing, threshold, and loss"],
                ["5  Locked test", "Evaluate P3 once with no further tuning"],
                ["6  Interpretation", "Run attention, integrated gradients, and grouped permutation"],
                ["7  Robustness", "Rotate P1, P2, and P3 roles using the frozen procedure"],
            ],
            [0.92 * inch, 2.26 * inch],
        ),
        P("Table 4. Prespecified sequence from preliminary validation to final evidence.", "Caption"),
        H("9", "Peer Evaluation"),
        P("This is an individual project. All data preparation, implementation, experiments, analysis, visualization, and report writing are assigned to Sanjib K C."),
        H("10", "Reproducibility"),
        P(
            "The full Python script and notebook regenerate every reported CSV table and PNG figure from the three "
            "5-minute JSON files with random seed 42. Code repository URL: "
            "<font color='#9B2226'>[insert public repository URL before submission]</font>. Add the author email and "
            "PantherID before submission."
        ),
        PageBreak(),
        H("", "References"),
        P("[1] W. K. Tobiska et al., \"Global real-time dose measurements using the Automated Radiation Measurements for Aerospace Safety system,\" <i>Space Weather</i>, vol. 14, pp. 1053-1080, 2016, doi: 10.1002/2016SW001419.", "Reference"),
        P("[2] C. J. Mertens et al., \"NAIRAS aircraft radiation model development, dose climatology, and initial validation,\" <i>Space Weather</i>, vol. 11, pp. 603-635, 2013, doi: 10.1002/swe.20100.", "Reference"),
        P("[3] V. M. Sadykov et al., \"Machine Learning-Ready Data Sets for the Analysis and Nowcasting of Atmospheric Radiation at Aviation Altitudes,\" arXiv:2602.07108, 2026.", "Reference"),
        P("[4] S. K C, V. M. Sadykov, and D. Kempton, \"Nowcasting of Aviation Radiation Using Geospace Environment Properties: A Machine Learning Approach,\" arXiv:2511.15746, 2025.", "Reference"),
        P("[5] A. Vaswani et al., \"Attention Is All You Need,\" in <i>Advances in Neural Information Processing Systems</i>, 2017.", "Reference"),
        P("[6] M. Sundararajan, A. Taly, and Q. Yan, \"Axiomatic Attribution for Deep Networks,\" in <i>International Conference on Machine Learning</i>, 2017.", "Reference"),
    ]
    return story


if __name__ == "__main__":
    Paper(str(PDF)).build(build_story())
    print(PDF)
