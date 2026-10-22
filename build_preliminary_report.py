"""Build the IEEE-style preliminary ARMAS project report as a PDF."""

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
RESULTS = ROOT / "outputs" / "preliminary_5m"
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT / "Sanjib_KC_preliminary_project_report.pdf"

NAVY = colors.HexColor("#16324F")
MID_BLUE = colors.HexColor("#2E5EAA")
PALE_BLUE = colors.HexColor("#EAF1F8")
LIGHT_GRAY = colors.HexColor("#D9D9D9")
TEXT = colors.HexColor("#111111")


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#444444"))
    canvas.drawCentredString(letter[0] / 2, 0.34 * inch, str(doc.page))
    canvas.restoreState()


def first_page(canvas, doc):
    page_number(canvas, doc)
    canvas.saveState()
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica-Bold", 16.5)
    canvas.drawCentredString(letter[0] / 2, 10.40 * inch,
                             "Physics Guided Temporal Learning for Aviation")
    canvas.drawCentredString(letter[0] / 2, 10.12 * inch,
                             "Effective Dose Rate Estimation")
    canvas.setFont("Helvetica", 10)
    canvas.drawCentredString(letter[0] / 2, 9.78 * inch, "Sanjib K C")
    canvas.setFont("Helvetica", 8.5)
    canvas.drawCentredString(letter[0] / 2, 9.57 * inch,
                             "Department of Physics and Astronomy  Georgia State University")
    canvas.drawCentredString(letter[0] / 2, 9.38 * inch,
                             "CSC 8851 Deep Learning  Fall 2026  Preliminary Project Report")
    canvas.restoreState()


class TwoColumnDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=0.55 * inch,
            rightMargin=0.55 * inch,
            topMargin=0.55 * inch,
            bottomMargin=0.60 * inch,
            title="Physics Guided Temporal Learning for Aviation Effective Dose Rate Estimation",
            author="Sanjib K C",
            subject="CSC 8851 preliminary project report",
        )
        gutter = 0.24 * inch
        col = (letter[0] - self.leftMargin - self.rightMargin - gutter) / 2
        first_height = 8.60 * inch
        later_height = 9.75 * inch
        first_frames = [
            Frame(self.leftMargin, self.bottomMargin, col, first_height, id="first-left", leftPadding=0,
                  rightPadding=0.04 * inch, topPadding=0, bottomPadding=0),
            Frame(self.leftMargin + col + gutter, self.bottomMargin, col, first_height, id="first-right",
                  leftPadding=0.04 * inch, rightPadding=0, topPadding=0, bottomPadding=0),
        ]
        later_frames = [
            Frame(self.leftMargin, self.bottomMargin, col, later_height, id="left", leftPadding=0,
                  rightPadding=0.04 * inch, topPadding=0, bottomPadding=0),
            Frame(self.leftMargin + col + gutter, self.bottomMargin, col, later_height, id="right",
                  leftPadding=0.04 * inch, rightPadding=0, topPadding=0, bottomPadding=0),
        ]
        self.addPageTemplates([
            PageTemplate(id="First", frames=first_frames, onPage=first_page),
            PageTemplate(id="Later", frames=later_frames, onPage=page_number),
        ])


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="PaperBody", parent=styles["BodyText"], fontName="Times-Roman", fontSize=8.5,
    leading=10.2, alignment=TA_JUSTIFY, textColor=TEXT, spaceAfter=4.5,
))
styles.add(ParagraphStyle(
    name="PaperHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10.3,
    leading=12.2, textColor=colors.black, spaceBefore=7, spaceAfter=3, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="PaperSubheading", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=8.7,
    leading=10.3, textColor=colors.black, spaceBefore=5, spaceAfter=2, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="AbstractLabel", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=9.2,
    leading=11, alignment=TA_CENTER, textColor=colors.black, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="Caption", parent=styles["BodyText"], fontName="Times-Roman", fontSize=7.4,
    leading=8.7, alignment=TA_LEFT, textColor=TEXT, spaceBefore=2, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="Equation", parent=styles["BodyText"], fontName="Times-Italic", fontSize=9,
    leading=11, alignment=TA_CENTER, spaceBefore=3, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Reference", parent=styles["BodyText"], fontName="Times-Roman", fontSize=7.7,
    leading=9.2, leftIndent=12, firstLineIndent=-12, alignment=TA_LEFT, spaceAfter=3,
))


def P(text, style="PaperBody"):
    return Paragraph(text, styles[style])


def section(number, title):
    return P(f"{number}  {title}", "PaperHeading")


def formatted_table(data, widths, header=True, font_size=7.2):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.45, LIGHT_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]
        for row in range(2, len(data), 2):
            commands.append(("BACKGROUND", (0, row), (-1, row), PALE_BLUE))
    table.setStyle(TableStyle(commands))
    return table


def build_story():
    summary = pd.read_csv(RESULTS / "dataset_summary.csv")
    metrics = pd.read_csv(RESULTS / "p3_test_metrics.csv")
    overall = metrics[metrics["Regime"] == "All"].set_index("Model")
    high = metrics[metrics["Regime"] == "ARMAS > 15"].set_index("Model")

    nairas_rmse = overall.loc["NAIRAS V3", "RMSE"]
    residual_rmse = overall.loc["Residual ridge", "RMSE"]
    improvement = 100 * (nairas_rmse - residual_rmse) / nairas_rmse
    high_improvement = 100 * (high.loc["NAIRAS V3", "RMSE"] - high.loc["Residual ridge", "RMSE"]) / high.loc["NAIRAS V3", "RMSE"]

    story = [
        P("Abstract", "AbstractLabel"),
        P(
            "This project studies whether the previous hour of solar-terrestrial conditions can improve estimates "
            "of aviation effective dose rate. The proposed model learns the residual between in-flight ARMAS "
            "measurements and the physics-based NAIRAS Version 3 estimate using a compact Temporal Transformer. "
            "The available data contain 92,415 observations in three flight-separated partitions, with 35 "
            "environmental variables sampled at 12 five-minute steps and eight current flight and geomagnetic "
            f"variables. In a leakage-controlled final test, a linear residual model reduced P3 RMSE "
            f"from {nairas_rmse:.3f} to {residual_rmse:.3f} microSv/h ({improvement:.1f} percent) and reduced high-dose "
            f"RMSE by {high_improvement:.1f} percent. Initial neural baselines did not converge within the short "
            "preliminary budget and performed worse than the linear residual model. These results support residual "
            "learning, identify high-dose underprediction as the main remaining error, and motivate the planned "
            "Transformer and imbalance-aware experiments.",
        ),
        section("1", "Introduction"),
        P(
            "Aircraft occupants receive increased ionizing radiation at cruising altitude because atmospheric "
            "shielding is reduced. ARMAS provides measurements taken aboard aircraft, while NAIRAS estimates dose "
            "rates from a physics-based model [1, 2]. Both are expressed in microSv/h. A useful operational model "
            "must remain accurate across ordinary flights and during the less common high-radiation observations "
            "that matter most for exposure assessment."
        ),
        P(
            "Earlier static machine-learning work found that geospace variables can improve dose-rate nowcasts over "
            "uncorrected NAIRAS estimates [3, 4]. Static inputs cannot describe whether neutron-monitor counts, solar "
            "wind, particle flux, or geomagnetic activity changed during the preceding hour. This project therefore "
            "tests a physics-guided temporal model that predicts the ARMAS minus NAIRAS V3 discrepancy. The study asks "
            "whether temporal history improves validation error, which variables and lags are associated with large "
            "corrections, and whether weighted training reduces errors above 15 microSv/h."
        ),
        section("2", "Related Work"),
        P(
            "Tobiska et al. described the ARMAS measurement system and its role in global aviation radiation "
            "monitoring [1]. Mertens et al. presented the NAIRAS aircraft radiation model and its validation framework "
            "[2]. The machine-learning-ready dataset used here combines ARMAS, NAIRAS, aircraft position, and "
            "solar-terrestrial measurements [3]. Prior work by K C, Sadykov, and Kempton evaluated static machine "
            "learning for aviation radiation nowcasting [4]. The present study extends that line of work by using the "
            "sequence already attached to each ARMAS observation rather than another static learner."
        ),
        P(
            "Transformers use self-attention to relate positions in a sequence without recurrent computation [5]. "
            "For this problem, attention can connect the current prediction to any of the 12 five-minute samples. "
            "Integrated gradients provides a complementary attribution method based on a path from a reference input "
            "to the observed input [6]. Attention and attribution will be interpreted as model associations because "
            "neither establishes physical causation."
        ),
        NextPageTemplate("Later"),
        section("3", "Model and Method"),
        P("3.1  Dataset and split protocol", "PaperSubheading"),
        P(
            "Only the 5-minute files P1, P2, and P3 are used. P1 fits all preprocessors and model parameters. P2 is "
            "the validation partition. P3 supplies the final performance evaluation after P2 selection is complete. Missing values are "
            "median-imputed and continuous inputs are standardized from P1 statistics only. Vehicle ID is excluded."
        ),
    ]

    table1 = [["Partition", "Rows", "ARMAS > 15", "Percent"]]
    for _, row in summary.iterrows():
        table1.append([row["Partition"], f"{int(row['Rows']):,}", f"{int(row['ARMAS > 15']):,}",
                       f"{row['High-dose percent']:.1f}"])
    story += [
        formatted_table(table1, [0.72 * inch, 0.73 * inch, 1.02 * inch, 0.70 * inch]),
        P("Table 1. Dataset size and high-dose prevalence for P1 train, P2 validation, and P3 test.", "Caption"),
        P(
            "Each observation contains eight current flight and geomagnetic variables. The current-time baseline adds "
            "the present value of 35 environmental variables, giving 43 predictors. Temporal models receive the same "
            "35 variables over 12 samples spanning approximately one hour, ordered from oldest to current."
        ),
        P("3.2  Physics guided residual formulation", "PaperSubheading"),
        P(
            "Let A<sub>i</sub> be the ARMAS measurement and N<sub>i</sub> the NAIRAS V3 estimate. The model learns the "
            "residual r<sub>i</sub> and adds the predicted correction to NAIRAS:"
        ),
        P("r<sub>i</sub> = A<sub>i</sub> - N<sub>i</sub>,&nbsp;&nbsp;&nbsp; r-hat<sub>i</sub> = f<sub>theta</sub>(X<sub>i</sub>),&nbsp;&nbsp;&nbsp; A-hat<sub>i</sub> = N<sub>i</sub> + r-hat<sub>i</sub>.&nbsp;&nbsp;&nbsp; (1)", "Equation"),
        P(
            "Equation 1 makes NAIRAS the physical starting point. The learned component estimates a correction rather "
            "than reconstructing the full dose rate from environmental inputs alone."
        ),
        P("3.3  Models and training", "PaperSubheading"),
        P(
            "The preliminary study compares a P1 mean predictor, uncorrected NAIRAS V3, current-time ridge "
            "regression, ridge residual correction, a temporal ridge residual model, a 64-32 current-time multilayer "
            "perceptron, and a 96-48 temporal residual multilayer perceptron. Neural models use ReLU activations, Adam, "
            "mini-batches of 512, all P1 records for fitting, and a maximum of 35 iterations. P2 is the validation set."
        ),
        P(
            "The main implementation adds a compact Transformer encoder with positional information and two "
            "self-attention blocks. Its pooled sequence representation is combined with the eight current variables "
            "and NAIRAS V3 before predicting the residual. Standard mean squared error will be compared with a "
            "weighted Huber loss. The high-dose weight will be selected on P2 before P3 is evaluated."
        ),
        section("4", "Experiments and Results"),
        P("4.1  Data distribution", "PaperSubheading"),
        P(
            "The three partitions have similar target distributions and high-dose prevalence between 16.2 and 16.9 "
            "percent. The upper tail remains sparse relative to the lower-dose regime, which justifies separate error "
            "reporting and an imbalance-aware loss experiment."
        ),
        KeepTogether([
            Image(str(RESULTS / "figure1_partition_distributions.png"), width=3.25 * inch, height=1.67 * inch),
            P("Figure 1. ARMAS distributions in the three flight-separated partitions. Values above 30 microSv/h are clipped only for display.", "Caption"),
        ]),
        P("4.2  Preliminary P2 validation", "PaperSubheading"),
        P(
            f"Uncorrected NAIRAS V3 obtained an overall RMSE of {nairas_rmse:.3f} microSv/h on P2. The current-time "
            f"residual ridge model achieved {residual_rmse:.3f} microSv/h, an {improvement:.1f} percent reduction. Its "
            f"high-dose RMSE fell from {high.loc['NAIRAS V3','RMSE']:.3f} to {high.loc['Residual ridge','RMSE']:.3f} "
            "microSv/h. The temporal ridge model produced nearly the same high-dose RMSE but a slightly worse overall "
            "RMSE, so the linear check does not yet show a clear benefit from the full one-hour history."
        ),
    ]

    ordered_models = [
        "Training mean", "NAIRAS V3", "Current ridge", "Residual ridge",
        "Temporal residual ridge", "Current MLP", "Temporal residual MLP",
    ]
    table2 = [["Model", "All RMSE", "High RMSE"]]
    names = {
        "Training mean": "P1 mean", "NAIRAS V3": "NAIRAS V3", "Current ridge": "Current ridge",
        "Residual ridge": "Residual ridge", "Temporal residual ridge": "Temporal ridge",
        "Current MLP": "Current MLP", "Temporal residual MLP": "Temporal MLP",
    }
    for model in ordered_models:
        table2.append([names[model], f"{overall.loc[model, 'RMSE']:.3f}", f"{high.loc[model, 'RMSE']:.3f}"])
    story += [
        formatted_table(table2, [1.62 * inch, 0.78 * inch, 0.78 * inch]),
        P("Table 2. P2 validation RMSE in microSv/h. Lower is better. P3 is not evaluated.", "Caption"),
        KeepTogether([
            Image(str(RESULTS / "figure2_validation_rmse.png"), width=3.25 * inch, height=1.67 * inch),
            P("Figure 2. Overall P2 RMSE for preliminary baselines. These values support model development and are not final test results.", "Caption"),
        ]),
        P(
            "Both neural baselines reached the 35-iteration cap without convergence and underperformed the residual "
            "ridge model. This result is evidence that optimization, regularization, and architecture selection remain "
            "unfinished. It does not show that temporal neural models are unsuitable. The next experiment will train "
            "the proposed Transformer with validation-based early stopping over a longer epoch budget and will compare "
            "3, 6, and 12 time steps as a temporal-window ablation."
        ),
        P("4.3  High dose error analysis", "PaperSubheading"),
        P(
            f"High-dose observations account for {summary.loc[summary['Partition']=='P2','High-dose percent'].iloc[0]:.1f} "
            f"percent of P2. NAIRAS V3 has a high-dose bias of {high.loc['NAIRAS V3','Bias']:.3f} microSv/h, and the "
            f"residual ridge bias is {high.loc['Residual ridge','Bias']:.3f} microSv/h. Both negative values indicate "
            "systematic underprediction above the threshold. Weighted Huber training will therefore be judged by the "
            "change in high-dose RMSE and bias together with any loss of lower-dose accuracy."
        ),
        P("4.4  Planned ablations and decision rules", "PaperSubheading"),
        P(
            "The final configuration will be selected entirely from P2. The study will vary temporal context, loss, "
            "and architecture one factor at a time. Overall RMSE is the primary measure; high-dose RMSE and bias are "
            "prespecified secondary measures. A more complex model will be retained only when its validation gain is "
            "consistent across seeds and does not rely on unacceptable lower-dose degradation."
        ),
        formatted_table(
            [
                ["Ablation", "Values compared"],
                ["History", "3, 6, and 12 five-minute steps"],
                ["Architecture", "MLP, CNN, direct and residual Transformers"],
                ["Loss", "MSE and weighted Huber"],
                ["High-dose weight", "1, 2, and 4 selected on P2"],
                ["Stability", "Three random seeds for the selected models"],
            ],
            [1.05 * inch, 2.13 * inch],
            font_size=6.9,
        ),
        P("Table 3. Planned ablations before the locked P3 evaluation.", "Caption"),
        section("5", "Conclusions"),
        P(
            "The preliminary study establishes a working data pipeline, leakage-controlled partition protocol, "
            "reproducible baselines, and report-ready evaluation. Residual correction is promising: the best current "
            "result improves both overall and high-dose P2 RMSE relative to NAIRAS V3. The first temporal models do not "
            "yet establish a benefit from one-hour history, and the neural runs require more training and tuning. The "
            "largest practical error remains underprediction when ARMAS exceeds 15 microSv/h."
        ),
        P(
            "Before the final report, configurations will be frozen using P2, then evaluated once on P3. The final "
            "comparison will include NAIRAS V3, the current-time MLP, a hybrid one-dimensional CNN and MLP, a direct "
            "Temporal Transformer, and the physics-guided residual Transformer. Attention maps, integrated gradients, "
            "feature-group permutation, and lag ablations will be used to study associations between the correction and "
            "the input history."
        ),
        section("6", "Peer Evaluation"),
        P("This is an individual project. All data preparation, implementation, analysis, and report writing are assigned to Sanjib K C."),
        section("7", "Reproducibility"),
        P(
            "The analysis script reads the three newline-delimited 5-minute JSON files, writes the summary and metric "
            "tables, and regenerates both figures. Random seeds are fixed at 42. Code repository URL: "
            "<font color='#9B2226'>[insert public repository URL before submission]</font>. Author email and PantherID "
            "should be added to the author line before submission."
        ),
        PageBreak(),
        section("", "References"),
        P("[1] W. K. Tobiska et al., \"Global real-time dose measurements using the Automated Radiation Measurements for Aerospace Safety system,\" <i>Space Weather</i>, vol. 14, pp. 1053-1080, 2016, doi: 10.1002/2016SW001419.", "Reference"),
        P("[2] C. J. Mertens et al., \"NAIRAS aircraft radiation model development, dose climatology, and initial validation,\" <i>Space Weather</i>, vol. 11, pp. 603-635, 2013, doi: 10.1002/swe.20100.", "Reference"),
        P("[3] V. M. Sadykov et al., \"Machine Learning-Ready Data Sets for the Analysis and Nowcasting of Atmospheric Radiation at Aviation Altitudes,\" arXiv:2602.07108, 2026.", "Reference"),
        P("[4] S. K C, V. M. Sadykov, and D. Kempton, \"Nowcasting of Aviation Radiation Using Geospace Environment Properties: A Machine Learning Approach,\" arXiv:2511.15746, 2025.", "Reference"),
        P("[5] A. Vaswani et al., \"Attention Is All You Need,\" in <i>Advances in Neural Information Processing Systems</i>, 2017.", "Reference"),
        P("[6] M. Sundararajan, A. Taly, and Q. Yan, \"Axiomatic Attribution for Deep Networks,\" in <i>International Conference on Machine Learning</i>, 2017.", "Reference"),
    ]
    return story


if __name__ == "__main__":
    doc = TwoColumnDoc(str(PDF_PATH))
    doc.build(build_story())
    print(PDF_PATH)
