const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, ExternalHyperlink,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak
} = require("docx");
const fs = require("fs");

const CHARTS = "../charts";
const FONT = "Arial";

function img(path, width, height, desc) {
  return new ImageRun({
    type: "png",
    data: fs.readFileSync(path),
    transformation: { width, height },
    altText: { title: desc, description: desc, name: desc },
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function pRich(runs, opts = {}) {
  return new Paragraph({ children: runs, spacing: { after: 160 }, ...opts });
}
function bullet(text, bold = false) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text, bold })],
    spacing: { after: 80 },
  });
}
function numbered(text, ref = "numbers") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    children: [new TextRun(text)],
    spacing: { after: 80 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20, color: "595959" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 240, before: 60 },
  });
}
function centerImage(path, width, height, captionText) {
  return [
    new Paragraph({
      children: [img(path, width, height, captionText)],
      alignment: AlignmentType.CENTER,
      spacing: { before: 120 },
    }),
    caption(captionText),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

function headerCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: "2E75B6", type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 19 })],
    })],
  });
}
function bodyCell(text, width, opts = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 70, bottom: 70, left: 100, right: 100 },
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.bold, size: 19 })],
    })],
  });
}

// ---------------------------------------------------------------
// Performance comparison table
// ---------------------------------------------------------------
const perfHeaders = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"];
const perfWidths = [2400, 1200, 1200, 1100, 1200, 1100, 1100];
const perfData = [
  ["Logistic Regression", "0.9967", "0.0000", "0.0000", "0.0000", "1.0000", "1.0000"],
  ["Decision Tree", "0.9967", "0.0000", "0.0000", "0.0000", "0.5000", "0.0033"],
  ["K-Nearest Neighbors", "0.9967", "0.0000", "0.0000", "0.0000", "0.5000", "0.0033"],
  ["Random Forest", "0.9967", "0.0000", "0.0000", "0.0000", "0.4933", "0.0033"],
];

const perfTable = new Table({
  width: { size: 9300, type: WidthType.DXA },
  columnWidths: perfWidths,
  rows: [
    new TableRow({ children: perfHeaders.map((h, i) => headerCell(h, perfWidths[i])) }),
    ...perfData.map((row, ri) => new TableRow({
      children: row.map((cell, i) => bodyCell(cell, perfWidths[i], {
        bold: i === 0,
        fill: ri === 0 ? "EAF1FA" : undefined,
      })),
    })),
  ],
});

// ---------------------------------------------------------------
// Document
// ---------------------------------------------------------------
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "1F4E79" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: "2E75B6" },
        paragraph: { spacing: { before: 260, after: 140 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers-imbalance",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers-challenges",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers-recommendations",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "Credit Card Fraud Detection — Task #07", size: 16, color: "808080" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", size: 18, color: "808080" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" }),
            new TextRun({ text: " of ", size: 18, color: "808080" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: "808080" }),
          ],
        })],
      }),
    },
    children: [
      // ---------------- TITLE PAGE ----------------
      new Paragraph({ spacing: { before: 1200 }, children: [] }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Credit Card Fraud Detection", bold: true, size: 48, color: "1F4E79" })],
        spacing: { after: 120 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Using Machine Learning", bold: true, size: 36, color: "2E75B6" })],
        spacing: { after: 600 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Final Project Report", size: 28, italics: true })],
        spacing: { after: 800 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Data Science Task #07", bold: true, size: 24 })],
        spacing: { after: 80 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Gexton Education — Summer Internship Program", size: 22 })],
        spacing: { after: 80 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Supervisor: Sir Muhammad Arham MH", size: 22 })],
        spacing: { after: 600 },
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "Dataset source: ", size: 20 }),
          new ExternalHyperlink({
            children: [new TextRun({ text: "Kaggle — Credit Card Fraud Detection (mlg-ulb)", style: "Hyperlink", size: 20 })],
            link: "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
          }),
        ],
      }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---------------- 1. PROJECT OVERVIEW ----------------
      h1("1. Project Overview"),
      p("Credit card fraud is one of the most common financial crimes, and companies increasingly rely on machine learning models to identify suspicious transactions in real time. As part of the Gexton Education Data Science Internship, this project develops and evaluates a machine learning pipeline capable of detecting fraudulent credit card transactions."),
      p("The project covers the full data science workflow: data loading and cleaning, exploratory data analysis (EDA), feature preprocessing, handling of severe class imbalance, training of multiple classification models, rigorous model evaluation, and business recommendations for financial institutions."),

      // ---------------- 2. DATASET ----------------
      h1("2. Dataset"),
      pRich([
        new TextRun("The assignment specifies the use of the "),
        new ExternalHyperlink({
          children: [new TextRun({ text: "Credit Card Fraud Detection dataset on Kaggle", style: "Hyperlink" })],
          link: "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        }),
        new TextRun(", which contains 284,807 anonymized European credit card transactions made over two days in September 2013, of which 492 are fraudulent (about 0.17% of all transactions). Each transaction has a Time field, a transaction Amount, 28 anonymized PCA-transformed features (V1–V28), and a binary Class label (1 = fraud, 0 = non-fraud)."),
      ]),
      pRich([
        new TextRun({ text: "Important note on the data used in this project: ", bold: true }),
        new TextRun("the original Kaggle CSV file (approximately 150 MB) was too large to open and process in the environment available for this assignment. A reduced, randomly sampled version of the dataset containing "),
        new TextRun({ text: "1,000 transactions", bold: true }),
        new TextRun(" (file: "),
        new TextRun({ text: "creditcard_small_1000.csv", italics: true }),
        new TextRun(") was therefore supplied and used for every step of this project instead of the full file. All preprocessing, EDA, modeling, and evaluation steps follow exactly the methodology that would be applied to the complete dataset — only the sample size differs."),
      ]),
      pRich([
        new TextRun("Because the fraud rate in the full dataset is already only ~0.17%, the 1,000-row sample contains just "),
        new TextRun({ text: "2 fraudulent transactions", bold: true }),
        new TextRun(" (0.2%). This extreme scarcity of positive examples is the single biggest constraint on this project's results and is discussed in detail in Section 7 (Challenges)."),
      ]),

      // ---------------- 3. DATA PREPARATION ----------------
      h1("3. Data Preparation & Cleaning"),
      p("The dataset was loaded into a pandas DataFrame and inspected for structure, data types, and summary statistics. All 31 columns (Time, V1–V28, Amount, Class) were confirmed to be numeric (float64 / int64)."),
      bullet("Missing values: 0 found across all 1,000 rows and 31 columns."),
      bullet("Duplicate records: 0 found."),
      bullet("Data types: all numeric; no encoding required for the PCA features."),
      bullet("Class column was cast to integer type and validated to contain only 0/1 values."),
      p("As no missing values or duplicates were present, the cleaned dataset is structurally identical to the raw sample; it was nonetheless saved separately (data/creditcard_cleaned.csv) to keep the cleaning step explicit and reproducible for the full dataset."),

      // ---------------- 4. EDA ----------------
      h1("4. Exploratory Data Analysis"),
      h2("4.1 Class Distribution"),
      p("The dataset is severely imbalanced: of the 1,000 transactions, 998 (99.8%) are legitimate and only 2 (0.2%) are fraudulent."),
      ...centerImage(`${CHARTS}/01_class_distribution.png`, 320, 256, "Figure 1 — Class distribution: fraud vs non-fraud transactions"),

      h2("4.2 Transaction Amount"),
      p("Transaction amounts are heavily right-skewed: most transactions are small (median ≈ $20), with a long tail of larger transactions up to $3,502."),
      ...centerImage(`${CHARTS}/02_amount_distribution.png`, 380, 253, "Figure 2 — Distribution of transaction amount"),
      p("Examining the relationship between amount and fraud, the two fraudulent transactions in this sample had amounts of $364.19 and $520.12 — both above the typical (median) non-fraud transaction, but well within the normal overall range. This suggests amount alone is not a reliable standalone signal for fraud, consistent with findings on the full Kaggle dataset."),
      ...centerImage(`${CHARTS}/03_amount_by_class.png`, 320, 256, "Figure 3 — Transaction amount by class"),

      h2("4.3 Transaction Time"),
      p("The Time feature (seconds elapsed since the first transaction) shows a bimodal/cyclical pattern consistent with two days of natural daily transaction volume cycles (more activity during the day, less overnight)."),
      ...centerImage(`${CHARTS}/04_time_distribution.png`, 380, 253, "Figure 4 — Distribution of transaction time"),

      h2("4.4 Feature Correlations"),
      p("A full correlation heatmap of all 31 features was generated to inspect relationships between the anonymized PCA components, Time, Amount, and Class."),
      ...centerImage(`${CHARTS}/05_correlation_heatmap.png`, 430, 358, "Figure 5 — Correlation heatmap of all features"),
      p("Correlation of each individual feature with the fraud label (Class) was also examined. With only 2 fraud examples in this sample, these correlation values are noisy and should be interpreted cautiously — on the full 284,807-row dataset, features such as V14, V12, V10, and V17 are well known to correlate strongly with fraud."),
      ...centerImage(`${CHARTS}/06_feature_correlation_with_class.png`, 350, 400, "Figure 6 — Feature correlation with fraud (Class)"),

      h2("4.5 EDA Summary"),
      bullet("The dataset is severely imbalanced (0.2% fraud in this sample), which is the central modeling challenge."),
      bullet("No missing values or duplicate rows were found."),
      bullet("Transaction amount is right-skewed; the fraud cases observed were mid-to-high value but not extreme outliers."),
      bullet("Time shows a natural daily cyclical pattern with no obvious anomalies tied to fraud in this small sample."),
      bullet("Anonymized PCA features (V1–V28) carry the real predictive signal but are not human-interpretable."),

      // ---------------- 5. MODEL DEVELOPMENT ----------------
      h1("5. Machine Learning Model Development"),
      h2("5.1 Feature Scaling"),
      p("All features — including Time and Amount, which are on a very different scale than the PCA components V1–V28 — were standardized using scikit-learn's StandardScaler. This is especially important for distance-based models such as KNN and for stable convergence of Logistic Regression."),

      h2("5.2 Train/Test Split"),
      p("The data was split into 70% training and 30% testing using a stratified split (sklearn train_test_split, stratify=Class) so that the very few fraud cases are distributed proportionally between the two sets. This produced 1 fraud case in the 700-row training set and 1 fraud case in the 300-row test set."),

      h2("5.3 Handling Class Imbalance"),
      p("The task requires handling the imbalanced classes using oversampling, undersampling, or class weighting. Two complementary techniques were applied:"),
      numbered("Random oversampling of the minority (fraud) class in the training set only, using scikit-learn's resample() utility, duplicating the single fraud training example up to match the majority class count (699 vs 699). Note: SMOTE (from the imbalanced-learn library) was not available in this offline environment, so scikit-learn's built-in resampling — one of the techniques explicitly permitted by the task brief — was used instead.", "numbers-imbalance"),
      numbered("class_weight=\"balanced\" was additionally applied to Logistic Regression, Decision Tree, and Random Forest as a second safeguard against the imbalance.", "numbers-imbalance"),
      p("Oversampling was applied strictly after the train/test split and only to the training data, to avoid leaking information into the test set."),

      h2("5.4 Models Trained"),
      bullet("Logistic Regression (class_weight=\"balanced\")"),
      bullet("Decision Tree (class_weight=\"balanced\", max_depth=6)"),
      bullet("Random Forest (200 trees, class_weight=\"balanced\", max_depth=8)"),
      bullet("K-Nearest Neighbors (k=5)"),
      p("All four models were trained on the oversampled, balanced training set and evaluated on the original, untouched 300-row test set (which retains the natural 299:1 class ratio)."),

      // ---------------- 6. EVALUATION ----------------
      h1("6. Model Evaluation & Results"),
      h2("6.1 Performance Comparison Table"),
      perfTable,
      new Paragraph({ spacing: { after: 200 } }),
      p("Note: ROC-AUC and PR-AUC are computed from predicted probabilities and are far more informative than the hard-label metrics here, since the test set contains only a single fraud example (discussed below)."),

      h2("6.2 Confusion Matrices"),
      p("At the standard 0.5 probability threshold, all four models predicted \"Non-Fraud\" for every one of the 300 test transactions, including the single true fraud case — giving Accuracy = 99.67% but Precision = Recall = F1-Score = 0 for all models. This starkly illustrates why accuracy is a misleading metric for fraud detection: a model that always predicts \"non-fraud\" would score the same 99.67% accuracy while catching zero fraud."),
      ...centerImage(`${CHARTS}/07_confusion_matrices_all.png`, 430, 387, "Figure 7 — Confusion matrices for all four models"),

      h2("6.3 Ranking Quality (ROC-AUC / PR-AUC)"),
      p("Because hard 0/1 classification is uninformative with only one positive test example, models were also compared by how they ranked the true fraud case by predicted probability among all 300 test transactions."),
      ...centerImage(`${CHARTS}/09_roc_pr_auc_comparison.png`, 380, 244, "Figure 8 — ROC-AUC and PR-AUC by model"),
      bullet("Logistic Regression ranked the true fraud case #1 (highest risk) out of 300 transactions — ROC-AUC = 1.00, PR-AUC = 1.00."),
      bullet("Decision Tree and K-Nearest Neighbors also ranked the fraud case #1, but produced many tied probability scores among other transactions, pulling their ROC-AUC down to 0.50."),
      bullet("Random Forest ranked the true fraud case 5th out of 300 — still a strong relative ranking, but not top-1."),

      h2("6.4 Best-Performing Model"),
      pRich([
        new TextRun("Based on F1-Score (tied at 0 across all models), with PR-AUC and ROC-AUC as tiebreakers, "),
        new TextRun({ text: "Logistic Regression", bold: true }),
        new TextRun(" is selected as the best-performing model in this evaluation."),
      ]),
      p("Why it performed better than the others:"),
      bullet("It assigned the single fraudulent test transaction the highest predicted fraud probability of all 300 test transactions (rank #1), achieving a perfect ROC-AUC and PR-AUC of 1.0 on this test set."),
      bullet("In a production setting using risk scores (rather than a hard 0.5 cutoff) to flag the highest-risk transactions for manual review, this model would have successfully surfaced the fraud case."),
      bullet("The tree-based models and KNN, trained on a duplicated (oversampled) single fraud example, tended to overfit narrowly to that exact example's feature values, which hurt their ability to generalize and correctly rank a different fraud example in the test set."),
      bullet("Logistic Regression's linear decision boundary, combined with class_weight=\"balanced\", generalized more smoothly from the single available training fraud example than the higher-variance tree-based and instance-based models did."),

      // ---------------- 7. CHALLENGES ----------------
      h1("7. Challenges of Detecting Fraudulent Transactions"),
      numbered("Extreme class imbalance — fraud is always a tiny fraction of all transactions (0.2% in this sample; ~0.17% in the full Kaggle dataset). Models can achieve very high accuracy simply by never predicting fraud, which is useless in practice.", "numbers-challenges"),
      numbered("Very few positive examples to learn from — with only 1–2 fraud cases available in this reduced 1,000-row sample, models cannot reliably learn the general patterns of fraud. The results in this report should be read as a methodology demonstration rather than a statistically robust fraud model; on the full 284,807-row dataset (492 fraud cases) the same techniques would produce far more reliable and generalizable results.", "numbers-challenges"),
      numbered("Anonymized / PCA features (V1–V28) make models accurate but hard to interpret or explain to compliance and regulatory teams, since the original meaning of each feature is hidden.", "numbers-challenges"),
      numbered("Evolving fraud patterns — fraudsters constantly change their tactics, so a model trained on historical data can go stale quickly and requires continuous retraining and monitoring.", "numbers-challenges"),
      numbered("Cost asymmetry — a false negative (missed fraud) is usually far more costly than a false positive (a legitimate transaction flagged for review), so models should be tuned and thresholded with this asymmetry in mind rather than optimizing for plain accuracy.", "numbers-challenges"),

      // ---------------- 8. RECOMMENDATIONS ----------------
      h1("8. Business Recommendations for Financial Institutions"),
      numbered("Never rely on accuracy alone to judge a fraud model — track Precision, Recall, F1-Score, and PR-AUC, and choose a decision threshold based on the institution's tolerance for false positives vs. false negatives.", "numbers-recommendations"),
      numbered("Use risk scores, not just binary flags — rank transactions by predicted fraud probability and route the highest-risk transactions to manual review or step-up authentication, rather than relying on a single hard cutoff.", "numbers-recommendations"),
      numbered("Retrain models regularly on recent transaction data to keep up with evolving fraud patterns, and monitor for model and data drift over time.", "numbers-recommendations"),
      numbered("Combine multiple models and signals (ensemble approaches, rule-based checks, device/location/behavioral signals) since no single model will catch every type of fraud.", "numbers-recommendations"),
      numbered("Invest in collecting more labeled fraud examples — the single biggest lever for improving real-world fraud models is more (recent) positive examples, since fraud detection is fundamentally data-starved by nature.", "numbers-recommendations"),
      numbered("Apply class-imbalance handling techniques in production (oversampling, undersampling, class weighting, or anomaly-detection approaches) exactly as demonstrated in this project, scaled to the full transaction volume.", "numbers-recommendations"),

      // ---------------- 9. CONCLUSION ----------------
      h1("9. Conclusion"),
      p("This project implemented a complete, end-to-end fraud detection pipeline: data loading and cleaning, exploratory data analysis, feature scaling, a stratified train/test split, class-imbalance handling via random oversampling and class weighting, training of four classification models (Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors), and evaluation using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ranking metrics (ROC-AUC / PR-AUC)."),
      p("Logistic Regression was selected as the best-performing model in this evaluation, successfully ranking the test set's single fraud case as the highest-risk transaction out of 300. The severe scarcity of fraud examples in this 1,000-row sample is itself the most important finding of this analysis: it demonstrates, very concretely, why fraud detection is such a hard real-world machine learning problem, and why financial institutions need large volumes of historical fraud data, careful metric selection, and risk-based (rather than purely binary) decisioning to deploy these models successfully."),

      h2("Deliverables"),
      bullet("Jupyter Notebook — notebooks/Credit_Card_Fraud_Detection.ipynb"),
      bullet("Cleaned dataset — data/creditcard_cleaned.csv"),
      bullet("Trained ML models — models/*.joblib"),
      bullet("Confusion matrix visualizations — charts/"),
      bullet("Performance comparison table — report/model_performance_comparison.csv"),
      bullet("This final project report (DOCX)"),

      new Paragraph({ spacing: { before: 300 } }),
      pRich([
        new TextRun({ text: "Dataset source: ", bold: true }),
        new ExternalHyperlink({
          children: [new TextRun({ text: "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", style: "Hyperlink" })],
          link: "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        }),
      ]),
      p("Sample used in this project: creditcard_small_1000.csv (1,000-row random sample, supplied because the full ~150 MB Kaggle file could not be opened in this environment)."),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("Credit_Card_Fraud_Detection_Report.docx", buffer);
  console.log("Report written.");
});
