# TON IoT Explainable IDS

## Explainable Intrusion Detection for IoT and IIoT Network Traffic

This project is being developed by **Vidhi Hemal Doshi** and **Rishikesh Pechetti** for the IS 665 Cybersecurity Analytics class.

Our project focuses on building an explainable Intrusion Detection System (IDS) for Internet of Things (IoT) and Industrial Internet of Things (IIoT) network traffic. The system will identify whether a network flow is normal or malicious, classify the type of attack, and explain which network features influenced the prediction.

## Project Goals

- Detect normal and malicious network traffic.
- Classify malicious traffic into attack categories.
- Compare baseline, supervised, and anomaly-detection models.
- Address class imbalance, especially for the smaller MITM category.
- Analyze possible data leakage from IP addresses and other identifiers.
- Provide understandable feature-importance explanations for model predictions.

## Dataset

We are using the **TON_IoT Network Dataset**, developed by UNSW Canberra. TON_IoT was collected in a cyber-range environment designed to represent IoT, IIoT, cloud, edge, and fog systems. The complete TON_IoT collection includes network traffic, IoT/IIoT telemetry, Windows activity, and Linux activity.

For this project, we are focusing on the processed network-flow dataset. Each record represents one labeled network flow or connection.

The downloaded network file contains:

- Approximately **211,043 records**
- **44 columns/features**
- **50,000 normal records**
- **161,043 attack records**
- Normal traffic and nine attack categories: backdoor, DDoS, DoS, injection, password, scanning, ransomware, XSS, and MITM

The dataset includes connection, protocol, packet, byte, DNS, HTTP, TLS, and connection-state information. The `label` column represents the binary target, where `0` means normal and `1` means attack. The `type` column identifies the specific traffic or attack category.

### Dataset access

The official dataset can be downloaded from the UNSW TON_IoT page:

<https://research.unsw.edu.au/projects/toniot-datasets>

The dataset is provided for academic research with proper citation. The CSV data is kept locally and is not uploaded to this repository. The repository contains documentation and code for obtaining and analyzing the dataset.

## Planned Technical Approach

1. Audit the dataset structure, labels, duplicate records, missing values, and class distribution.
2. Review IP addresses and other identifiers for potential data leakage.
3. Encode categorical features and prepare numerical features for modeling.
4. Establish a baseline using Logistic Regression or a Decision Tree.
5. Compare Random Forest, gradient-boosted trees, and Isolation Forest models.
6. Evaluate binary intrusion detection and multiclass attack classification.
7. Use feature importance, SHAP, or permutation importance to explain predictions.
8. Present results using a reproducible notebook and a lightweight dashboard or report.

## Evaluation

Because the dataset is imbalanced, accuracy will not be the only evaluation measure. We plan to report:

- Precision and recall
- Macro-F1 and weighted-F1
- Balanced accuracy
- PR-AUC and ROC-AUC
- False-positive and false-negative rates
- Confusion matrices
- Per-class performance, including MITM recall

We will use a stratified train, validation, and test split. We will also conduct a leakage-control experiment that removes IP-address fields and other identifiers to test whether the models learn general traffic behavior instead of memorizing the test environment.

## Repository Structure

```text
ton-iot-explainable-ids/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md                 # how to download train_test_network.csv (not committed)
├── docs/
│   └── dataset_reference/        # TON_IoT feature descriptions and record statistics
├── notebooks/
│   ├── 01_data_audit_eda.ipynb             # profiling, data quality, leakage checks, EDA plots
│   ├── 02_preprocessing.ipynb              # 70/15/15 stratified split + saved preprocessing pipeline
│   ├── 03_preliminary_baseline_models.ipynb # Logistic Regression and Random Forest baselines
│   └── 04_supplementary_checks.ipynb       # duplicates across splits, de-duplicated retraining,
│                                            # majority-class baseline, port ablation
├── src/
│   ├── data_preprocessing.py     # script version of the preprocessing (being aligned with notebook 02)
│   └── make_report_figures.py    # redraws Figures 1–3 in the interim report
└── results/
    ├── metrics/                  # CSV/JSON outputs from every notebook
    ├── figures/                  # plots and confusion matrices from notebooks 01 and 03
    └── report_figures/           # Figures 1–3 used in the interim report
```

Model files (`models/*.joblib`) are not committed because the Random Forest models are too large for GitHub. Running notebooks 02 and 03 recreates them.

## How to Run

1. Download `train_test_network.csv` and place it in `data/` (see `data/README.md`).
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks in order from the `notebooks/` folder: 01 → 02 → 03 → 04.

## Responsible AI and Security Considerations

The project will report class-specific performance because missed attacks and excessive false alerts have different security consequences. Model explanations will be reviewed for consistency with the underlying network features. The prototype is intended to support security-analyst review and will not automatically block network traffic. We will not use the project to identify individuals or expose private information.

## Project Status

**Current stage:** Interim Progress Report 1 — data audit, preprocessing pipeline and preliminary baselines complete.

| Model (stratified random test split) | Task | Accuracy | Macro-F1 | MITM recall |
|---|---|---|---|---|
| Majority class | Binary | 0.763 | 0.433 | — |
| Logistic Regression | Binary | 0.960 | 0.944 | — |
| Random Forest | Binary | 0.999 | 0.999 | — |
| Majority class | 10-class | 0.237 | 0.038 | 0.00 |
| Logistic Regression | 10-class | 0.838 | 0.797 | 0.94 |
| Random Forest | 10-class | 0.988 | 0.966 | 0.82 |

These scores are treated as an upper bound. Each attack type comes from only 2–6 source hosts, so the next step is a host-grouped evaluation, followed by MITM tuning on the validation split and SHAP explanations.

## Citation

UNSW Canberra. *The TON_IoT Datasets*. Available at: <https://research.unsw.edu.au/projects/toniot-datasets>.

Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., and Anwar, A. “TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven Intrusion Detection Systems.” *IEEE Access*, 2020.

Moustafa, N. “A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets.” *Sustainable Cities and Society*, 2021.
