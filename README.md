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
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   └── project_proposal.pdf
├── notebooks/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_candidate_models.ipynb
│   └── 04_model_explainability.ipynb
├── src/
├── results/
└── dashboard/
```

## Responsible AI and Security Considerations

The project will report class-specific performance because missed attacks and excessive false alerts have different security consequences. Model explanations will be reviewed for consistency with the underlying network features. The prototype is intended to support security-analyst review and will not automatically block network traffic. We will not use the project to identify individuals or expose private information.

## Project Status

**Current stage:** Project proposal and dataset preparation.

Model development, evaluation, explainability analysis, and the demonstration will be added as the project progresses.

## Citation

UNSW Canberra. *The TON_IoT Datasets*. Available at: <https://research.unsw.edu.au/projects/toniot-datasets>.

Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., and Anwar, A. “TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven Intrusion Detection Systems.” *IEEE Access*, 2020.

Moustafa, N. “A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets.” *Sustainable Cities and Society*, 2021.
