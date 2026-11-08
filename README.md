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
│   ├── 04_supplementary_checks.ipynb       # duplicates across splits, de-duplicated retraining,
│   │                                        # majority-class baseline, port ablation
│   ├── 05_status_report_experiments.ipynb  # Status Report: host-grouped CV, model comparison, tuning,
│   │                                        # fingerprint ablation, error analysis, SHAP, evasion test,
│   │                                        # UNSW-NB15 generalization, drift (PSI)
│   └── 06_spoofing_and_calibration.ipynb   # MITM spoofing detectors, rules vs ML vs hybrid,
│                                            # alert-threshold calibration on validation hosts
├── app/                          # Streamlit demo dashboard (see "Demo dashboard" below)
│   ├── app.py                    # entry point
│   ├── build_demo_assets.py      # trains demo models and prepares replay traffic (run once)
│   ├── demo_lib.py               # shared helpers for the pages
│   └── views/                    # overview, live monitor, MITM lab, results, attack simulator, drift
├── configs/
│   └── status_report_config.json # seeds, candidate hyperparameters, experiment settings
├── src/
│   ├── ids_common.py             # shared: de-duplication, spoofing detectors, models, PSI
│   ├── status_report_experiments.py  # same code as notebook 05 (VS Code "# %%" cells)
│   ├── spoofing_and_calibration.py   # same code as notebook 06
│   ├── data_preprocessing.py     # script version of the preprocessing (being aligned with notebook 02)
│   └── make_report_figures.py    # redraws Figures 1–3 in the interim report
└── results/
    ├── metrics/                  # CSV/JSON outputs from every notebook
    ├── figures/                  # plots and confusion matrices from notebooks 01 and 03
    ├── report_figures/           # Figures 1–3 used in the interim report
    ├── status_report/            # metrics/ and figures/ for the Status Report (notebook 05)
    └── spoofing/                 # metrics/ and figures/ for spoofing detection and calibration (notebook 06)
```

Model files (`models/*.joblib`) are not committed because the Random Forest models are too large for GitHub. Running notebooks 02 and 03 recreates them.

## How to Run

1. Download `train_test_network.csv` and place it in `data/` (see `data/README.md`).
2. Install dependencies: `pip install -r requirements.txt`
3. For notebook 05, also download UNSW-NB15 into `data/external/unsw_nb15/` (see `data/README.md`).
4. Run the notebooks in order from the `notebooks/` folder: 01 → 02 → 03 → 04 → 05 → 06. Notebook 05 takes about 50 minutes and caches finished steps in `results/status_report/cache/`; notebook 06 takes about 4 minutes.

## Demo Dashboard

A Streamlit app for presenting the system live. From the project folder:

```bash
pip install -r requirements.txt
python app/build_demo_assets.py   # once, about 3 minutes (needs data/train_test_network.csv)
streamlit run app/app.py          # opens http://localhost:8501
```

| Page | What it shows |
|---|---|
| Overview | Headline results and how the system works |
| Live monitor | Replays traffic from unseen devices; switch between the calibrated and default threshold; open any alert to see its SHAP reasons and spoofing-rule hits, then confirm or dismiss it |
| MITM spoofing lab | The man-in-the-middle attack chain, the four spoofing detectors, and rules vs. ML vs. hybrid |
| Model results | Random split vs. unseen devices, fingerprint ablation, confusion matrix, external dataset |
| Attack simulator | Pad or slow down an attack flow and watch the standard vs. adversarially trained model |
| Drift monitor | Population Stability Index vs. training traffic, including UNSW-NB15, with the response workflow |

Screenshots are in `docs/demo_screenshots/`.

## Responsible AI and Security Considerations

The project will report class-specific performance because missed attacks and excessive false alerts have different security consequences. Model explanations will be reviewed for consistency with the underlying network features. The prototype is intended to support security-analyst review and will not automatically block network traffic. We will not use the project to identify individuals or expose private information.

## Project Status

**Current stage:** Interim Status Report (Deliverable 3) — models compared on unseen hosts, robustness and generalization tested.

| Model (host-grouped 5-fold: test hosts never seen in training) | 10-class macro-F1 | Attack recall | False-positive rate | Recall at 1% FPR |
|---|---|---|---|---|
| Majority class | 0.06 | 0.000 | 0.0% | — |
| Logistic Regression | 0.65 ± 0.19 | 0.995 | 27.2% | 0.90 |
| **Random Forest** | **0.70 ± 0.08** | **0.998** | 19.5% | **0.98** |
| Histogram Gradient Boosting | 0.64 ± 0.14 | 0.990 | 25.7% | 0.91 |
| Isolation Forest (unsupervised) | — | 0.213 | 6.7% | 0.005 |

On ordinary random folds the same models reach 0.91–0.96 macro-F1 with under 1.5% false positives, so earlier scores were optimistic.
Adding IP addresses looks perfect on random folds (0.99) but collapses on unseen hosts (0.17); dropping ports changes little.
The model does not transfer to UNSW-NB15 without adaptation (ROC-AUC 0.47) but recovers with 5% labelled data (0.97).
**Final system (notebook 06):** hybrid Random Forest + four MITM spoofing detectors (ICMP redirect, TTL-exceeded, rogue DNS responder, TLS stripping) with an alert threshold calibrated on validation hosts.
On unseen devices it catches 96.2% of attacks with 1.4% false alarms (down from 15.2% at the default threshold), and MITM F1 rises from 0.53 to 0.59 compared with the ML model alone at the same false-alarm budget.

## Citation

UNSW Canberra. *The TON_IoT Datasets*. Available at: <https://research.unsw.edu.au/projects/toniot-datasets>.

Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., and Anwar, A. “TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven Intrusion Detection Systems.” *IEEE Access*, 2020.

Moustafa, N. “A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets.” *Sustainable Cities and Society*, 2021.

Moustafa, N., and Slay, J. “UNSW-NB15: A comprehensive data set for network intrusion detection systems.” *MilCIS*, 2015.
