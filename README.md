# CreditGuard — Credit Default Risk Predictor

<div align="center">

```
   _____              _ _ _    _____                     _ 
  / ____|            | (_) |  / ____|                   | |
 | |     _ __ ___  __| |_| |_| |  __ _   _  __ _ _ __ __| |
 | |    | '__/ _ \/ _` | | __| | |_ | | | |/ _` | '__/ _` |
 | |____| | |  __/ (_| | | |_| |__| | |_| | (_| | | | (_| |
  \_____|_|  \___|\__,_|_|\__|\_____|\__,_|\__,_|_|  \__,_|
```

**An Educational Machine-Learning Decision-Support System for Retail Credit Risk Underwriting**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3B82F6.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FFD84D.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-B6FF4A.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-8B5CF6.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-111111.svg)](LICENSE)

</div>

---

## 🛡️ Executive Summary & Project Goal

**CreditGuard** is a responsive, full-stack machine-learning web application designed as an **educational decision-support tool** for estimating the probability that a loan applicant may default. 

### ⚠️ Critical Operational Principle
> **Educational Decision Support Only. Not an Automated Lending Decision System.**  
> CreditGuard outputs transparent default probability estimates and factor breakdowns to assist accredited human underwriters. It **must never autonomously approve or reject credit applications**. All predictions are statistical estimates derived from historical cohorts and require human verification.

---

## 🎨 Neo-Brutalist UI/UX Design System

CreditGuard is styled with a bold **Neo-Brutalist Fintech** aesthetic:
- **Canvas / Background:** Light cream (`#F7F3E8`) for high readability without harsh glare.
- **Borders & Shadows:** Heavy black borders (`3px solid #111111`) with offset drop shadows (`6px 6px 0px #111111`).
- **Accent Color Palette:**
  - 🔵 **Fintech Blue:** `#3B82F6` (Primary action & discrimination)
  - 🟣 **Deep Purple:** `#8B5CF6` (Metrics & balance)
  - 🟢 **Neon Green:** `#B6FF4A` (Low Risk & verified status)
  - 🟡 **Vibrant Yellow:** `#FFD84D` (Moderate Risk & warnings)
  - 🔴 **Coral Red:** `#FF6B5E` (High Risk & default indicators)
- **Interactive States:** Buttons physically depress and shift on hover (`translate(-3px, -3px)`) and click (`translate(2px, 2px)`).
- **Typography:** Space Grotesk, Inter, and Archivo Black with uppercase labels and clear visual hierarchy.

---

## 🗂️ Project Architecture & Directory Structure

```text
credit_score-project/
├── app.py                             # Main Streamlit web application
├── requirements.txt                   # Production dependencies
├── README.md                          # Comprehensive documentation
├── assets/
│   ├── style.css                      # Neo-Brutalist custom CSS stylesheet
│   └── logo.svg                       # High-resolution vector emblem
├── data/
│   ├── raw/
│   │   └── german_credit.csv          # Raw benchmark dataset (auto-downloaded)
│   └── processed/
│       ├── train.csv                  # Stratified training split (800 records)
│       └── test.csv                   # Stratified holdout test split (200 records)
├── models/
│   ├── best_model_pipeline.joblib     # Selected champion model pipeline
│   ├── logistic_pipeline.joblib       # Balanced Logistic Regression pipeline
│   ├── random_forest_pipeline.joblib  # Balanced Random Forest pipeline
│   ├── gradient_boosting_pipeline.joblib # Gradient Boosting pipeline
│   ├── model_comparison_metrics.joblib # Evaluated test curves & matrices
│   ├── feature_importances.joblib     # Extracted MDI & regression coefficients
│   └── training_metadata.joblib       # Timestamped training metadata
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Ingestion, EDA, and preprocessing notebook
│   └── 02_model_training_evaluation.ipynb # Modeling, benchmarking, & curves
├── src/
│   ├── __init__.py                    # Python package initializer
│   ├── data_preprocessing.py          # Data cleaning, mapping, and split logic
│   ├── train_model.py                 # Pipeline training and artifact export
│   ├── evaluate_model.py              # Confusion matrices, ROC, and PR curves
│   └── utils.py                       # Neo-Brutalist Plotly styling & presets
└── scratch/
    └── generate_notebooks.py          # Notebook generator script
```

---

## ⚡ Quickstart & Setup Instructions

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.

### 2. Clone / Open Repository
```bash
cd "credit_score -project"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Model Training (Optional, artifacts already pre-generated)
```bash
# Ingest data, clean, and split
python -m src.data_preprocessing

# Train and evaluate all 3 models
python -m src.train_model
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Dataset: UCI Statlog German Credit Data

CreditGuard utilizes the gold-standard **UCI Statlog German Credit Dataset** (1,000 historical credit agreements, 20 financial & demographic features, and 1 binary repayment outcome).

### Automated Ingestion
CreditGuard automatically downloads the dataset directly from the official **UCI Machine Learning Repository** if `data/raw/german_credit.csv` is not found.

### Placing Custom or Local Data
To use your own dataset, place a CSV file named `german_credit.csv` into `data/raw/` with the standard column structure:
- **Numerical Features (7):** `duration_months`, `credit_amount`, `installment_rate`, `residence_duration`, `age_years`, `existing_credits`, `num_dependents`
- **Categorical Features (13):** `checking_status`, `credit_history`, `purpose`, `savings_status`, `employment_duration`, `personal_status_sex`, `other_debtors`, `property`, `other_installment_plans`, `housing`, `job_type`, `telephone`, `foreign_worker`
- **Target:** `default_risk` (`0`: Good / Non-Default, `1`: Bad / Default)

---

## 🧠 Machine-Learning Workflow & Methodology

```
┌─────────────────────────┐
│ Raw German Credit Data  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Clean & Handle Missing  │ ──> Numeric: Median Imputation
│  Categorical Decoding   │ ──> Categorical: "Unknown" Imputation
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Stratified Split (80/20)│ ──> 800 Train / 200 Test (30% Default Rate Preserved)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Scikit-Learn Pipeline   │ ──> StandardScaler (Numeric)
│  (ColumnTransformer)    │ ──> OneHotEncoder (Categorical, handle_unknown='ignore')
└────────────┬────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ Train & Compare Under Class Balancing:                         │
│  1. Logistic Regression (Interpretable linear baseline)        │
│  2. Random Forest Classifier (Non-linear tree ensemble)        │
│  3. Gradient Boosting Classifier (Sequential gradient trees)   │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ Multi-Metric Evaluation:                                       │
│  ROC-AUC, Recall, Precision, F1-Score, PR-Curve, CM Cost Matrix │
└────────────┬───────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────┐
│ Champion Model Selection: Balanced Logistic Regression         │
│  Highest Recall (80.0%), Top ROC-AUC (0.808), Full Odds Ratios │
└────────────────────────────────────────────────────────────────┘
```

---

## 📈 Model Performance Benchmarks

Evaluated on the out-of-sample 20% holdout test set (200 retail applicants, 60 actual defaults):

| Model Architecture | Accuracy | Precision | Recall (Default Catch) | F1-Score | ROC-AUC | Decision Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Balanced) | **75.0%** | **55.8%** | **80.0%** | **0.658** | **0.808** | 🏆 **SELECTED BEST** |
| **Random Forest** (Balanced) | 76.0% | 59.1% | 65.0% | 0.619 | 0.800 | Strong Ensemble |
| **Gradient Boosting** | 78.5% | 67.3% | 55.0% | 0.606 | 0.799 | High Precision |

### 💡 Why Accuracy Alone Is Not Enough in Credit Lending
In consumer credit, non-defaulters significantly outnumber defaulters (70% vs 30% in this dataset; 97% vs 3% in prime retail banking). 
- A naive dummy model that approves everyone achieves **70.0% Accuracy**, yet catches **0% of defaulters (0% Recall)**.
- Missing a default (**False Negative**) costs thousands of dollars in lost principal.
- Flagging a good borrower for manual check (**False Positive**) costs only minor underwriter review time.
- **Therefore, CreditGuard selects models based on ROC-AUC, Recall, and business loss minimization—not accuracy alone.**

---

## 🔍 Feature Importance & Attribution

### Random Forest (Top Structural Features via MDI)
1. **Credit Duration (Months):** Longer tenors increase exposure to macroeconomic volatility.
2. **Credit Amount (DM):** High principal requests relative to income increase debt service stress.
3. **Applicant Age (Years):** Younger cohorts show higher statistical variance in credit maturity.
4. **Checking Account Liquidity:** Immediate cash balances (< 0 DM vs >= 200 DM) act as immediate filters.

### Logistic Regression (Directional Odds Ratios)
- **Top Risk Escalators (+ Odds):** Overdrawn checking balance (`< 0 DM`), extended duration, retail installment plans, prior payment delays.
- **Top Protective Factors (- Odds):** Healthy checking reserves (`>= 200 DM`), substantial savings (`>= 1000 DM`), real estate ownership, long employment tenure (`>= 7 years`).

> **Causation Notice:** Feature importance represents historical empirical associations in observational data. It does not prove direct causality.

---

## ⚖️ Responsible AI, Ethics & Compliance

### 1. Cost Asymmetry in Underwriting
- **False Positive:** Good borrower flagged for review $\rightarrow$ Added friction, potential delay.
- **False Negative:** Defaulter approved without scrutiny $\rightarrow$ Unrecoverable principal loss.

### 2. Fair Lending Laws & Protected Characteristics
Under the **Equal Credit Opportunity Act (ECOA)**, **Fair Credit Reporting Act (FCRA)**, and **EU AI Act (High-Risk AI Systems for Credit Scoring)**:
- Protected demographic characteristics (age, sex, marital status, nationality) **must never be used to automatically reject applicants**.
- Proxy variables (such as zip code or housing type) can inadvertently encode socioeconomic disparities.
- Any adverse action requires human oversight and explainable Adverse Action Notices.

### 3. Intended Use vs. Out-of-Scope
- ✅ **Approved Intended Use:** Educational research, decision support for accredited underwriters, sensitivity scenario analysis.
- ❌ **Prohibited Out-of-Scope:** Fully automated loan approval/denial without human discretion, employment screening, credit scoring without ongoing disparate impact audits.

---

## 🖥️ Application Pages Overview

1. **🏠 Dashboard / Home:** Executive hero banner, active model health, live KPI cards (ROC-AUC, Recall, Precision, F1), historical portfolio breakdown, and ethical disclaimer.
2. **⚡ Risk Prediction:** Multi-group input form (Personal, Employment, Loan, Credit History), 1-click persona loaders, input validation, large percentage risk display, Neo-Brutalist status badges, factor explanations, and dynamic threshold slider.
3. **📊 Model Performance:** Comparative metrics table, interactive Plotly ROC & Precision-Recall curves, confusion matrix heatmap, and financial cost translations.
4. **🔍 Feature Insights:** Interactive horizontal bar charts for Random Forest importance and Logistic Regression coefficients, plain-language insights, and empirical cohort explorer.
5. **⚖️ Responsible AI & Limitations:** Full compliance documentation, false positive/negative trade-offs, dataset limitations, and regulatory guidelines.

---

## 📸 Interactive Experience Highlights

- **Persona Presets:** Test Prime Borrower, Young Professional, and Stretched Borrower with 1 click.
- **Dynamic Decision Threshold:** Adjust the slider from 0.10 to 0.90 to observe how risk sensitivity changes between cautious (high recall) and lenient modes.
- **Turnkey Offline Fallback:** Gracefully handles missing files with a 1-click in-app pipeline runner.

---

<div align="center">
Built with precision for explainable, responsible machine learning in credit risk.
</div>
