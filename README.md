# Loan Default Risk Decisioning System

> End-to-end financial decisioning pipeline for loan default risk: SQL feature engineering, LightGBM modeling, SHAP explainability, batch scoring, threshold analysis, and Power BI reporting.

## Overview

This project simulates a financial-services decision-support workflow. It predicts applicant repayment-difficulty risk, converts scores into risk-based action bands, writes batch predictions back to a DuckDB table, and visualizes model and threshold tradeoffs in Power BI.

The goal is not to build a production underwriting system. The goal is to demonstrate applied ML engineering for a financial decisioning use case.

## Business Problem

A lender needs to balance portfolio growth against credit losses. A model score alone is not enough; the business needs to understand how thresholds affect approval volume, default capture, manual review workload, and expected value.

This project answers:

> Which applicants are most likely to experience repayment difficulty, and how should score thresholds be set to balance approval rate, default risk, review volume, and expected portfolio value?

## Architecture

```text
Kaggle CSV files
   ↓
Parquet conversion
   ↓
DuckDB staging tables
   ↓
SQL feature extraction
   ↓
Applicant-level feature mart
   ↓
Python model training/evaluation
   ↓
Batch scoring table
   ↓
Power BI dashboard
```

## Dataset

Primary dataset: **Home Credit Default Risk** public Kaggle dataset.

The model predicts:

```text
TARGET = 1: applicant experienced repayment difficulty
TARGET = 0: applicant did not experience observed repayment difficulty
```

v1 uses:

- `application_train.csv`
- `application_test.csv`
- `bureau.csv`
- `previous_application.csv`
- `installments_payments.csv`

## Stack

| Layer | Tools |
|---|---|
| Storage | Parquet |
| Database | DuckDB |
| Feature engineering | SQL |
| Modeling | Python, pandas, scikit-learn, LightGBM |
| Imbalance/evaluation | scikit-learn, imbalanced-learn if justified |
| Explainability | SHAP |
| Testing | pytest |
| Reproducibility | Docker, Makefile |
| Reporting | Power BI |

## Modeling Approach

Models:

1. Logistic regression baseline
2. LightGBM primary model

Evaluation emphasizes metrics appropriate for imbalanced financial outcomes:

- PR-AUC
- ROC-AUC
- Brier score
- top-decile lift
- recall at review capacity
- confusion matrix by threshold
- expected business value

Accuracy is not used as the headline metric.

## Decision Policy

Model scores are converted into simulated business actions:

| Score range | Risk band | Simulated action |
|---:|---|---|
| `< T_low` | Low risk | Approve |
| `T_low` to `< T_high` | Medium risk | Manual review |
| `>= T_high` | High risk | Decline or high-priority review |

Thresholds are selected using validation-set performance and explicit business assumptions.

## Expected-Value Framework

```text
Expected Value =
    approved_good_loans * expected_margin_per_good_loan
  - approved_bad_loans * expected_loss_per_bad_loan
  - manual_reviews * manual_review_cost
```

Starting assumptions:

| Assumption | Value |
|---|---:|
| Expected margin per good approved loan | $1,000 |
| Expected loss per bad approved loan | $5,000 |
| Manual review cost | $50 |
| Manual review capacity | 10% of applicants |

These are illustrative scenario assumptions, not real Home Credit economics.

## Power BI Dashboard

The main dashboard page will show:

- score distribution;
- risk band counts;
- KPI cards for PR-AUC, ROC-AUC, Brier score, and lift;
- threshold scenario comparison;
- confusion matrix;
- lift by decile;
- expected value by threshold;
- approval/default tradeoff;
- top model drivers.

## Repository Structure

```text
loan-default-decisioning/
├── README.md
├── PROJECT_SPEC.md
├── Makefile
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── configs/
├── data/
├── sql/
├── src/
├── tests/
├── notebooks/
├── reports/
├── powerbi/
└── models/
```

## How to Run

```bash
make setup
make ingest
make features
make train
make evaluate
make score
make dashboard-data
make test
```

Raw Kaggle data is not committed to this repository. Download the dataset separately and place the CSV files in `data/raw/`.

## Project Outputs

- applicant-level feature mart;
- trained logistic regression baseline;
- trained LightGBM model;
- evaluation metrics;
- lift and calibration tables;
- threshold scenario table;
- batch scoring table;
- SHAP feature importance and reason-code-style outputs;
- Power BI dashboard screenshots.

## Limitations

This is a portfolio project and decision-support simulation, not an automated underwriting system.

Production credit systems require fair-lending review, monitoring, governance, adverse-action controls, legal/compliance approval, and stronger model-risk management than this project claims to provide.

Direct demographic and protected-status-like fields are excluded from v1 model features. If age, gender, or marital/family-status fields are inspected, they are retained only in a separate diagnostic layer for segment checks, not model training, deployment approval, or fair-lending compliance claims.

SHAP outputs are used for model interpretation and debugging. They are not legally compliant adverse-action notices.
