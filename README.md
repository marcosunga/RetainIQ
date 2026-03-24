# RetainIQ — E-Commerce Churn Intelligence

> Machine learning pipeline for predicting and analysing customer churn in e-commerce datasets.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

RetainIQ is a complete ML pipeline that trains, evaluates, and compares multiple churn prediction models on e-commerce customer data. It produces:

- **Model accuracy reports** with full classification metrics
- **Feature importance rankings** (SHAP-style analysis)
- **Segment-level churn breakdowns** (loyalty tier, category, tenure)
- **Interactive HTML dashboard** for business stakeholders
- **Saved model artefacts** for deployment

---

## Project Structure

```
retainiq/
├── data/
│   └── ecommerce_churn_dataset.csv   # Raw dataset (4,000 customers)
├── models/                            # Saved model artefacts (.joblib)
├── reports/
│   ├── ecommerce-churn-dashboard.html # Pre-built HTML dashboard
│   └── model_report.html              # Auto-generated model report
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py               # Feature engineering & encoding
│   ├── evaluation.py                  # Metrics, plots, reports
│   └── visualisation.py               # Chart helpers
├── train.py                           # Train & compare all models
├── predict.py                         # Score new customers
├── evaluate.py                        # Deep-dive model evaluation
├── config.py                          # Central configuration
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-org/retainiq.git
cd retainiq
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train all models

```bash
python train.py
```

This runs Logistic Regression, Random Forest, Gradient Boosting, and XGBoost, then prints a comparison table and saves the best model to `models/`.

### 3. Full evaluation report

```bash
python evaluate.py --model xgboost
```

Outputs classification report, confusion matrix, ROC curve, precision-recall curve, and feature importance chart.

### 4. Predict on new data

```bash
python predict.py --input data/new_customers.csv --output data/predictions.csv
```

---

## Deploying to GitHub

### First-time setup

**Step 1 — Create the repository on GitHub**

Go to [github.com/new](https://github.com/new), give it a name (e.g. `retainiq`), set it to Public or Private, and **do not** initialise with a README (you already have one).

**Step 2 — Open the project in VS Code**

Open the `retainiq/` folder in VS Code. Open the integrated terminal with `` Ctrl+` ``.

**Step 3 — Initialise Git and push**

```bash
git init
git add .
git commit -m "init: RetainIQ churn pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/retainiq.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### Subsequent pushes (after making changes)

```bash
git add .
git commit -m "your message here"
git push
```

### Cloning on another machine

```bash
git clone https://github.com/YOUR_USERNAME/retainiq.git
cd retainiq
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## License

MIT © RetainIQ"# RetainIQ" 
