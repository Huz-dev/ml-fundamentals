# Employee Attrition — Supervised ML Benchmark

![Python](https://img.shields.io/badge/python-3.12-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A complete, reproducible classification benchmark comparing **7 supervised learning algorithms** on the IBM HR Analytics Employee Attrition dataset — from raw CSV to a justified final model selection.

> Predicting employee attrition from HR data: preprocessing, 7 trained models, required experiments (KNN K-tuning, tree depth, scaling effects, imbalance analysis), full evaluation, and a documented final recommendation.

---

## Table of Contents

- [Overview](#overview)
- [Results at a Glance](#results-at-a-glance)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Methodology](#methodology)
- [Outputs](#outputs)
- [Final Model](#final-model)
- [License](#license)

---

## Overview

This project builds an end-to-end supervised ML pipeline:

**Preprocess → Train 7 models → Evaluate → Run required experiments → Select & justify final model**

Everything is modular (`src/`), reproducible (fixed random seed throughout), and leakage-free (scaler fit only on training data).

## Results at a Glance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.869 | 0.652 | 0.417 | **0.509** | 0.832 |
| SVM (scaled) | 0.873 | **1.000** | 0.222 | 0.364 | 0.816 |
| Gradient Boosting | 0.855 | 0.625 | 0.278 | 0.385 | 0.813 |
| XGBoost | 0.855 | 0.625 | 0.278 | 0.385 | 0.762 |
| Random Forest | 0.846 | 0.667 | 0.111 | 0.191 | 0.788 |
| KNN | 0.846 | 0.625 | 0.139 | 0.227 | 0.636 |
| Decision Tree | 0.796 | 0.371 | 0.361 | 0.366 | 0.621 |

*(Test set, 221 held-out employees, 16.3% attrition rate)*

⭐ **Logistic Regression** was selected as the final model — see [Final Model](#final-model).

## Key Findings

- 🎯 **Scaling matters — a lot.** Unscaled SVM completely fails on the minority class (F1 = 0.000, silently predicting "No Attrition" for everyone). Scaling the same features gets F1 to 0.333.
- 📉 **Deep trees overfit visibly.** An unrestricted Decision Tree hits 100% training accuracy but drops 26 points on validation. A depth-2 tree underfits instead — both are documented side by side.
- ⚖️ **Accuracy is misleading on imbalanced data.** With ~84% "No Attrition," a model can score 84%+ accuracy while catching almost none of the actual at-risk employees — F1 and Recall tell the real story here.
- 🏆 **Simplicity won.** Logistic Regression outperformed every tree-based ensemble on the test set, with the smallest train/validation gap (best generalization) and the fastest inference time.

## Project Structure

```
assignment_1_supervised/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── src/
│   ├── data_preprocessing.py   # load → clean → encode → split → scale
│   ├── models.py               # one training function per algorithm
│   ├── evaluate.py             # metrics, confusion matrices, comparison plots
│   ├── experiments.py          # required comparisons (KNN K's, tree depth, etc.)
│   └── main.py                 # runs the full pipeline end-to-end
├── results/
│   ├── results_table.xlsx      # every metric + every experiment, one sheet each
│   ├── confusion_matrices/     # one PNG per model
│   └── plots/                  # ROC curves, timing, metric comparisons
├── requirements.txt
├── observations.md             # full experiment write-up + model justification
└── README.md
```

## Installation

```bash
git clone <your-repo-url>
cd assignment_1_supervised
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **macOS users:** XGBoost requires the OpenMP runtime. If you hit a `libomp.dylib` error, run `brew install libomp` first.

## Usage

```bash
cd src
python main.py
```

Runs the full pipeline in one command — preprocessing, training all 7 models, evaluation, plots, required experiments, and the Excel results file. Takes under a minute. Outputs land in `results/`.

## Dataset

**IBM HR Analytics Employee Attrition** — 1,470 employees, 35 original columns, target `Attrition` (Yes/No).

- 4 constant/identifier columns dropped (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`)
- No missing values, no duplicates
- Imbalanced target: 16.1% Yes / 83.9% No
- Mixed numerical (Age, MonthlyIncome, DistanceFromHome...) and categorical (Department, JobRole, OverTime...) features

## Models

Logistic Regression · KNN · Decision Tree · Random Forest · Gradient Boosting · XGBoost · SVM

Distance/margin-based models (Logistic Regression, KNN, SVM) train on **scaled** features; tree-based models train on **unscaled** (one-hot only) features, since tree splits are scale-invariant.

## Methodology

- **Split:** 70% train / 15% validation / 15% test, stratified on target to preserve the imbalance ratio in every split
- **Encoding:** one-hot encoding for categoricals (max cardinality: 9) — chosen over target encoding to avoid leakage risk on a small dataset and to keep Logistic Regression coefficients interpretable
- **Scaling:** `StandardScaler` fit only on training data, applied to val/test — no leakage
- **Required experiments:** KNN K-value comparison, shallow vs. deep Decision Tree, Decision Tree vs. Random Forest, Random Forest vs. XGBoost, SVM with vs. without scaling, class imbalance analysis

Full reasoning and all experiment tables: [`observations.md`](observations.md)

## Outputs

- `results/results_table.xlsx` — 10 sheets: train/val/test metrics, bias-variance diagnosis, all 6 experiments
- `results/confusion_matrices/*.png` — per-model confusion matrices
- `results/plots/` — ROC curves, training/inference timing, metric comparison bar charts

## Final Model

**Logistic Regression**, selected for:
- Best F1-score (0.509) and second-best ROC-AUC (0.832) on the held-out test set
- Smallest train/validation gap of any model — genuine generalization, not memorization
- Fully interpretable coefficients — meaningful for HR stakeholders who need to understand *why* someone is flagged
- Fastest inference (~0.005ms/sample)

Full justification and answers to every required observation question: [`observations.md`](observations.md)

## License

MIT
