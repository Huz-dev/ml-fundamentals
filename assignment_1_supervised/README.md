# Assignment 1 — Supervised ML Benchmark

Classification benchmark on the **IBM HR Analytics Employee Attrition** dataset (1,470 rows), comparing 7 supervised learning algorithms with full preprocessing, evaluation, and required experiments.

## Dataset

- Source: IBM HR Analytics Employee Attrition & Performance (Kaggle)
- File: `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- 1,470 rows, 35 columns (34 usable features after dropping constants/ID column)
- Target: `Attrition` (Yes/No) — imbalanced, ~16.1% Yes

## Project structure

```
assignment_1_supervised/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── src/
│   ├── data_preprocessing.py   # load, clean, encode, scale, split
│   ├── models.py                # training function per model
│   ├── evaluate.py              # metrics, confusion matrices, plots
│   ├── experiments.py           # required comparisons (KNN K's, tree depth, etc.)
│   └── main.py                  # runs the full pipeline end-to-end
├── results/
│   ├── results_table.xlsx       # all metrics + all experiments, one sheet each
│   ├── confusion_matrices/      # PNG per model
│   └── plots/                   # ROC curves, timing, metric comparison charts
├── requirements.txt
├── README.md
└── observations.md              # model-selection reasoning & conclusions
```

## Installation

```bash
cd assignment_1_supervised
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

```bash
cd src
python main.py
```

This runs the entire pipeline in one go:
1. Preprocesses the data (clean → encode → split → scale)
2. Trains all 7 models: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM
3. Evaluates every model on train / validation / test sets
4. Saves confusion matrices (`results/confusion_matrices/`) and comparison plots (`results/plots/`)
5. Runs all required experiments (KNN K comparison, tree depth, tree vs forest, forest vs boosting, SVM scaling, class imbalance summary)
6. Writes everything to `results/results_table.xlsx` (one sheet per result set)

Random seed is fixed (`RANDOM_SEED = 42`) throughout for reproducibility.

## Models

Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM.

Distance/margin-based models (Logistic Regression, KNN, SVM) are trained on **scaled** features; tree-based models (Decision Tree, Random Forest, Gradient Boosting, XGBoost) are trained on **unscaled** (one-hot encoded only) features, since tree splits are scale-invariant.

## Results

See `results/results_table.xlsx` for full metrics (accuracy, precision, recall, F1, ROC-AUC, training time, inference time) across train/val/test, plus every required experiment as its own sheet.

See `observations.md` for the full write-up: experiment interpretation, bias/variance diagnosis, and final model-selection reasoning.

**Summary**: Logistic Regression was selected as the final model — best F1 and ROC-AUC on the test set, smallest train/validation gap (best generalization), fastest inference, and fully interpretable coefficients. Tree ensembles (Random Forest, XGBoost, Gradient Boosting) showed stronger raw training capacity but overfit on this relatively small dataset.
