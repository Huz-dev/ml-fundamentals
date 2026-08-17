# Observations — Assignment 1: Supervised ML Benchmark

Dataset: IBM HR Analytics Employee Attrition (1,470 rows, 35 columns → 34 features after dropping constants/ID).
Target: `Attrition` (Yes=1 / No=0), imbalanced at **~16.1% Yes / 83.9% No**.

## Data preparation decisions

- **Dropped columns**: `EmployeeCount`, `StandardHours`, `Over18` (constant across all rows — zero predictive value), `EmployeeNumber` (row identifier).
- **Missing values**: none found (0 nulls, 0 duplicates) — no imputation strategy was needed.
- **Categorical encoding**: one-hot encoding (not target encoding), chosen because cardinality is low (max 9 categories in `JobRole`) and one-hot keeps Logistic Regression coefficients directly interpretable and avoids any leakage risk from target-mean encoding on a small (1,470-row) dataset.
- **Split**: 70% train / 15% validation / 15% test, stratified on `Attrition` at every split to preserve the ~16% minority rate in all three sets (confirmed: train 16.05%, val 16.29%, test 16.29%).
- **Scaling**: `StandardScaler` fit only on the training set, then applied to val/test — avoids data leakage. Only applied for distance/margin-based models (Logistic Regression, KNN, SVM); tree-based models (Decision Tree, Random Forest, Gradient Boosting, XGBoost) were trained on unscaled data since splits are scale-invariant.

## Required experiment results

### KNN — comparing K values
| K | Val Accuracy | Val F1 |
|---|---|---|
| 3 | 0.8416 | 0.3137 |
| 7 | 0.8507 | 0.2326 |
| 15 | 0.8416 | 0.1026 |

Accuracy stays flat across K, but **F1 drops sharply as K increases** (0.31 → 0.10). Larger K oversmooths the decision boundary and votes are dominated by the majority class, so KNN increasingly fails to catch the minority (Attrition=Yes) cases. K=3 is the best tradeoff here.

### Decision Tree — shallow vs deep
| Max Depth | Train Acc | Val Acc | Gap |
|---|---|---|---|
| 2 | 0.8512 | 0.8326 | 0.0186 |
| None (unrestricted) | 1.0000 | 0.7330 | 0.2624 |

Depth=2 **underfits**: it can't capture enough structure (train and val accuracy are both mediocre and close together). The unrestricted tree **overfits**: 100% training accuracy but a 26-point drop on validation — it has memorized the training set rather than learning generalizable patterns. This pair directly satisfies the "show one underfit and one overfit model" requirement.

### Decision Tree vs Random Forest
| Model | Val Accuracy | Val F1 |
|---|---|---|
| Decision Tree (depth=8) | 0.7783 | 0.3636 |
| Random Forest (200 trees) | 0.8733 | 0.3913 |

Random Forest outperforms a single tree on both accuracy and F1. Averaging predictions across many decorrelated trees reduces the variance that a single deep tree suffers from.

### Random Forest vs Boosting (XGBoost)
| Model | Val Accuracy | Val F1 |
|---|---|---|
| Random Forest | 0.8733 | 0.3913 |
| XGBoost | 0.8688 | 0.4912 |

Random Forest has marginally higher accuracy, but **XGBoost has a meaningfully better F1 (0.49 vs 0.39)** — it does noticeably better at catching the minority class, since boosting explicitly focuses later trees on previously misclassified (often minority-class) examples. On an imbalanced problem, F1 is the more informative metric, so XGBoost is the stronger model here.

### SVM — with vs without scaling
| Model | Val Accuracy | Val F1 |
|---|---|---|
| SVM - unscaled | 0.8371 | **0.0000** |
| SVM - scaled | 0.8552 | 0.3333 |

This is the clearest result in the whole assignment. Without scaling, features like `MonthlyIncome` (range: thousands) completely dominate the distance calculation over features like `JobSatisfaction` (range: 1–4), so the model collapses to predicting the majority class for every row (F1 = 0, despite "reasonable" 83.7% accuracy — which is just the majority-class baseline). After scaling, SVM actually learns to identify some Attrition=Yes cases (F1 jumps to 0.33). **This demonstrates why distance-based models require scaling.**

### Class imbalance
| Split | No | Yes | % Attrition |
|---|---|---|---|
| Train | 863 | 165 | 16.05% |
| Val | 185 | 36 | 16.29% |
| Test | 185 | 36 | 16.29% |

With ~84% of rows being "No Attrition," a model that predicts "No" for every row scores ~84% accuracy while being useless. This is visible directly in the results: unscaled SVM and Random Forest both have high accuracy but very low recall/F1, because they lean toward the majority class. **Accuracy alone is misleading on this dataset — Precision, Recall, F1, and ROC-AUC are more trustworthy indicators of real performance.**

## Final test-set results (all 7 models)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time (s) | Inference (ms/sample) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8688 | 0.6522 | 0.4167 | **0.5085** | 0.8318 | 0.008 | 0.005 |
| KNN | 0.8462 | 0.6250 | 0.1389 | 0.2273 | 0.6362 | 0.002 | 0.021 |
| Decision Tree | 0.7964 | 0.3714 | 0.3611 | 0.3662 | 0.6211 | 0.012 | 0.005 |
| Random Forest | 0.8462 | 0.6667 | 0.1111 | 0.1905 | 0.7881 | 0.406 | 0.072 |
| Gradient Boosting | 0.8552 | 0.6250 | 0.2778 | 0.3846 | 0.8125 | 0.697 | 0.009 |
| XGBoost | 0.8552 | 0.6250 | 0.2778 | 0.3846 | 0.7622 | 0.181 | 0.021 |
| SVM | 0.8733 | **1.0000** | 0.2222 | 0.3636 | 0.8164 | 0.145 | 0.040 |

## Required observations — answered

- **Best model on training data (by accuracy)**: Decision Tree, Random Forest, and XGBoost all hit 1.0000 — but this reflects memorization, not skill.
- **Best model on validation data**: Random Forest and Gradient Boosting, both at 0.8733 accuracy — though XGBoost's F1 (0.49) was the strongest for actually catching Attrition cases.
- **Best model generalizing to the test set**: **Logistic Regression** — highest F1 (0.5085) and second-highest ROC-AUC (0.8318) on unseen data, with almost no train/val gap (0.0339), meaning it generalizes reliably rather than memorizing.
- **Most interpretable algorithm**: **Logistic Regression** (and to a lesser extent, a shallow Decision Tree) — coefficients map directly to feature effects, easy to explain to non-technical stakeholders (e.g., HR).
- **Fastest at inference**: Decision Tree and Logistic Regression, both ~0.005ms/sample.
- **Best pick if explainability is required**: **Logistic Regression** — interpretable coefficients, fast, and it's also the best-performing model on the test set, so there's no explainability/performance tradeoff needed here.
- **Best pick if predictive performance is the primary objective**: Depends on the metric — **XGBoost** for balanced Recall/F1 on the minority class, or **SVM** if Precision matters most (perfect precision, 1.0000, but weak recall). Given the business goal (catching at-risk employees), F1-oriented models (XGBoost, Logistic Regression) are more useful than accuracy-oriented ones.
- **Signs of high bias or high variance**: Decision Tree, Random Forest, and XGBoost all show **high variance (overfitting)** — perfect or near-perfect training accuracy with a 12–27 point drop to validation. Logistic Regression, KNN, and SVM show **reasonable fit** (train/val gap under 0.06), with no strong high-bias case in this run (the depth=2 tree, examined separately, mildly underfits — see the tree-depth experiment above).

## Final model selection

**Logistic Regression is the recommended final model.** It has the best F1-score and second-best ROC-AUC on the untouched test set, the smallest train/validation gap of any model (indicating genuine generalization rather than memorization), the fastest inference time, and full interpretability — a meaningful advantage for an HR use case where stakeholders need to understand *why* an employee is flagged as at-risk. Tree-based ensembles (Random Forest, XGBoost, Gradient Boosting) showed stronger raw capacity but consistently overfit on this relatively small (1,470-row) dataset; with more data or regularization tuning (reserved for Assignment 3), they could likely close the gap.
