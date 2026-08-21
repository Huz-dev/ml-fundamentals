# ML Fundamentals

![Python](https://img.shields.io/badge/python-3.12-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Practical machine learning assignments covering supervised learning, unsupervised learning, and (eventually) production-style ML engineering. Each assignment is modular and reproducible — one script per pipeline stage, full result tables and plots, and a written justification of every modeling decision.

---

## Table of Contents

- [Repo Structure](#repo-structure)
- [Assignment 1 — Supervised ML Benchmark](#assignment-1--supervised-ml-benchmark)
- [Assignment 2 — Customer Segmentation](#assignment-2--customer-segmentation-kmeans-dbscan-pca)
- [Assignment 3 — Production-Style ML Service](#assignment-3--production-style-ml-service)
- [Running Any Assignment](#running-any-assignment)
- [Shared Design Principles](#shared-design-principles)
- [License](#license)

---

## Repo Structure

```
ml-fundamentals/
├── assignment_1_supervised/
│   ├── data/
│   ├── src/                 # data_preprocessing, models, evaluate, experiments, main
│   ├── results/              # results_table.xlsx, confusion matrices, plots
│   ├── README.md
│   └── observations.md
│
├── assignment_2_clustering/
│   ├── data/
│   ├── src/                 # data_preprocessing, kmeans_analysis, dbscan_analysis, pca_analysis, main
│   ├── results/              # cluster_profiles.xlsx, plots
│   ├── README.md
│   └── observations.md
│
└── README.md                 # you are here
```

---

## Assignment 1 — Supervised ML Benchmark

**Goal:** Predict employee attrition using 7 classification algorithms on the IBM HR Analytics dataset.

**Dataset:** 1,470 employees, 35 columns → 34 usable features after dropping constants/ID. Target `Attrition` (Yes/No), imbalanced at **16.1% Yes / 83.9% No**. No missing values, no duplicates.

**Preprocessing:** one-hot encoding for categoricals (max cardinality 9), 70/15/15 stratified train/val/test split, `StandardScaler` fit only on training data (no leakage). Distance-based models (Logistic Regression, KNN, SVM) trained on scaled features; tree-based models trained on unscaled features.

### Final test-set results (all 7 models)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time (s) | Inference (ms/sample) |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.8688 | 0.6522 | 0.4167 | **0.5085** | 0.8318 | 0.008 | 0.005 |
| SVM (scaled) | 0.8733 | **1.0000** | 0.2222 | 0.3636 | 0.8164 | 0.145 | 0.040 |
| Gradient Boosting | 0.8552 | 0.6250 | 0.2778 | 0.3846 | 0.8125 | 0.697 | 0.009 |
| XGBoost | 0.8552 | 0.6250 | 0.2778 | 0.3846 | 0.7622 | 0.181 | 0.021 |
| Random Forest | 0.8462 | 0.6667 | 0.1111 | 0.1905 | 0.7881 | 0.406 | 0.072 |
| KNN | 0.8462 | 0.6250 | 0.1389 | 0.2273 | 0.6362 | 0.002 | 0.021 |
| Decision Tree | 0.7964 | 0.3714 | 0.3611 | 0.3662 | 0.6211 | 0.012 | 0.005 |

### Required experiment results

**KNN — K comparison**
| K | Val Accuracy | Val F1 |
|---|---|---|
| 3 | 0.8416 | 0.3137 |
| 7 | 0.8507 | 0.2326 |
| 15 | 0.8416 | 0.1026 |

**Decision Tree — shallow vs deep**
| Max Depth | Train Acc | Val Acc | Gap |
|---|---|---|---|
| 2 | 0.8512 | 0.8326 | 0.0186 (underfit) |
| None | 1.0000 | 0.7330 | 0.2624 (overfit) |

**Decision Tree vs Random Forest**
| Model | Val Accuracy | Val F1 |
|---|---|---|
| Decision Tree (depth=8) | 0.7783 | 0.3636 |
| Random Forest (200 trees) | 0.8733 | 0.3913 |

**Random Forest vs Boosting**
| Model | Val Accuracy | Val F1 |
|---|---|---|
| Random Forest | 0.8733 | 0.3913 |
| XGBoost | 0.8688 | **0.4912** |

**SVM — with vs without scaling**
| Model | Val Accuracy | Val F1 |
|---|---|---|
| SVM - unscaled | 0.8371 | **0.0000** |
| SVM - scaled | 0.8552 | 0.3333 |

### Key findings

- 🎯 **Scaling is not optional for SVM.** Unscaled, SVM collapses to predicting "No Attrition" for every employee (F1 = 0.000) because high-magnitude features like `MonthlyIncome` dominate the distance calculation. Scaled, it recovers to F1 = 0.333.
- 📉 **Tree-based models overfit visibly.** Decision Tree, Random Forest, and XGBoost all hit ~100% training accuracy but drop 12–27 points on validation — a textbook high-variance signature on this relatively small (1,470-row) dataset.
- ⚖️ **Accuracy is misleading here.** With ~84% "No Attrition," several models score 84%+ accuracy while barely catching any real attrition cases (Recall as low as 0.11) — F1 and Recall tell the real story.
- 🏆 **Logistic Regression won on the test set** — best F1 (0.509), second-best ROC-AUC (0.832), smallest train/val gap of any model, fastest inference, and fully interpretable coefficients.

**Final model: Logistic Regression** — best generalization, best explainability, no performance tradeoff required.

📄 Full write-up: [`assignment_1_supervised/observations.md`](assignment_1_supervised/observations.md) · 📄 Setup: [`assignment_1_supervised/README.md`](assignment_1_supervised/README.md)

---

## Assignment 2 — Customer Segmentation (KMeans, DBSCAN, PCA)

**Goal:** Segment credit card customers by behavior using unsupervised clustering, and compare how KMeans and DBSCAN structure the same data.

**Dataset:** `CC_GENERAL.csv`, 8,950 customers, 17 numeric features available. 8 selected for interpretability: `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, `PURCHASES_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`, `TENURE`. Missing values in `CREDIT_LIMIT` (1) and `MINIMUM_PAYMENTS` (313) imputed with median. All features scaled with `StandardScaler` before clustering.

### KMeans — choosing K

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 56,231 | **0.418** |
| 3 | 48,371 | 0.228 |
| **4** | **41,650** | **0.261** |
| 5 | 35,821 | 0.271 |
| 6 | 32,192 | 0.264 |
| 7 | 29,147 | 0.271 |
| 8 | 27,066 | 0.256 |
| 9 | 25,289 | 0.253 |
| 10 | 24,060 | 0.253 |

K=2 has the highest silhouette but is too coarse to be actionable. **K=4 chosen** as the practical balance between cluster quality and usable segment granularity.

### KMeans cluster profile (K=4)

| Cluster | Label | Balance | Purchases | Cash Advance | Purchase Freq | Cash Adv Freq | Tenure | Size (%) |
|---|---|---|---|---|---|---|---|---|
| 0 | Active regular spenders | 971 | 1,622 | 165 | 0.87 | 0.03 | 11.9 | 39.7% |
| 1 | High-value revolvers | 5,533 | 1,992 | 4,594 | 0.42 | 0.42 | 11.7 | 11.7% |
| 2 | Newer, low-engagement | 837 | 433 | 1,037 | 0.43 | 0.19 | 7.4 | 8.0% |
| 3 | Long-tenured, low-engagement | 1,144 | 224 | 720 | 0.15 | 0.14 | 11.9 | 40.5% |

### DBSCAN — sensitivity to eps / min_samples

| eps | min_samples | Clusters | Noise % | Silhouette |
|---|---|---|---|---|
| 0.5 | 5 | 35 | 31.8% | -0.19 |
| 1.0 | 5 | 7 | 9.0% | 0.26 |
| **1.5** | **5** | **4** | **2.9%** | **0.36** |
| 1.5 | 10+ | 1 | 3.9–5.2% | N/A |
| 2.0 | any | 1 | 1.5–2.2% | N/A |
| 2.5 | 5 | 2 | 0.7% | 0.64 |

DBSCAN is **highly sensitive** to both parameters — cluster count ranges from 35 down to 1 depending on the combination. **Final config: eps=1.5, min_samples=5** — smallest eps giving multiple genuine clusters with low noise and the best silhouette among multi-cluster configs.

### DBSCAN cluster profile

| Cluster | Balance | Purchases | Cash Advance | Payments | Size (%) |
|---|---|---|---|---|---|
| **-1 (noise)** | **5,625** | **5,193** | **6,212** | **11,469** | 2.9% |
| 0 (main cluster) | 1,441 | 869 | 820 | 1,431 | **97.0%** |
| 1 | 1,739 | 13,534 | 0 | 14,761 | 0.06% |
| 2 | 4,077 | 1,885 | 4,146 | 790 | 0.04% |
| 3 | 2,677 | 2,219 | 3,768 | 8,412 | 0.03% |

### PCA — explained variance

| Component | Variance | Cumulative |
|---|---|---|
| PC1 | 33.81% | 33.81% |
| PC2 | 24.24% | **58.05%** |
| PC3 | 12.52% | 70.58% |
| PC4 | 9.37% | 79.94% |

First 2 components explain **58.05%** of total variance — enough to visualize meaningfully, but ~42% of structure isn't shown in the 2D plots.

### Key findings

- 🎯 **KMeans found 4 balanced, marketable segments** — differentiated mainly by cash-advance reliance, purchase frequency, and tenure.
- 🔍 **DBSCAN told a different story**: 97% of customers fall into one dense "typical" cluster, with a few tiny pockets of extreme behavior and **2.9% flagged as noise**.
- 💡 **The noise points were the most interesting finding**: they have the *highest* average balance, purchases, and cash advance of any group — higher even than the small extreme clusters. KMeans structurally cannot surface this, since it force-assigns every point to one of its K clusters regardless of fit.
- 📊 **KMeans → better for building an actionable segmentation scheme.** **DBSCAN → better for flagging outlier/high-value customers** worth individual review.

📄 Full write-up: [`assignment_2_clustering/observations.md`](assignment_2_clustering/observations.md) · 📄 Setup: [`assignment_2_clustering/README.md`](assignment_2_clustering/README.md)

---

## Assignment 3 — Production-Style ML Service

🔜 **Planned.** Will take a model from Assignment 1 and turn it into a locally deployable ML service: Scikit-learn Pipeline (preprocessing + model as one object), hyperparameter tuning, model persistence with metadata, a FastAPI serving layer (`/health`, `/predict`), Dockerized deployment, basic request/latency monitoring, and a data-drift simulation.

---

## Running Any Assignment

Each assignment has its own virtual environment and dependencies, kept separate so packages never clash:

```bash
cd assignment_1_supervised   # or assignment_2_clustering
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd src
python main.py
```

Each `main.py` runs its entire pipeline end-to-end and writes all results (Excel tables + plots) into that assignment's `results/` folder.

> **macOS + XGBoost note (Assignment 1 only):** if you hit a `libomp.dylib` load error, run `brew install libomp` once, then re-run.

## Shared Design Principles

- **Modular pipelines** — one script per stage (preprocessing, modeling, evaluation, experiments/analysis), orchestrated by a single `main.py`.
- **Reproducibility** — a fixed random seed throughout every pipeline.
- **No data leakage** — scalers and encoders are always fit on training data only, then applied to validation/test.
- **Documented reasoning, not just code** — every assignment ships an `observations.md` explaining *why* decisions were made (feature selection, final model/parameter choice) and what the results actually mean.

## License

MIT
