# ML Fundamentals

![Python](https://img.shields.io/badge/python-3.12-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Practical machine learning assignments covering supervised learning, unsupervised learning, and (eventually) production-style ML engineering. Each assignment is self-contained, reproducible, and modular — one script per pipeline stage, with full result tables, plots, and a written justification of every modeling decision.

---

## Assignments

| # | Assignment | Focus | Status |
|---|---|---|---|
| 1 | [Supervised ML Benchmark](assignment_1_supervised/) | Classification, preprocessing, 7 core algorithms, evaluation | ✅ Complete |
| 2 | [Customer Segmentation](assignment_2_clustering/) | KMeans, DBSCAN, PCA, unsupervised analysis | ✅ Complete |
| 3 | Production-Style ML Service | Pipelines, tuning, FastAPI, Docker, monitoring | 🔜 Planned |

---

## Assignment 1 — Supervised ML Benchmark

Predicts employee attrition on the IBM HR Analytics dataset (1,470 employees), training and comparing **7 classification algorithms**: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and SVM.

**Highlights:**
- Unscaled SVM completely fails on the minority class (F1 = 0.000) — scaling fixes this (F1 = 0.333), demonstrating why distance-based models need it.
- Tree-based ensembles hit up to 100% training accuracy but drop sharply on validation — a clear overfitting signal documented alongside an underfitting shallow tree.
- **Logistic Regression** was selected as the final model: best F1 (0.509) and ROC-AUC (0.832) on the test set, smallest train/validation gap, and fully interpretable coefficients.

📄 Full write-up: [`assignment_1_supervised/observations.md`](assignment_1_supervised/observations.md)
📄 Setup & run instructions: [`assignment_1_supervised/README.md`](assignment_1_supervised/README.md)

---

## Assignment 2 — Customer Segmentation (KMeans, DBSCAN, PCA)

Segments 8,950 credit card customers by behavior (balance, purchases, cash advance, repayment, tenure) using unsupervised clustering, comparing how KMeans and DBSCAN structure the same data very differently.

**Highlights:**
- **KMeans (K=4)** found four balanced, interpretable segments — e.g. active regular spenders vs. high-value cash-advance revolvers.
- **DBSCAN** told a different story: one dense cluster holding 97% of customers, a few tiny extreme-behavior pockets, and 2.9% flagged as noise — and those noise points turned out to have the *highest* average spending and cash-advance activity of any group, a finding KMeans structurally cannot surface since it force-assigns every point to a cluster.
- **PCA**: the first 2 components explain 58% of total variance — enough to visualize meaningfully, with the caveat that ~42% of structure isn't shown in the 2D plots.

📄 Full write-up: [`assignment_2_clustering/observations.md`](assignment_2_clustering/observations.md)
📄 Setup & run instructions: [`assignment_2_clustering/README.md`](assignment_2_clustering/README.md)

---

## Repo structure

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

## Running any assignment

Each assignment has its own virtual environment and dependencies (kept separate so packages never clash between assignments):

```bash
cd assignment_1_supervised   # or assignment_2_clustering
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd src
python main.py
```

Each `main.py` runs its entire pipeline end-to-end and writes all results (Excel tables + plots) into that assignment's `results/` folder. See each assignment's own README for details specific to that pipeline.

> **macOS + XGBoost note (Assignment 1 only):** if you hit a `libomp.dylib` load error, run `brew install libomp` once, then re-run.

## Common design principles across assignments

- **Modular pipelines** — one script per stage (preprocessing, modeling, evaluation, experiments/analysis), orchestrated by a single `main.py`.
- **Reproducibility** — a fixed random seed throughout every pipeline.
- **No data leakage** — scalers and encoders are always fit on training data only, then applied to validation/test.
- **Documented reasoning, not just code** — every assignment ships an `observations.md` explaining *why* decisions were made (feature selection, final model/parameter choice) and what the results actually mean, not just what number came out.

## License

MIT
