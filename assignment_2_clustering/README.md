# Assignment 2 — Customer Segmentation with KMeans, DBSCAN, and PCA

Unsupervised segmentation of 8,950 credit card customers using behavioral features, comparing KMeans and DBSCAN, and visualizing results with PCA.

## Dataset

- Source: `CC_GENERAL.csv` — credit card customer behavioral dataset
- 8,950 customers, 17 original numeric features (`CUST_ID` dropped as identifier)
- 8 features selected for clustering (chosen for interpretability): `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, `PURCHASES_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`, `TENURE`
- Missing values in `CREDIT_LIMIT` and `MINIMUM_PAYMENTS` imputed with median

## Project structure

```
assignment_2_clustering/
├── data/
│   └── CC_GENERAL.csv
├── src/
│   ├── data_preprocessing.py   # load, clean, select features, scale
│   ├── kmeans_analysis.py      # K sweep, inertia/silhouette, cluster profiling
│   ├── dbscan_analysis.py      # eps/min_samples sweep, cluster profiling
│   ├── pca_analysis.py         # explained variance, 2D projection, plotting
│   └── main.py                 # runs the full pipeline end-to-end
├── results/
│   ├── cluster_profiles.xlsx   # KMeans sweep, DBSCAN sweep, cluster profiles, PCA variance
│   └── plots/
│       ├── kmeans_elbow_silhouette.png
│       ├── pca_kmeans_clusters.png
│       └── pca_dbscan_clusters.png
├── requirements.txt
├── README.md
└── observations.md             # full experiment write-up & interpretation
```

## Installation

```bash
cd assignment_2_clustering
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
1. Preprocesses the data (clean → select 8 features → scale)
2. Sweeps KMeans across K=2 to 10, records inertia + silhouette, saves elbow/silhouette plot, fits final K=4, computes cluster profile
3. Sweeps DBSCAN across multiple eps/min_samples combinations, fits final config (eps=1.5, min_samples=5), computes cluster profile
4. Runs PCA, reports explained variance, saves 2D visualizations of both KMeans and DBSCAN clusters
5. Writes everything to `results/cluster_profiles.xlsx` (5 sheets) and saves all plots to `results/plots/`

Random seed fixed (`RANDOM_SEED = 42`) throughout for reproducibility.

## Results summary

- **KMeans (K=4)**: found 4 reasonably balanced segments (8-40% of customers each), differentiated mainly by cash-advance reliance, purchase frequency, and tenure.
- **DBSCAN (eps=1.5, min_samples=5)**: found one dominant cluster (97% of customers) plus 3 tiny extreme-behavior pockets and 2.9% noise — the noise points had the highest average balance/spending/cash-advance of any group, making them useful outlier flags.
- **PCA**: first 2 components explain 58.05% of total variance.

Full reasoning, all sweep tables, and answers to every required observation: see [`observations.md`](observations.md).
