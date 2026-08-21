"""
main.py

Orchestrates the entire Assignment 2 pipeline end to end:
  1. Preprocess data (clean, select features, scale)
  2. KMeans sweep across K=2..10, record inertia + silhouette,
     save elbow/silhouette plots, fit final K, save cluster profile
  3. DBSCAN sweep across multiple eps/min_samples, save results,
     fit final config, save cluster profile
  4. PCA: explained variance report + 2D visualizations of both
     KMeans and DBSCAN clusters
  5. Save everything into one Excel workbook + PNG plots

Run with:  python main.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_preprocessing import preprocess_pipeline
from kmeans_analysis import run_kmeans_sweep, fit_final_kmeans, cluster_profile as kmeans_profile
from dbscan_analysis import run_dbscan_sweep, fit_final_dbscan, cluster_profile as dbscan_profile
from pca_analysis import run_pca, explained_variance_report, plot_clusters_pca

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "CC_GENERAL.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
EXCEL_PATH = os.path.join(RESULTS_DIR, "cluster_profiles.xlsx")

# Final choices -- justified in observations.md after inspecting the sweeps
FINAL_K = 4
FINAL_EPS = 1.5
FINAL_MIN_SAMPLES = 5


def plot_elbow_and_silhouette(kmeans_sweep_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(kmeans_sweep_df["k"], kmeans_sweep_df["inertia"], marker="o")
    axes[0].set_xlabel("K (number of clusters)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method - Inertia vs K")

    axes[1].plot(kmeans_sweep_df["k"], kmeans_sweep_df["silhouette_score"], marker="o", color="darkorange")
    axes[1].set_xlabel("K (number of clusters)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs K")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Preprocessing data")
    print("=" * 60)
    data = preprocess_pipeline(DATA_PATH)
    print(f"Rows: {data['df_selected'].shape[0]}, Features used: {data['features_used']}")
    print(f"Nulls found & imputed: {data['summary']['null_counts']}")

    X_scaled = data["df_scaled"].values
    df_unscaled = data["df_selected"]

    print("\n" + "=" * 60)
    print("STEP 2: KMeans sweep (K=2..10)")
    print("=" * 60)
    kmeans_sweep_df = run_kmeans_sweep(X_scaled, k_range=range(2, 11))
    print(kmeans_sweep_df)
    plot_elbow_and_silhouette(kmeans_sweep_df, os.path.join(PLOTS_DIR, "kmeans_elbow_silhouette.png"))
    print("Saved elbow/silhouette plot.")

    print(f"\nFitting final KMeans with K={FINAL_K}")
    kmeans_model, kmeans_labels = fit_final_kmeans(X_scaled, FINAL_K)
    kmeans_cluster_profile_df = kmeans_profile(df_unscaled, kmeans_labels)
    print(kmeans_cluster_profile_df)

    print("\n" + "=" * 60)
    print("STEP 3: DBSCAN sweep (multiple eps / min_samples)")
    print("=" * 60)
    dbscan_sweep_df = run_dbscan_sweep(
        X_scaled,
        eps_values=(0.5, 1.0, 1.5, 2.0, 2.5),
        min_samples_values=(5, 10, 15, 20),
    )
    print(dbscan_sweep_df)

    print(f"\nFitting final DBSCAN with eps={FINAL_EPS}, min_samples={FINAL_MIN_SAMPLES}")
    dbscan_model, dbscan_labels = fit_final_dbscan(X_scaled, FINAL_EPS, FINAL_MIN_SAMPLES)
    dbscan_cluster_profile_df = dbscan_profile(df_unscaled, dbscan_labels)
    print(dbscan_cluster_profile_df)

    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = int((dbscan_labels == -1).sum())
    print(f"DBSCAN found {n_dbscan_clusters} clusters, {n_noise} noise points ({n_noise/len(dbscan_labels)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("STEP 4: PCA analysis")
    print("=" * 60)
    variance_df = explained_variance_report(X_scaled, max_components=8)
    print(variance_df)

    pca_2d_model, X_pca_2d = run_pca(X_scaled, n_components=2)
    top2_variance = variance_df.iloc[:2]["explained_variance_ratio"].sum()
    print(f"\nFirst 2 components explain {top2_variance*100:.2f}% of total variance")

    plot_clusters_pca(
        X_pca_2d, kmeans_labels,
        f"KMeans Clusters (K={FINAL_K}) in PCA Space",
        os.path.join(PLOTS_DIR, "pca_kmeans_clusters.png"),
    )
    plot_clusters_pca(
        X_pca_2d, dbscan_labels,
        f"DBSCAN Clusters (eps={FINAL_EPS}, min_samples={FINAL_MIN_SAMPLES}) in PCA Space",
        os.path.join(PLOTS_DIR, "pca_dbscan_clusters.png"),
        noise_label=-1,
    )
    print("Saved PCA cluster visualizations.")

    print("\n" + "=" * 60)
    print("STEP 5: Writing results to Excel")
    print("=" * 60)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        kmeans_sweep_df.round(4).to_excel(writer, sheet_name="kmeans_sweep", index=False)
        kmeans_cluster_profile_df.to_excel(writer, sheet_name="kmeans_cluster_profile", index=False)
        dbscan_sweep_df.to_excel(writer, sheet_name="dbscan_sweep", index=False)
        dbscan_cluster_profile_df.to_excel(writer, sheet_name="dbscan_cluster_profile", index=False)
        variance_df.to_excel(writer, sheet_name="pca_explained_variance", index=False)
    print(f"Saved: {EXCEL_PATH}")

    print("\nDone. All results in the 'results/' folder.")

    return {
        "data": data,
        "kmeans_sweep_df": kmeans_sweep_df,
        "kmeans_labels": kmeans_labels,
        "kmeans_cluster_profile_df": kmeans_cluster_profile_df,
        "dbscan_sweep_df": dbscan_sweep_df,
        "dbscan_labels": dbscan_labels,
        "dbscan_cluster_profile_df": dbscan_cluster_profile_df,
        "variance_df": variance_df,
    }


if __name__ == "__main__":
    main()
