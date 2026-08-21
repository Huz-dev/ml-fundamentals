"""
dbscan_analysis.py

Runs DBSCAN across multiple eps/min_samples configurations, reports
cluster counts and noise-point counts for each, and provides a function
to fit a final chosen configuration.

Unlike KMeans, DBSCAN does not require the number of clusters in advance --
it discovers clusters as dense regions of points, and labels points in
low-density regions as noise (-1). This makes it well suited to finding
irregularly shaped clusters and flagging outlier customers.
"""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def run_dbscan_sweep(X_scaled, eps_values=(0.5, 1.0, 1.5, 2.0), min_samples_values=(5, 10, 15)):
    """
    Try every combination of eps and min_samples.
    Returns a DataFrame with columns: eps, min_samples, n_clusters,
    n_noise, noise_pct, silhouette_score (computed excluding noise points,
    only when there are 2+ real clusters -- otherwise NaN).
    """
    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())
            noise_pct = round(n_noise / len(labels) * 100, 2)

            sil = None
            if n_clusters >= 2:
                mask = labels != -1
                if mask.sum() > 1:
                    try:
                        sil = silhouette_score(X_scaled[mask], labels[mask])
                    except Exception:
                        sil = None

            rows.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": noise_pct,
                "silhouette_score": round(sil, 4) if sil is not None else None,
            })

    return pd.DataFrame(rows)


def fit_final_dbscan(X_scaled, eps, min_samples):
    """Fit DBSCAN with the chosen final eps/min_samples. Returns (model, labels)."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    return model, labels


def cluster_profile(df_unscaled, labels, cluster_col_name="cluster"):
    """
    Same idea as KMeans cluster_profile: mean of every feature per cluster,
    using unscaled values. Cluster -1 represents noise points.
    """
    profile_df = df_unscaled.copy()
    profile_df[cluster_col_name] = labels

    means = profile_df.groupby(cluster_col_name).mean().round(2)
    sizes = profile_df[cluster_col_name].value_counts().sort_index()
    means["cluster_size"] = sizes
    means["cluster_pct"] = (sizes / len(profile_df) * 100).round(2)

    return means.reset_index()
