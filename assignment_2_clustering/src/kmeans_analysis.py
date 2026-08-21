"""
kmeans_analysis.py

Runs KMeans across a range of K values, records inertia and silhouette
score for each, and provides functions to fit a final chosen K and
produce per-cluster descriptive statistics.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_SEED = 42


def run_kmeans_sweep(X_scaled, k_range=range(2, 11)):
    """
    Fit KMeans for every K in k_range.
    Returns a DataFrame with columns: k, inertia, silhouette_score.
    """
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": sil,
        })
    return pd.DataFrame(rows)


def fit_final_kmeans(X_scaled, k):
    """Fit KMeans with the chosen final K. Returns (model, labels)."""
    model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels


def cluster_profile(df_unscaled, labels, cluster_col_name="cluster"):
    """
    Compute descriptive statistics (mean of every feature) per cluster,
    using UNSCALED values so the numbers are human-readable
    (e.g. actual dollar amounts, not z-scores).
    Also returns cluster sizes.
    """
    profile_df = df_unscaled.copy()
    profile_df[cluster_col_name] = labels

    means = profile_df.groupby(cluster_col_name).mean().round(2)
    sizes = profile_df[cluster_col_name].value_counts().sort_index()
    means["cluster_size"] = sizes
    means["cluster_pct"] = (sizes / len(profile_df) * 100).round(2)

    return means.reset_index()
