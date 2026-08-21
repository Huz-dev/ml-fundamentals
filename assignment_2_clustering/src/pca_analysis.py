"""
pca_analysis.py

Applies PCA to the scaled features for visualization purposes, and
provides plotting functions for KMeans / DBSCAN clusters projected
into 2D PCA space.

Note: a 2-component PCA plot is a PROJECTION of higher-dimensional data.
It captures as much variance as possible in 2 dimensions, but it can
still lose real structure that exists in the original feature space --
two points that look close in the PCA plot may not actually be close
in full 8-dimensional space, and vice versa.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def run_pca(X_scaled, n_components=2):
    """
    Fit PCA on scaled features. Returns (pca_model, transformed_array).
    Also usable with n_components > 2 to inspect fuller explained variance.
    """
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(X_scaled)
    return pca, transformed


def explained_variance_report(X_scaled, max_components=8):
    """
    Fit PCA with all components to see how variance is distributed,
    so we can report explained variance ratio per component and
    cumulative variance.
    """
    pca_full = PCA(n_components=min(max_components, X_scaled.shape[1]), random_state=42)
    pca_full.fit(X_scaled)

    ratios = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    import pandas as pd
    return pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(ratios))],
        "explained_variance_ratio": ratios.round(4),
        "cumulative_variance": cumulative.round(4),
    })


def plot_clusters_pca(pca_2d, labels, title, save_path, noise_label=None):
    """
    Scatter plot of points in 2D PCA space, colored by cluster label.
    If noise_label is given (e.g. -1 for DBSCAN), those points are
    drawn in gray and marked distinctly.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        if noise_label is not None and lab == noise_label:
            ax.scatter(
                pca_2d[mask, 0], pca_2d[mask, 1],
                c="lightgray", label="Noise", s=15, alpha=0.6, marker="x",
            )
        else:
            ax.scatter(
                pca_2d[mask, 0], pca_2d[mask, 1],
                c=[cmap(i % 10)], label=f"Cluster {lab}", s=15, alpha=0.7,
            )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
