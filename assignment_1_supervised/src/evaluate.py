"""
evaluate.py

Evaluation utilities used across all models:
  - compute_metrics(): accuracy, precision, recall, F1, ROC-AUC, inference time
  - plot_confusion_matrix(): saves a PNG per model
  - plot_roc_curve(): saves a combined ROC comparison plot
  - plot_timing_comparison(): bar chart of train/inference time per model
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def compute_metrics(model, X, y, model_name="model"):
    """
    Run predictions on X, compare to y, and return a dict of metrics
    plus inference time (in seconds, for the whole batch, and per-sample ms).
    """
    start = time.perf_counter()
    y_pred = model.predict(X)
    elapsed = time.perf_counter() - start

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "inference_time_sec": elapsed,
        "inference_time_ms_per_sample": (elapsed / len(X)) * 1000,
    }

    # ROC-AUC needs probability scores; not all models expose predict_proba
    # in the same way, so guard against failure.
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X)
        else:
            y_proba = None

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y, y_proba)
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan

    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Save a confusion matrix heatmap for a single model."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Attrition", "Attrition"])
    ax.set_yticklabels(["No Attrition", "Attrition"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(models_dict, X_test, y_test, save_path):
    """
    Plot ROC curves for every model that supports probability/decision scores
    on one combined chart for easy comparison.
    models_dict: {model_name: fitted_model}
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models_dict.items():
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_score)
            auc = roc_auc_score(y_test, y_score)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models (Test Set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_timing_comparison(results_df, save_path):
    """Bar chart comparing training time and inference time per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(results_df["model"], results_df["training_time_sec"], color="steelblue")
    axes[0].set_title("Training Time by Model")
    axes[0].set_ylabel("Seconds")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(results_df["model"], results_df["inference_time_ms_per_sample"], color="darkorange")
    axes[1].set_title("Inference Time per Sample by Model")
    axes[1].set_ylabel("Milliseconds")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(results_df, save_path):
    """Grouped bar chart comparing accuracy/precision/recall/F1 across models."""
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(results_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["model"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Core Metrics (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
