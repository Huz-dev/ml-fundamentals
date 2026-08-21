import os
import pandas as pd

from data_preprocessing import preprocess_pipeline
from models import MODEL_REGISTRY
from evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_timing_comparison,
    plot_metric_comparison,
)
from experiments import run_all_experiments

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
EXCEL_PATH = os.path.join(RESULTS_DIR, "results_table.xlsx")

# Models that are distance/margin-based and therefore trained on SCALED data.
# Tree-based models are trained on unscaled (raw, but one-hot encoded) data
# since scaling has no effect on their splits -- this is documented in
# observations.md.
SCALE_SENSITIVE_MODELS = {"Logistic Regression", "KNN", "SVM"}


def train_all_models(data):
    """
    Train every model in MODEL_REGISTRY once on the full training set,
    using scaled data for distance/margin-based models and unscaled
    (one-hot only) data for tree-based / boosting models.

    Returns: fitted_models (dict), training_times (dict)
    """
    fitted_models = {}
    training_times = {}

    for name, train_fn in MODEL_REGISTRY.items():
        if name in SCALE_SENSITIVE_MODELS:
            X_train = data["X_train_scaled"]
        else:
            X_train = data["X_train"]

        print(f"Training {name}...")
        model, train_time = train_fn(X_train, data["y_train"])
        fitted_models[name] = model
        training_times[name] = train_time

    return fitted_models, training_times


def evaluate_all_models(fitted_models, training_times, data):
    """
    Evaluate every fitted model on train, validation, and test sets.
    Returns three DataFrames (train_results, val_results, test_results)
    and a dict of {model_name: y_pred_test} for confusion matrices.
    """
    train_rows, val_rows, test_rows = [], [], []
    test_preds = {}

    for name, model in fitted_models.items():
        if name in SCALE_SENSITIVE_MODELS:
            X_train, X_val, X_test = data["X_train_scaled"], data["X_val_scaled"], data["X_test_scaled"]
        else:
            X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]

        train_metrics, _ = compute_metrics(model, X_train, data["y_train"], name)
        val_metrics, _ = compute_metrics(model, X_val, data["y_val"], name)
        test_metrics, y_pred_test = compute_metrics(model, X_test, data["y_test"], name)

        train_metrics["training_time_sec"] = training_times[name]
        val_metrics["training_time_sec"] = training_times[name]
        test_metrics["training_time_sec"] = training_times[name]

        train_rows.append(train_metrics)
        val_rows.append(val_metrics)
        test_rows.append(test_metrics)
        test_preds[name] = y_pred_test

    cols = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "training_time_sec", "inference_time_sec", "inference_time_ms_per_sample"]

    train_df = pd.DataFrame(train_rows)[cols]
    val_df = pd.DataFrame(val_rows)[cols]
    test_df = pd.DataFrame(test_rows)[cols]

    return train_df, val_df, test_df, test_preds


def identify_bias_variance(train_df, val_df):
    """
    Flag models showing signs of high bias (underfit) or high variance
    (overfit) by comparing train vs validation accuracy.
    Rule of thumb used here:
      - gap > 0.10  -> high variance (overfitting)
      - both train & val accuracy low (<0.85) -> high bias (underfitting)
    """
    merged = train_df[["model", "accuracy"]].merge(
        val_df[["model", "accuracy"]], on="model", suffixes=("_train", "_val")
    )
    merged["gap"] = merged["accuracy_train"] - merged["accuracy_val"]

    def flag(row):
        if row["gap"] > 0.10:
            return "High variance (overfitting)"
        elif row["accuracy_train"] < 0.85 and row["accuracy_val"] < 0.85:
            return "High bias (underfitting)"
        else:
            return "Reasonable fit"

    merged["diagnosis"] = merged.apply(flag, axis=1)
    return merged


def main():
    os.makedirs(CM_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Preprocessing data")
    print("=" * 60)
    data = preprocess_pipeline(DATA_PATH)
    print(f"Train: {data['X_train'].shape}, Val: {data['X_val'].shape}, Test: {data['X_test'].shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Training all 7 models")
    print("=" * 60)
    fitted_models, training_times = train_all_models(data)

    print("\n" + "=" * 60)
    print("STEP 3: Evaluating all models (train / val / test)")
    print("=" * 60)
    train_df, val_df, test_df, test_preds = evaluate_all_models(fitted_models, training_times, data)
    print(test_df.round(4))

    print("\n" + "=" * 60)
    print("STEP 4: Saving confusion matrices")
    print("=" * 60)
    for name, y_pred in test_preds.items():
        safe_name = name.replace(" ", "_")
        save_path = os.path.join(CM_DIR, f"{safe_name}_confusion_matrix.png")
        plot_confusion_matrix(data["y_test"], y_pred, name, save_path)
        print(f"Saved: {save_path}")

    print("\n" + "=" * 60)
    print("STEP 5: Saving comparison plots")
    print("=" * 60)
    # ROC curves need models + their correct X_test (scaled or not)
    roc_models_scaled = {n: m for n, m in fitted_models.items() if n in SCALE_SENSITIVE_MODELS}
    roc_models_unscaled = {n: m for n, m in fitted_models.items() if n not in SCALE_SENSITIVE_MODELS}

    plot_roc_curves(roc_models_scaled, data["X_test_scaled"], data["y_test"],
                     os.path.join(PLOTS_DIR, "roc_curves_scaled_models.png"))
    plot_roc_curves(roc_models_unscaled, data["X_test"], data["y_test"],
                     os.path.join(PLOTS_DIR, "roc_curves_tree_models.png"))
    plot_timing_comparison(test_df, os.path.join(PLOTS_DIR, "timing_comparison.png"))
    plot_metric_comparison(test_df, os.path.join(PLOTS_DIR, "metric_comparison.png"))
    print("Plots saved to results/plots/")

    print("\n" + "=" * 60)
    print("STEP 6: Bias / variance diagnosis")
    print("=" * 60)
    bias_variance_df = identify_bias_variance(train_df, val_df)
    print(bias_variance_df.round(4))

    print("\n" + "=" * 60)
    print("STEP 7: Running required experiments")
    print("=" * 60)
    experiment_results = run_all_experiments(data)
    for exp_name, exp_df in experiment_results.items():
        print(f"\n--- {exp_name} ---")
        print(exp_df.round(4))

    print("\n" + "=" * 60)
    print("STEP 8: Writing results to Excel")
    print("=" * 60)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        train_df.round(4).to_excel(writer, sheet_name="train_results", index=False)
        val_df.round(4).to_excel(writer, sheet_name="val_results", index=False)
        test_df.round(4).to_excel(writer, sheet_name="test_results", index=False)
        bias_variance_df.round(4).to_excel(writer, sheet_name="bias_variance", index=False)
        for exp_name, exp_df in experiment_results.items():
            sheet_name = exp_name[:31]
            exp_df.round(4).to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved: {EXCEL_PATH}")

    print("\nDone. All results in the 'results/' folder.")

    return {
        "data": data,
        "fitted_models": fitted_models,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "bias_variance_df": bias_variance_df,
        "experiment_results": experiment_results,
    }


if __name__ == "__main__":
    main()
