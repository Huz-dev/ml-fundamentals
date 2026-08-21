import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from models import (
    train_knn,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
    train_svm,
)


def experiment_knn_k_values(X_train, y_train, X_val, y_val, k_values=(3, 7, 15)):
    """Compare multiple K values for KNN on validation accuracy/F1."""
    rows = []
    for k in k_values:
        model, train_time = train_knn(X_train, y_train, k=k)
        y_pred = model.predict(X_val)
        rows.append({
            "k": k,
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_f1": f1_score(y_val, y_pred, zero_division=0),
            "train_time_sec": train_time,
        })
    return pd.DataFrame(rows)


def experiment_tree_depth(X_train, y_train, X_val, y_val, depths=(2, None)):
    """
    Compare a shallow Decision Tree (depth=2, likely underfits) with
    an unrestricted deep tree (depth=None, likely overfits).
    Reports BOTH training and validation accuracy to reveal over/underfitting.
    """
    rows = []
    for depth in depths:
        model, train_time = train_decision_tree(X_train, y_train, max_depth=depth)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        rows.append({
            "max_depth": "None (unrestricted)" if depth is None else depth,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "gap (overfit signal)": train_acc - val_acc,
            "train_time_sec": train_time,
        })
    return pd.DataFrame(rows)


def experiment_tree_vs_forest(X_train, y_train, X_val, y_val):
    """Compare a single Decision Tree against a Random Forest."""
    tree_model, tree_time = train_decision_tree(X_train, y_train, max_depth=8)
    forest_model, forest_time = train_random_forest(X_train, y_train, n_estimators=200)

    rows = []
    for name, model, t in [
        ("Decision Tree (depth=8)", tree_model, tree_time),
        ("Random Forest (200 trees)", forest_model, forest_time),
    ]:
        y_pred = model.predict(X_val)
        rows.append({
            "model": name,
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_f1": f1_score(y_val, y_pred, zero_division=0),
            "train_time_sec": t,
        })
    return pd.DataFrame(rows), forest_model


def experiment_forest_vs_boosting(X_train, y_train, X_val, y_val):
    """Compare Random Forest against a boosting model (XGBoost)."""
    forest_model, forest_time = train_random_forest(X_train, y_train, n_estimators=200)
    xgb_model, xgb_time = train_xgboost(X_train, y_train, n_estimators=200)

    rows = []
    for name, model, t in [
        ("Random Forest", forest_model, forest_time),
        ("XGBoost", xgb_model, xgb_time),
    ]:
        y_pred = model.predict(X_val)
        rows.append({
            "model": name,
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_f1": f1_score(y_val, y_pred, zero_division=0),
            "train_time_sec": t,
        })
    return pd.DataFrame(rows)


def experiment_svm_scaling(
    X_train_raw, X_val_raw, X_train_scaled, X_val_scaled, y_train, y_val
):
    """
    Compare SVM trained on UNSCALED features vs SCALED features.
    SVM is distance-based, so we expect a meaningful performance gap.
    """
    model_unscaled, t_unscaled = train_svm(X_train_raw, y_train)
    model_scaled, t_scaled = train_svm(X_train_scaled, y_train)

    rows = []
    for name, model, X_v, t in [
        ("SVM - unscaled", model_unscaled, X_val_raw, t_unscaled),
        ("SVM - scaled", model_scaled, X_val_scaled, t_scaled),
    ]:
        y_pred = model.predict(X_v)
        rows.append({
            "model": name,
            "val_accuracy": accuracy_score(y_val, y_pred),
            "val_f1": f1_score(y_val, y_pred, zero_division=0),
            "train_time_sec": t,
        })
    return pd.DataFrame(rows)


def experiment_class_imbalance(y_train, y_val, y_test):
    """
    Summarize class imbalance across splits (no scaling/model needed --
    this is a descriptive experiment used to write the imbalance
    discussion in observations.md).
    """
    rows = []
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        counts = y.value_counts()
        rows.append({
            "split": name,
            "no_attrition_count": int(counts.get(0, 0)),
            "yes_attrition_count": int(counts.get(1, 0)),
            "pct_attrition": round(counts.get(1, 0) / len(y) * 100, 2),
        })
    return pd.DataFrame(rows)


def run_all_experiments(data):
    """
    Run every required experiment using the preprocessed data dict
    returned by data_preprocessing.preprocess_pipeline().
    Returns a dict of DataFrames, one per experiment.
    """
    X_train, X_val = data["X_train_scaled"], data["X_val_scaled"]
    X_train_raw, X_val_raw = data["X_train"], data["X_val"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    results = {}
    results["knn_k_comparison"] = experiment_knn_k_values(X_train, y_train, X_val, y_val)
    results["tree_depth_comparison"] = experiment_tree_depth(X_train, y_train, X_val, y_val)
    results["tree_vs_forest"], _ = experiment_tree_vs_forest(X_train, y_train, X_val, y_val)
    results["forest_vs_boosting"] = experiment_forest_vs_boosting(X_train, y_train, X_val, y_val)
    results["svm_scaling_comparison"] = experiment_svm_scaling(
        X_train_raw, X_val_raw, X_train, X_val, y_train, y_val
    )
    results["class_imbalance_summary"] = experiment_class_imbalance(y_train, y_val, y_test)

    return results
