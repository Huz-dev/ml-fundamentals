
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_SEED = 42


def _timed_fit(model, X_train, y_train):
    """Helper: fit a model and return (model, elapsed_seconds)."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return model, elapsed


def train_logistic_regression(X_train, y_train, **kwargs):
    model = LogisticRegression(
        max_iter=1000, random_state=RANDOM_SEED, **kwargs
    )
    return _timed_fit(model, X_train, y_train)


def train_knn(X_train, y_train, k=5, **kwargs):
    model = KNeighborsClassifier(n_neighbors=k, **kwargs)
    return _timed_fit(model, X_train, y_train)


def train_decision_tree(X_train, y_train, max_depth=None, **kwargs):
    model = DecisionTreeClassifier(
        max_depth=max_depth, random_state=RANDOM_SEED, **kwargs
    )
    return _timed_fit(model, X_train, y_train)


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=None, **kwargs):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        **kwargs,
    )
    return _timed_fit(model, X_train, y_train)


def train_gradient_boosting(X_train, y_train, n_estimators=200, learning_rate=0.1, **kwargs):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=RANDOM_SEED,
        **kwargs,
    )
    return _timed_fit(model, X_train, y_train)


def train_xgboost(X_train, y_train, n_estimators=200, learning_rate=0.1, **kwargs):
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        **kwargs,
    )
    return _timed_fit(model, X_train, y_train)


def train_svm(X_train, y_train, kernel="rbf", probability=True, **kwargs):
    # probability=True is needed so we can compute ROC-AUC later
    model = SVC(kernel=kernel, probability=probability, random_state=RANDOM_SEED, **kwargs)
    return _timed_fit(model, X_train, y_train)


# Registry so main.py / experiments.py can loop over all models generically
MODEL_REGISTRY = {
    "Logistic Regression": train_logistic_regression,
    "KNN": train_knn,
    "Decision Tree": train_decision_tree,
    "Random Forest": train_random_forest,
    "Gradient Boosting": train_gradient_boosting,
    "XGBoost": train_xgboost,
    "SVM": train_svm,
}
