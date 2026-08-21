import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RANDOM_SEED = 42

# Columns that are constant or pure identifiers -> carry zero predictive signal
COLUMNS_TO_DROP = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

TARGET_COLUMN = "Attrition"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    df = pd.read_csv(path)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Return a dictionary summary of the dataset:
    shape, dtypes, null counts, duplicate count, target distribution.
    This is what we report in observations.md / README.
    """
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "target_distribution_pct": df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
    }
    return summary


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop constant / identifier columns that carry no predictive signal.
    No missing values exist in this dataset (verified during inspection),
    so no imputation strategy is needed here -- documented explicitly
    rather than silently skipped.
    """
    df = df.copy()
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target Attrition: Yes -> 1, No -> 0."""
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return df


def get_feature_types(df: pd.DataFrame) -> tuple[list, list]:
    """
    Split feature columns (excluding target) into categorical and numerical.
    Returns (categorical_cols, numerical_cols).
    """
    features = df.drop(columns=[TARGET_COLUMN])
    categorical_cols = features.select_dtypes(include=["object", "str"]).columns.tolist()
    numerical_cols = features.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numerical_cols


def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    encoding_info = {"method": "one-hot", "columns_encoded": categorical_cols}
    return df_encoded, encoding_info


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_SEED,
):
    """
    Split into train / validation / test sets.
    Stratified on target because Attrition is imbalanced (~16% Yes).

    First split off the test set, then split remaining into train/val,
    so proportions stay exactly as requested.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # val_size is a fraction of the ORIGINAL data; recompute relative to X_temp
    relative_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, numerical_cols: list):
    """
    Fit StandardScaler on numerical columns using TRAINING data only,
    then apply the same fitted transform to val/test.
    This avoids data leakage (never fit on val/test).

    Returns scaled copies of X_train, X_val, X_test, and the fitted scaler.
    """
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # only scale numerical cols that are still present after encoding
    cols_present = [c for c in numerical_cols if c in X_train.columns]

    X_train_scaled[cols_present] = scaler.fit_transform(X_train[cols_present])
    X_val_scaled[cols_present] = scaler.transform(X_val[cols_present])
    X_test_scaled[cols_present] = scaler.transform(X_test[cols_present])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_pipeline(csv_path: str):
    """
    Full pipeline: load -> inspect -> clean -> encode target ->
    encode categorical -> split -> scale.

    Returns a dict containing every artifact needed by downstream modules:
    train/val/test splits (scaled + unscaled), summary info, column lists.
    """
    df_raw = load_data(csv_path)
    summary = inspect_data(df_raw)

    df = clean_data(df_raw)
    df = encode_target(df)

    categorical_cols, numerical_cols = get_feature_types(df)
    df_encoded, encoding_info = encode_categorical(df, categorical_cols)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test, numerical_cols
    )

    return {
        "summary": summary,
        "encoding_info": encoding_info,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
    }


if __name__ == "__main__":
    # quick manual check when running this file directly
    result = preprocess_pipeline("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    print("Summary:", result["summary"])
    print("Train shape:", result["X_train"].shape)
    print("Val shape:", result["X_val"].shape)
    print("Test shape:", result["X_test"].shape)
    print("Categorical cols:", result["categorical_cols"])
    print("Numerical cols:", result["numerical_cols"])
