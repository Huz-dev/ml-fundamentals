"""
data_preprocessing.py

Loads, inspects, cleans, selects features, and scales the CC_GENERAL
credit card customer dataset for clustering.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

# Identifier column -- must never influence distance calculations
ID_COLUMN = "CUST_ID"

# The 8 features selected for clustering, covering spending, credit risk,
# and repayment behavior -- chosen for interpretability over using all 17.
SELECTED_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "TENURE",
]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> dict:
    """Summary used for README / observations.md."""
    return {
        "shape": df.shape,
        "null_counts": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the identifier column (irrelevant to distance calculations),
    then impute missing values:
      - CREDIT_LIMIT: 1 missing value -> median (robust to outliers)
      - MINIMUM_PAYMENTS: 313 missing values -> median
    Median chosen over mean because these financial columns are
    right-skewed (a few very high spenders), and median is more robust
    to that skew than mean.
    """
    df = df.copy()
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    for col in ["CREDIT_LIMIT", "MINIMUM_PAYMENTS"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    return df


def select_features(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """Select the meaningful subset of numerical features for clustering."""
    features = features or SELECTED_FEATURES
    return df[features].copy()


def scale_features(df: pd.DataFrame):
    """
    Scale all selected features with StandardScaler.
    Critical for KMeans/DBSCAN since both rely on Euclidean distance --
    unscaled, a feature like PURCHASES (range: 0-49,000) would completely
    dominate a feature like PURCHASES_FREQUENCY (range: 0-1).
    Returns (scaled_df, fitted_scaler).
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df, scaler


def preprocess_pipeline(csv_path: str, features: list = None):
    """
    Full pipeline: load -> inspect -> clean -> select features -> scale.
    Returns a dict with raw selected features, scaled features, and metadata.
    """
    df_raw = load_data(csv_path)
    summary = inspect_data(df_raw)

    df_clean = clean_data(df_raw)
    df_selected = select_features(df_clean, features)
    df_scaled, scaler = scale_features(df_selected)

    return {
        "summary": summary,
        "df_clean_full": df_clean,       # all cleaned columns, for later profiling
        "df_selected": df_selected,       # only the clustering features, unscaled
        "df_scaled": df_scaled,           # only the clustering features, scaled
        "scaler": scaler,
        "features_used": list(df_selected.columns),
    }


if __name__ == "__main__":
    result = preprocess_pipeline("data/CC_GENERAL.csv")
    print("Summary:", result["summary"])
    print("Selected features:", result["features_used"])
    print("Selected (unscaled) head:")
    print(result["df_selected"].head())
    print("\nScaled head:")
    print(result["df_scaled"].head())
