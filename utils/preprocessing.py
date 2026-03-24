"""
utils/preprocessing.py — Feature engineering and preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from config import (
    RAW_CSV, PREPROCESSOR,
    NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
    DROP_COLS, TARGET_COL, POSITIVE_LABEL,
    RANDOM_STATE,
)


# ── Loyalty tier ordering for ordinal encoding ────────────────────────────────
LOYALTY_ORDER = ["No Loyalty", "Silver", "Gold", "Platinum"]


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw CSV, return a DataFrame."""
    path = path or RAW_CSV
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features on top of the raw columns.
    All new columns are prefixed with `fe_`.
    """
    df = df.copy()

    # Recency / frequency ratio
    df["fe_orders_per_tenure"] = df["total_orders"] / (df["tenure_days"].clip(lower=1))

    # Spend efficiency
    df["fe_spend_per_order"] = df["total_spend"] / (df["total_orders"].clip(lower=1))

    # Return rate
    df["fe_return_rate"] = df["returns_count"] / (df["total_orders"].clip(lower=1))

    # Friction score: returns + support tickets normalised by orders
    df["fe_friction_score"] = (
        (df["returns_count"] + df["support_tickets"]) / df["total_orders"].clip(lower=1)
    )

    # Engagement composite: email open + push opt-in + wishlist + reviews (0–4)
    df["fe_engagement_score"] = (
        (df["email_open_rate_pct"] / 100)
        + df["push_notif_opt_in"]
        + df["has_wishlist"]
        + df["has_reviews"]
    )

    # High discount dependency flag
    df["fe_discount_heavy"] = (df["discount_usage_pct"] > 50).astype(int)

    # Dormancy flag: hasn't ordered in 90+ days
    df["fe_dormant"] = (df["days_since_last_order"] >= 90).astype(int)

    return df


def encode_target(series: pd.Series) -> pd.Series:
    """Convert 'Yes'/'No' target to 1/0."""
    return (series == POSITIVE_LABEL).astype(int)


def build_preprocessor(engineered_df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles numeric, binary, and categorical cols.
    Automatically detects engineered fe_ columns as numeric.
    """
    fe_cols = [c for c in engineered_df.columns if c.startswith("fe_")]
    num_cols = NUMERIC_FEATURES + fe_cols

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    # Categorical: loyalty_tier is ordinal, others nominal → OHE via get_dummies later
    # For pipeline simplicity we use OrdinalEncoder for all categoricals
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                LOYALTY_ORDER if col == "loyalty_tier"
                else "auto"
                for col in CATEGORICAL_FEATURES
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("bin", binary_pipe, BINARY_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, engineered_df: pd.DataFrame) -> list:
    """Return flat list of feature names after transformation."""
    fe_cols = [c for c in engineered_df.columns if c.startswith("fe_")]
    return NUMERIC_FEATURES + fe_cols + BINARY_FEATURES + CATEGORICAL_FEATURES


def prepare_data(path=None):
    """
    Full preprocessing pipeline:
    1. Load → 2. Engineer → 3. Encode target → 4. Build & fit preprocessor
    Returns X_raw (df), y, preprocessor, feature_names
    """
    df = load_raw(path)
    df = engineer_features(df)

    y = encode_target(df[TARGET_COL])
    feature_cols = (
        NUMERIC_FEATURES
        + [c for c in df.columns if c.startswith("fe_")]
        + BINARY_FEATURES
        + CATEGORICAL_FEATURES
    )
    X = df[feature_cols]

    return X, y


def save_preprocessor(preprocessor, path=None):
    path = path or PREPROCESSOR
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path=None):
    path = path or PREPROCESSOR
    return joblib.load(path)
