"""
config.py — RetainIQ central configuration
All paths, model params, and constants live here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"

RAW_CSV       = DATA_DIR / "ecommerce_churn_dataset.csv"
BEST_MODEL    = MODELS_DIR / "best_model.joblib"
PREPROCESSOR  = MODELS_DIR / "preprocessor.joblib"

# ── Target ────────────────────────────────────────────────────────────────────
TARGET_COL    = "churned"
POSITIVE_LABEL = "Yes"          # Which value means "churned"

# ── Feature lists ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "days_since_last_order",
    "total_orders",
    "avg_order_value",
    "total_spend",
    "tenure_days",
    "returns_count",
    "support_tickets",
    "discount_usage_pct",
    "email_open_rate_pct",
]

BINARY_FEATURES = [
    "has_wishlist",
    "has_reviews",
    "push_notif_opt_in",
]

CATEGORICAL_FEATURES = [
    "primary_category",
    "device_type",
    "loyalty_tier",
]

DROP_COLS = ["customer_id"]

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
CV_FOLDS      = 5

# ── Model hyperparameters ─────────────────────────────────────────────────────
MODEL_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.08,
        "max_depth": 4,
        "subsample": 0.8,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.06,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
    },
}

# ── Dashboard colours (mirrors the HTML design tokens) ───────────────────────
COLORS = {
    "accent":  "#2563EB",
    "danger":  "#DC2626",
    "warning": "#D97706",
    "success": "#16A34A",
    "teal":    "#0D9488",
    "muted":   "#94A3B8",
    "border":  "#E2E8F0",
    "bg":      "#F8FAFC",
    "text":    "#0F172A",
}
