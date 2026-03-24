"""
train.py — RetainIQ model training pipeline
─────────────────────────────────────────────
Usage:
    python train.py                       # Train all models
    python train.py --model xgboost       # Train a specific model
    python train.py --no-cv               # Skip cross-validation (faster)

Outputs:
    models/best_model.joblib
    models/preprocessor.joblib
    models/<model_name>.joblib
    reports/plots/  — ROC, PR, feature importance, confusion matrix
"""

import sys
import time
import warnings
import joblib
from pathlib import Path

import numpy as np
import click

warnings.filterwarnings("ignore")

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS,
    MODEL_PARAMS, NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
)
from utils.preprocessing import (
    prepare_data, build_preprocessor, get_feature_names,
    save_preprocessor, engineer_features, load_raw, encode_target,
)
from utils.evaluation import (
    compute_metrics, print_classification_report, compare_models_table,
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
    plot_feature_importance, plot_model_comparison_bar, run_cross_validation,
)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn

console = Console()
PLOTS_DIR = REPORTS_DIR / "plots"


def get_model(name: str):
    p = MODEL_PARAMS.get(name, {})
    if name == "logistic_regression":
        return LogisticRegression(**p)
    elif name == "random_forest":
        return RandomForestClassifier(**p)
    elif name == "gradient_boosting":
        return GradientBoostingClassifier(**p)
    elif name == "xgboost":
        if not XGBOOST_AVAILABLE:
            console.print("[yellow]⚠ XGBoost not installed, skipping.[/]")
            return None
        params = {k: v for k, v in p.items() if k != "use_label_encoder"}
        return XGBClassifier(**params, verbosity=0)
    else:
        raise ValueError(f"Unknown model: {name}")


def extract_feature_importances(model, feature_names):
    """Extract feature importances from various model types."""
    clf = model.named_steps.get("clf") or model
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    elif hasattr(clf, "coef_"):
        return np.abs(clf.coef_[0])
    return None


@click.command()
@click.option("--model",   default="all",     help="Model to train: all | logistic_regression | random_forest | gradient_boosting | xgboost")
@click.option("--no-cv",   is_flag=True,      help="Skip cross-validation")
@click.option("--test-size", default=0.20,    help="Test set proportion (default: 0.20)")
def main(model, no_cv, test_size):
    w = console.width or 80
    console.print()
    console.print("━" * w, style="#2563EB")
    console.print("  [bold #2563EB]RetainIQ[/]  [#64748B]Churn Prediction Training Pipeline[/]")
    console.print("━" * w, style="#2563EB")

    # ── 1. Load & engineer data ───────────────────────────────────────────────
    console.print("\n[bold]1 / 5 — Loading & engineering features[/]")
    raw_df = load_raw()
    df     = engineer_features(raw_df)
    y      = encode_target(df["churned"])

    fe_cols       = [c for c in df.columns if c.startswith("fe_")]
    all_feat_cols = NUMERIC_FEATURES + fe_cols + BINARY_FEATURES + CATEGORICAL_FEATURES
    X = df[all_feat_cols]

    # ── 2. Split ──────────────────────────────────────────────────────────────
    console.print("\n[bold]2 / 5 — Train / test split[/]")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y,
    )
    console.print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # ── 3. Build preprocessor ─────────────────────────────────────────────────
    console.print("\n[bold]3 / 5 — Building preprocessor[/]")
    preprocessor  = build_preprocessor(df[all_feat_cols])
    feature_names = get_feature_names(preprocessor, df)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 4. Train models ───────────────────────────────────────────────────────
    console.print("\n[bold]4 / 5 — Training models[/]")
    all_models = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
    to_train   = all_models if model == "all" else [model]

    results    = {}
    pipelines  = {}
    roc_data   = {}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    with Progress(
        SpinnerColumn(style="#2563EB"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, style="#2563EB"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for mname in to_train:
            clf = get_model(mname)
            if clf is None:
                continue

            task = progress.add_task(f"  {mname:<28}", total=None)
            t0   = time.time()

            pipe = Pipeline([
                ("pre", preprocessor),
                ("clf", clf),
            ])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)

            if not no_cv:
                cv_res  = run_cross_validation(pipe, X_train, y_train, cv=cv)
                metrics.update(cv_res)

            elapsed = time.time() - t0
            metrics["train_time_s"] = round(elapsed, 2)
            results[mname]  = metrics
            pipelines[mname] = pipe
            roc_data[mname]  = (y_test.values, y_prob)

            # Save model
            joblib.dump(pipe, MODELS_DIR / f"{mname}.joblib")
            progress.update(task, description=f"  [green]✓[/] {mname:<25}  AUC={metrics.get('roc_auc',0):.4f}")
            progress.stop_task(task)

    # ── 5. Report & save best model ───────────────────────────────────────────
    console.print("\n[bold]5 / 5 — Evaluation & reporting[/]")

    compare_models_table(results)

    best_name = max(results, key=lambda n: results[n].get("roc_auc", 0))
    console.print(f"\n  [bold #16A34A]Best model:[/] [bold]{best_name}[/]  "
                  f"(ROC-AUC = {results[best_name]['roc_auc']:.4f})")

    best_pipe = pipelines[best_name]
    joblib.dump(best_pipe, MODELS_DIR / "best_model.joblib")
    save_preprocessor(preprocessor)

    # Per-model classification reports
    for mname, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        print_classification_report(y_test, y_pred, mname)

    # Plots
    console.print("\n  Generating plots…")

    # Confusion matrix for best model
    plot_confusion_matrix(
        y_test, best_pipe.predict(X_test), best_name,
        save_path=PLOTS_DIR / f"confusion_matrix_{best_name}.png",
    )

    # ROC curve for all models
    plot_roc_curve(roc_data, save_path=PLOTS_DIR / "roc_curves.png")

    # PR curve
    plot_pr_curve(roc_data, save_path=PLOTS_DIR / "pr_curves.png")

    # Model comparison bar
    plot_model_comparison_bar(results, metric="roc_auc", save_path=PLOTS_DIR / "model_comparison.png")

    # Feature importance for best model
    imps = extract_feature_importances(best_pipe, feature_names)
    if imps is not None:
        plot_feature_importance(
            feature_names, imps, best_name,
            save_path=PLOTS_DIR / f"feature_importance_{best_name}.png",
        )

    console.print(f"\n  [bold]Plots saved to:[/] {PLOTS_DIR}")
    console.print(f"  [bold]Models saved to:[/] {MODELS_DIR}")
    console.print("\n[bold #16A34A]✓ Training complete.[/]\n")

    # Save results summary
    import json
    summary_path = REPORTS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {k: {m: v for m, v in metrics.items() if m != "cv_scores"}
             for k, metrics in results.items()},
            f, indent=2,
        )
    console.print(f"  Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()