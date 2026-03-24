"""
evaluate.py — RetainIQ deep-dive model evaluation
──────────────────────────────────────────────────
Usage:
    python evaluate.py                          # Evaluate best model
    python evaluate.py --model random_forest    # Evaluate a specific model
    python evaluate.py --threshold 0.35         # Custom decision threshold
    python evaluate.py --segment loyalty_tier   # Segment-level evaluation

Outputs:
    reports/plots/  — extended evaluation charts
    reports/segment_report.csv
"""

import sys
import warnings
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import click

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODELS_DIR, REPORTS_DIR, RANDOM_STATE,
    NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
    TARGET_COL, POSITIVE_LABEL,
)
from utils.preprocessing import (
    load_raw, engineer_features, encode_target, build_preprocessor, get_feature_names,
)
from utils.evaluation import (
    compute_metrics, print_classification_report,
    plot_confusion_matrix, plot_feature_importance,
)
from utils.visualisation import (
    plot_churn_overview, plot_numeric_distributions,
    plot_correlation_heatmap, plot_churn_by_category,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib" if name != "best" else MODELS_DIR / "best_model.joblib"
    if not path.exists():
        console.print(f"[red]Model not found: {path}[/]  Run `python train.py` first.")
        sys.exit(1)
    return joblib.load(path)


def threshold_analysis(y_true, y_prob, model_name: str):
    """Evaluate metrics across a range of decision thresholds."""
    thresholds = np.linspace(0.1, 0.9, 81)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        rows.append({"threshold": t, "precision": prec, "recall": recall, "f1": f1})
    df = pd.DataFrame(rows)

    # Plot
    from config import COLORS
    C = COLORS
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["threshold"], df["precision"], color=C["accent"],  linewidth=2, label="Precision")
    ax.plot(df["threshold"], df["recall"],    color=C["danger"],  linewidth=2, label="Recall")
    ax.plot(df["threshold"], df["f1"],        color=C["success"], linewidth=2, label="F1")
    ax.axvline(df.loc[df["f1"].idxmax(), "threshold"], color=C["muted"], linestyle="--",
               linewidth=1.2, label=f"Best F1 threshold ({df.loc[df['f1'].idxmax(), 'threshold']:.2f})")
    ax.set_xlabel("Decision Threshold", fontsize=9, color=C["muted"])
    ax.set_ylabel("Score", fontsize=9, color=C["muted"])
    ax.set_title(f"Threshold Analysis — {model_name}", fontsize=11, fontweight="600", color=C["text"])
    ax.legend(fontsize=8, framealpha=0.85)
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0, 1)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"threshold_analysis_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def segment_evaluation(df_raw, y_true, y_prob, segment_col: str, model_name: str):
    """Per-segment AUC and churn rate breakdown."""
    df = df_raw.copy()
    df["y_true"] = y_true.values
    df["y_prob"] = y_prob
    df["y_pred"] = (y_prob >= 0.5).astype(int)

    rows = []
    for seg_val, grp in df.groupby(segment_col):
        if len(grp) < 10:
            continue
        auc = roc_auc_score(grp["y_true"], grp["y_prob"]) if grp["y_true"].nunique() > 1 else None
        churn_rate = grp["y_true"].mean() * 100
        n_churned  = grp["y_true"].sum()
        rows.append({
            "segment":    str(seg_val),
            "n":          len(grp),
            "churn_rate": round(churn_rate, 2),
            "n_churned":  n_churned,
            "auc":        round(auc, 4) if auc else "N/A",
        })

    seg_df = pd.DataFrame(rows).sort_values("churn_rate", ascending=False)

    table = Table(title=f"Segment Evaluation — {segment_col}", box=box.ROUNDED,
                  border_style="#E2E8F0", header_style="bold #2563EB")
    table.add_column("Segment")
    table.add_column("N",           justify="right")
    table.add_column("Churn Rate",  justify="right")
    table.add_column("Churned",     justify="right")
    table.add_column("Seg. AUC",    justify="right")

    for _, row in seg_df.iterrows():
        rate_str = f"{row['churn_rate']:.1f}%"
        style = "bold red" if row["churn_rate"] > 8 else ("yellow" if row["churn_rate"] > 4 else "green")
        table.add_row(
            row["segment"], str(row["n"]),
            f"[{style}]{rate_str}[/]",
            str(row["n_churned"]),
            str(row["auc"]),
        )
    console.print(table)

    # Save CSV
    seg_df.to_csv(REPORTS_DIR / f"segment_report_{segment_col}.csv", index=False)
    return seg_df


@click.command()
@click.option("--model",     default="best",  help="Model name or 'best'")
@click.option("--threshold", default=0.5,     help="Decision threshold (default: 0.5)")
@click.option("--segment",   default=None,    help="Segment column for breakdown (e.g. loyalty_tier)")
@click.option("--eda",       is_flag=True,    help="Run exploratory data analysis plots")
def main(model, threshold, segment, eda):
    w = console.width or 80
    console.print()
    console.print("━" * w, style="#2563EB")
    console.print(f"  [bold #2563EB]RetainIQ[/]  [#64748B]Model Evaluation  ·  model={model}  threshold={threshold}[/]")
    console.print("━" * w, style="#2563EB")

    # ── Load data & model ─────────────────────────────────────────────────────
    raw_df = load_raw()
    df     = engineer_features(raw_df)
    y      = encode_target(df["churned"])

    fe_cols       = [c for c in df.columns if c.startswith("fe_")]
    all_feat_cols = NUMERIC_FEATURES + fe_cols + BINARY_FEATURES + CATEGORICAL_FEATURES
    X = df[all_feat_cols]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )
    # Match test index to raw_df for segment analysis
    test_raw = raw_df.iloc[X_test.index]

    pipe   = load_model(model)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    console.print("\n[bold]── Core Metrics ─────────────────────────────────────[/]")
    metrics = compute_metrics(y_test, y_pred, y_prob)
    m_table = Table(box=box.SIMPLE, header_style="bold #2563EB")
    m_table.add_column("Metric")
    m_table.add_column("Value", justify="right")
    for k, v in metrics.items():
        m_table.add_row(k.upper().replace("_", " "), str(v))
    console.print(m_table)

    print_classification_report(y_test, y_pred, model_name=model)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred, model, save_path=PLOTS_DIR / f"cm_eval_{model}.png")

    # ── Threshold analysis ────────────────────────────────────────────────────
    console.print("\n[bold]── Threshold Analysis ───────────────────────────────[/]")
    thresh_df = threshold_analysis(y_test.values, y_prob, model)
    best_t = thresh_df.loc[thresh_df["f1"].idxmax(), "threshold"]
    console.print(f"  Optimal F1 threshold: [bold #16A34A]{best_t:.2f}[/]  "
                  f"(F1 = {thresh_df['f1'].max():.4f})")

    # ── Feature importance ────────────────────────────────────────────────────
    clf = pipe.named_steps.get("clf")
    if clf and hasattr(clf, "feature_importances_"):
        preprocessor = build_preprocessor(df[all_feat_cols])
        feature_names = get_feature_names(preprocessor, df)
        imps = clf.feature_importances_
        plot_feature_importance(
            feature_names, imps, model,
            save_path=PLOTS_DIR / f"fi_eval_{model}.png",
        )

    # ── Segment evaluation ────────────────────────────────────────────────────
    seg_cols = [segment] if segment else ["loyalty_tier", "primary_category"]
    for scol in seg_cols:
        if scol in test_raw.columns:
            console.print(f"\n[bold]── Segment: {scol} ──────────────────────────────[/]")
            segment_evaluation(test_raw, y_test, y_prob, scol, model)

    # ── EDA plots ─────────────────────────────────────────────────────────────
    if eda:
        console.print("\n[bold]── EDA Plots ─────────────────────────────────────[/]")
        plot_churn_overview(raw_df, save_path=PLOTS_DIR / "eda_churn_overview.png")
        plot_numeric_distributions(
            raw_df, NUMERIC_FEATURES[:6],
            save_path=PLOTS_DIR / "eda_distributions.png",
        )
        plot_correlation_heatmap(
            raw_df, NUMERIC_FEATURES,
            save_path=PLOTS_DIR / "eda_correlation.png",
        )
        for cat_col in ["primary_category", "loyalty_tier", "device_type"]:
            plot_churn_by_category(
                raw_df, cat_col,
                save_path=PLOTS_DIR / f"eda_{cat_col}.png",
            )
        console.print(f"  EDA plots saved to {PLOTS_DIR}")

    console.print(f"\n[bold #16A34A]✓ Evaluation complete.[/]  Plots → {PLOTS_DIR}\n")


if __name__ == "__main__":
    main()