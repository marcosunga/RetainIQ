"""
predict.py — RetainIQ churn prediction on new data
────────────────────────────────────────────────────
Usage:
    python predict.py --input data/new_customers.csv
    python predict.py --input data/new_customers.csv --output data/predictions.csv
    python predict.py --input data/new_customers.csv --threshold 0.35
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
from config import MODELS_DIR, RANDOM_STATE
from utils.preprocessing import engineer_features, load_raw
from utils.evaluation import compute_metrics

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def load_best_model():
    path = MODELS_DIR / "best_model.joblib"
    if not path.exists():
        console.print("[red]No model found. Run `python train.py` first.[/]")
        sys.exit(1)
    return joblib.load(path)


@click.command()
@click.option("--input",     "-i", "input_path",  required=True,  help="Path to input CSV")
@click.option("--output",    "-o", "output_path", default=None,   help="Path to output CSV")
@click.option("--threshold", "-t", default=0.5,   help="Churn probability threshold (default: 0.5)")
@click.option("--top-n",     default=20,          help="Show top-N highest risk customers")
def main(input_path, output_path, threshold, top_n):
    w = console.width or 80
    console.print()
    console.print("━" * w, style="#2563EB")
    console.print(f"  [bold #2563EB]RetainIQ[/]  [#64748B]Churn Prediction  ·  input={input_path}  threshold={threshold}[/]")
    console.print("━" * w, style="#2563EB")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    df_eng = engineer_features(df)
    console.print(f"  Loaded {len(df):,} customers from {input_path}")

    # ── Predict ───────────────────────────────────────────────────────────────
    pipe  = load_best_model()
    from config import NUMERIC_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES

    fe_cols       = [c for c in df_eng.columns if c.startswith("fe_")]
    all_feat_cols = NUMERIC_FEATURES + fe_cols + BINARY_FEATURES + CATEGORICAL_FEATURES

    X = df_eng[[c for c in all_feat_cols if c in df_eng.columns]]
    y_prob = pipe.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── Attach results ────────────────────────────────────────────────────────
    results = df.copy()
    results["churn_probability"] = np.round(y_prob, 4)
    results["churn_predicted"]   = y_pred.astype(bool)
    results["risk_tier"] = pd.cut(
        y_prob,
        bins=[0, 0.2, 0.4, 0.6, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print("\n[bold]── Prediction Summary ───────────────────────────────[/]")
    summary = results.groupby("risk_tier", observed=False).agg(
        count=("churn_probability", "count"),
        avg_prob=("churn_probability", "mean"),
    ).reset_index()

    t = Table(box=box.ROUNDED, border_style="#E2E8F0", header_style="bold #2563EB")
    t.add_column("Risk Tier")
    t.add_column("Customers", justify="right")
    t.add_column("Avg Prob",  justify="right")
    tier_styles = {"Critical": "bold red", "High": "yellow", "Medium": "cyan", "Low": "green"}
    for _, row in summary.iterrows():
        style = tier_styles.get(str(row["risk_tier"]), "")
        t.add_row(
            f"[{style}]{row['risk_tier']}[/]",
            str(int(row["count"])),
            f"{row['avg_prob']:.3f}",
        )
    console.print(t)

    console.print(f"\n  Predicted churn: [bold red]{y_pred.sum():,}[/]  "
                  f"({y_pred.mean()*100:.1f}% of total)")

    # ── Top-N at risk ─────────────────────────────────────────────────────────
    top = results.sort_values("churn_probability", ascending=False).head(top_n)
    console.print(f"\n[bold]── Top {top_n} Highest-Risk Customers ──────────────────[/]")
    at_risk = Table(box=box.SIMPLE, header_style="bold #2563EB")
    display_cols = ["customer_id", "churn_probability", "risk_tier", "loyalty_tier",
                    "days_since_last_order", "total_spend"]
    for c in display_cols:
        if c in top.columns:
            at_risk.add_column(c)
    for _, row in top[display_cols].iterrows():
        at_risk.add_row(*[str(v) for v in row.values])
    console.print(at_risk)

    # ── Save output ───────────────────────────────────────────────────────────
    output_path = output_path or str(Path(input_path).with_suffix("_predictions.csv"))
    results.to_csv(output_path, index=False)
    console.print(f"\n[bold #16A34A]✓ Predictions saved to:[/] {output_path}\n")


if __name__ == "__main__":
    main()