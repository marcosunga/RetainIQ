"""
utils/evaluation.py — Model evaluation metrics and report generation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from config import COLORS, REPORTS_DIR, CV_FOLDS, RANDOM_STATE

# ── Palette ───────────────────────────────────────────────────────────────────
C = COLORS
PLT_STYLE = {
    "axes.facecolor":    C["bg"],
    "figure.facecolor":  C["bg"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["text"],
    "xtick.color":       C["muted"],
    "ytick.color":       C["muted"],
    "axes.grid":         True,
    "grid.color":        C["border"],
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Return a dict of all key classification metrics."""
    metrics = {
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc":               round(matthews_corrcoef(y_true, y_pred), 4),
    }
    if y_prob is not None:
        metrics["roc_auc"]  = round(roc_auc_score(y_true, y_prob), 4)
        metrics["avg_prec"] = round(average_precision_score(y_true, y_prob), 4)
        metrics["brier"]    = round(brier_score_loss(y_true, y_prob), 4)
        metrics["log_loss"] = round(log_loss(y_true, y_prob), 4)
    return metrics


def print_classification_report(y_true, y_pred, model_name: str = "Model"):
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    report = classification_report(y_true, y_pred, target_names=["Retained", "Churned"], output_dict=True)

    console.print(f"\n[bold #2563EB]── {model_name} — Classification Report ──[/]")
    table = Table(box=box.ROUNDED, border_style="#E2E8F0", header_style="bold #2563EB")
    table.add_column("Class",     style="#0F172A")
    table.add_column("Precision", justify="right")
    table.add_column("Recall",    justify="right")
    table.add_column("F1-Score",  justify="right")
    table.add_column("Support",   justify="right")

    for cls in ["Retained", "Churned", "macro avg", "weighted avg"]:
        r = report[cls]
        table.add_row(
            cls,
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1-score']:.3f}",
            str(int(r["support"])),
        )
    console.print(table)


def compare_models_table(results: dict):
    """Print a Rich comparison table for multiple models."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print("\n[bold #2563EB]── Model Comparison ──────────────────────────────[/]")
    table = Table(box=box.ROUNDED, border_style="#E2E8F0", header_style="bold #2563EB")

    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg Prec", "MCC", "Brier ↓", "CV AUC"]
    for c in cols:
        table.add_column(c, justify="right" if c != "Model" else "left")

    # Rank by ROC-AUC
    sorted_models = sorted(results.items(), key=lambda x: x[1].get("roc_auc", 0), reverse=True)

    for i, (name, m) in enumerate(sorted_models):
        style = "bold" if i == 0 else ""
        prefix = "🥇 " if i == 0 else ("🥈 " if i == 1 else ("🥉 " if i == 2 else "   "))
        table.add_row(
            prefix + name,
            f"{m.get('accuracy', 0):.4f}",
            f"{m.get('precision', 0):.4f}",
            f"{m.get('recall', 0):.4f}",
            f"{m.get('f1', 0):.4f}",
            f"{m.get('roc_auc', 0):.4f}",
            f"{m.get('avg_prec', 0):.4f}",
            f"{m.get('mcc', 0):.4f}",
            f"{m.get('brier', 0):.4f}",
            f"{m.get('cv_auc_mean', 0):.4f} ± {m.get('cv_auc_std', 0):.4f}",
            style=style,
        )
    console.print(table)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="600", color=C["text"], pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=C["muted"])
    ax.set_ylabel(ylabel, fontsize=9, color=C["muted"])
    ax.tick_params(colors=C["muted"], labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(C["border"])


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path=None):
    plt.rcParams.update(PLT_STYLE)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        linewidths=1, linecolor=C["border"],
        ax=ax, cbar=False,
        annot_kws={"size": 14, "weight": "bold", "color": C["text"]},
    )
    _apply_style(ax, title=f"Confusion Matrix — {model_name}", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_roc_curve(models_data: dict, save_path=None):
    """
    models_data: {name: (y_true, y_prob)}
    """
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(6, 5))

    palette = [C["accent"], C["danger"], C["success"], C["teal"], C["warning"]]
    ax.plot([0, 1], [0, 1], "--", color=C["muted"], linewidth=1.2, label="Random (AUC = 0.50)")

    for i, (name, (y_true, y_prob)) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        col = palette[i % len(palette)]
        ax.plot(fpr, tpr, linewidth=2, color=col, label=f"{name}  (AUC = {auc:.4f})")

    _apply_style(ax, title="ROC Curve Comparison", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(fontsize=8, framealpha=0.85, edgecolor=C["border"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_pr_curve(models_data: dict, save_path=None):
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(6, 5))

    palette = [C["accent"], C["danger"], C["success"], C["teal"], C["warning"]]
    for i, (name, (y_true, y_prob)) in enumerate(models_data.items()):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        col = palette[i % len(palette)]
        ax.plot(rec, prec, linewidth=2, color=col, label=f"{name}  (AP = {ap:.4f})")

    _apply_style(ax, title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=8, framealpha=0.85, edgecolor=C["border"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_feature_importance(feature_names: list, importances: np.ndarray, model_name: str, top_n=15, save_path=None):
    plt.rcParams.update(PLT_STYLE)

    idx = np.argsort(importances)[-top_n:]
    names  = [feature_names[i] for i in idx]
    values = importances[idx]

    # Colour by importance tier
    colours = []
    for v in values:
        if v >= np.percentile(values, 80):
            colours.append(C["danger"])
        elif v >= np.percentile(values, 50):
            colours.append(C["warning"])
        else:
            colours.append(C["accent"])

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    bars = ax.barh(names, values, color=colours, height=0.65)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8, color=C["muted"])

    legend_patches = [
        mpatches.Patch(color=C["danger"],  label="High importance"),
        mpatches.Patch(color=C["warning"], label="Mid importance"),
        mpatches.Patch(color=C["accent"],  label="Low importance"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, framealpha=0.85, edgecolor=C["border"])
    _apply_style(ax, title=f"Feature Importance — {model_name} (Top {top_n})", xlabel="Importance", ylabel="")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_model_comparison_bar(results: dict, metric="roc_auc", save_path=None):
    plt.rcParams.update(PLT_STYLE)
    names  = list(results.keys())
    values = [results[n].get(metric, 0) for n in names]
    colours = [C["success"] if v == max(values) else C["accent"] for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colours, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="600", color=C["text"])

    _apply_style(ax, title=f"Model Comparison — {metric.upper()}", xlabel="Model", ylabel=metric)
    ax.set_ylim(min(values) * 0.97, max(values) * 1.03)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def run_cross_validation(model, X, y, cv=None, scoring="roc_auc") -> dict:
    """Run stratified k-fold CV and return mean + std."""
    cv = cv or StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "cv_auc_mean": round(scores.mean(), 4),
        "cv_auc_std":  round(scores.std(), 4),
        "cv_scores":   scores.tolist(),
    }
