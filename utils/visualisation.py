"""
utils/visualisation.py — Segment-level and EDA visualisation helpers
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from config import COLORS

C = COLORS
PLT_STYLE = {
    "axes.facecolor":   C["bg"],
    "figure.facecolor": C["bg"],
    "axes.edgecolor":   C["border"],
    "axes.labelcolor":  C["text"],
    "xtick.color":      C["muted"],
    "ytick.color":      C["muted"],
    "axes.grid":        True,
    "grid.color":       C["border"],
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
}


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="600", color=C["text"], pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=C["muted"])
    ax.set_ylabel(ylabel, fontsize=9, color=C["muted"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(C["border"])


def plot_churn_overview(df: pd.DataFrame, save_path=None):
    """Pie/donut of churn vs retained with summary stats."""
    plt.rcParams.update(PLT_STYLE)
    churned  = (df["churned"] == "Yes").sum()
    retained = (df["churned"] == "No").sum()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Donut
    axes[0].pie(
        [churned, retained],
        labels=["Churned", "Retained"],
        colors=[C["danger"], C["success"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": C["bg"], "linewidth": 2},
        textprops={"color": C["text"], "fontsize": 9},
    )
    axes[0].set_title("Churn Split", fontsize=11, fontweight="600", color=C["text"])

    # Bar by loyalty tier
    tier_order = ["No Loyalty", "Silver", "Gold", "Platinum"]
    tier_churn = (
        df.groupby("loyalty_tier")["churned"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reindex(tier_order)
    )
    bar_cols = [C["danger"], C["warning"], C["accent"], C["success"]]
    bars = axes[1].bar(tier_churn.index, tier_churn.values, color=bar_cols, width=0.55)
    for bar, val in zip(bars, tier_churn.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=C["text"],
        )
    _ax_style(axes[1], "Churn Rate by Loyalty Tier", "Tier", "Churn Rate (%)")

    fig.suptitle("RetainIQ — Churn Overview", fontsize=13, fontweight="700", color=C["text"], y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_numeric_distributions(df: pd.DataFrame, features: list, save_path=None):
    """KDE plots for numeric features split by churn label."""
    plt.rcParams.update(PLT_STYLE)
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for label, colour in [("No", C["accent"]), ("Yes", C["danger"])]:
            subset = df[df["churned"] == label][feat].dropna()
            subset.plot.kde(ax=ax, color=colour, linewidth=1.8, label=label)
            ax.fill_between(
                *[subset.sort_values().values, np.zeros(len(subset))],
                alpha=0.08, color=colour,
            )
        _ax_style(ax, title=feat.replace("_", " ").title(), xlabel="")
        ax.legend(["Retained", "Churned"], fontsize=7, framealpha=0.8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions — Churned vs Retained", fontsize=13, fontweight="700",
                 color=C["text"], y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list, save_path=None):
    plt.rcParams.update(PLT_STYLE)
    corr_df = df[numeric_cols + ["churned"]].copy()
    corr_df["churned_bin"] = (corr_df["churned"] == "Yes").astype(int)
    corr = corr_df.drop(columns=["churned"]).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.5, linecolor=C["border"],
        ax=ax, cbar_kws={"shrink": 0.8},
        vmin=-1, vmax=1,
    )
    _ax_style(ax, title="Feature Correlation Matrix")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_churn_by_category(df: pd.DataFrame, col: str, save_path=None):
    plt.rcParams.update(PLT_STYLE)
    churn_rates = (
        df.groupby(col)["churned"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    palette = [C["danger"], C["warning"], C["accent"], C["teal"], C["success"]]
    colours  = [palette[i % len(palette)] for i in range(len(churn_rates))]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(churn_rates.index, churn_rates.values, color=colours, width=0.55)
    for bar, val in zip(bars, churn_rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color=C["text"])

    _ax_style(ax, f"Churn Rate by {col.replace('_', ' ').title()}", col, "Churn Rate (%)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
