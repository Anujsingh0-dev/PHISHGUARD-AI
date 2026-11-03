"""
scripts/inspect_dataset.py
───────────────────────────
Dataset inspection script for PhiUSIIL.

Run BEFORE any preprocessing or training.
This script gives you a complete picture of what the dataset looks like.

Run with:
  python scripts/inspect_dataset.py

Outputs saved to:
  reports/figures/eda_*.png
  reports/metrics/dataset_summary.json
"""

import sys
import pathlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_loader import load_or_download_dataset
from src.utils import FIGURES_DIR, METRICS_DIR, ensure_dirs, print_section

COLORS = {"phishing": "#e74c3c", "legitimate": "#2ecc71", "primary": "#3498db"}


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Print a comprehensive inspection of the dataset and return a summary dict."""
    summary = {}

    # ── 1. Basic shape ─────────────────────────────────────────────────────────
    print_section("1. Dataset Shape")
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    summary["rows"]    = df.shape[0]
    summary["columns"] = df.shape[1]

    # ── 2. Column names ────────────────────────────────────────────────────────
    print_section("2. Column Names")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>3}. {col}")
    summary["column_names"] = list(df.columns)

    # ── 3. Data types ──────────────────────────────────────────────────────────
    print_section("3. Data Types")
    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_str    = {str(k): int(v) for k, v in dtype_counts.items()}
    summary["dtype_counts"] = dtype_str
    print(df.dtypes.to_string())

    # ── 4. Target distribution ─────────────────────────────────────────────────
    print_section("4. Target Distribution (label column)")
    if "label" in df.columns:
        vc = df["label"].value_counts()
        phishing_count   = int((df["label"] == 0).sum())
        legitimate_count = int((df["label"] == 1).sum())
        total            = len(df)
        print(f"  Label 0 (Phishing)   : {phishing_count:>8,}  ({phishing_count/total*100:.2f}%)")
        print(f"  Label 1 (Legitimate) : {legitimate_count:>8,}  ({legitimate_count/total*100:.2f}%)")
        summary["phishing_count"]   = phishing_count
        summary["legitimate_count"] = legitimate_count
    else:
        print("  [ERROR] 'label' column not found!")

    # ── 5. Missing values ──────────────────────────────────────────────────────
    print_section("5. Missing Values")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if missing_cols.empty:
        print("  No missing values found — dataset is complete.")
        summary["missing_values"] = 0
    else:
        print(missing_cols.to_string())
        summary["missing_values"] = int(missing.sum())

    # ── 6. Duplicate rows ──────────────────────────────────────────────────────
    print_section("6. Duplicate Rows")
    n_dup = df.duplicated().sum()
    print(f"  Duplicate rows: {n_dup:,}")
    summary["duplicate_rows"] = int(n_dup)

    # ── 7. Numerical columns summary ───────────────────────────────────────────
    print_section("7. Numerical Column Summary (first 10 stats)")
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"  Numerical columns: {len(numeric_df.columns)}")
    print(numeric_df.describe().round(3).to_string())
    summary["numerical_columns"] = list(numeric_df.columns)

    # ── 8. Categorical / text columns ──────────────────────────────────────────
    print_section("8. Non-Numeric (Text) Columns")
    cat_df = df.select_dtypes(exclude=[np.number])
    print(f"  Non-numeric columns: {list(cat_df.columns)}")
    summary["categorical_columns"] = list(cat_df.columns)

    for col in cat_df.columns:
        n_unique = df[col].nunique()
        sample   = df[col].dropna().head(3).tolist()
        print(f"  {col:<15} unique={n_unique:>6}  sample={sample}")

    return summary


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Save a bar chart of phishing vs legitimate counts."""
    counts = df["label"].value_counts().sort_index()
    labels_text = ["Phishing (0)", "Legitimate (1)"]
    colors = [COLORS["phishing"], COLORS["legitimate"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axes[0].bar(labels_text, counts.values, color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_title("Class Distribution — PhiUSIIL Dataset", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Records", fontsize=11)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Pie chart
    axes[1].pie(
        counts.values,
        labels     = labels_text,
        colors     = colors,
        autopct    = "%1.1f%%",
        startangle = 90,
        textprops  = {"fontsize": 11},
    )
    axes[1].set_title("Class Proportion", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = FIGURES_DIR / "eda_class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Class distribution saved → {out}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Save a correlation heatmap (top 30 numeric features by variance)."""
    numeric_df = df.select_dtypes(include=[np.number])

    # Select top 30 features by variance to keep the heatmap readable
    variances = numeric_df.var().sort_values(ascending=False)
    top_cols  = variances.head(30).index.tolist()
    sub_df    = numeric_df[top_cols]

    corr = sub_df.corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr,
        ax       = ax,
        cmap     = "coolwarm",
        center   = 0,
        annot    = False,
        linewidths = 0.3,
        vmin     = -1,
        vmax     = 1,
    )
    ax.set_title("Feature Correlation Heatmap (Top 30 by Variance)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = FIGURES_DIR / "eda_correlation_heatmap.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Correlation heatmap saved → {out}")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Save distribution plots for top numeric features split by class."""
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["label"], errors="ignore")

    # Pick top 12 features by variance
    top_cols = numeric_df.var().sort_values(ascending=False).head(12).index.tolist()

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    phish  = df[df["label"] == 0]
    legit  = df[df["label"] == 1]

    for i, col in enumerate(top_cols):
        ax = axes[i]
        ax.hist(phish[col].dropna(),  bins=40, alpha=0.6, color=COLORS["phishing"],   label="Phishing",   density=True)
        ax.hist(legit[col].dropna(),  bins=40, alpha=0.6, color=COLORS["legitimate"], label="Legitimate", density=True)
        ax.set_title(col, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylabel("Density", fontsize=8)

    # Hide unused subplots
    for j in range(len(top_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions: Phishing vs Legitimate", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = FIGURES_DIR / "eda_feature_distributions.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature distributions saved → {out}")


if __name__ == "__main__":
    ensure_dirs()

    print("=" * 60)
    print("  PHISHGUARD AI — Dataset Inspection")
    print("=" * 60)

    df = load_or_download_dataset()

    summary = inspect_dataset(df)

    # Save summary JSON
    out_json = METRICS_DIR / "dataset_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\n[INFO] Summary saved → {out_json}")

    # Generate visualisations
    print_section("Generating Visualisations")
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)

    print("\n[DONE] Dataset inspection complete.")
    print(f"  Figures saved to: {FIGURES_DIR}")
