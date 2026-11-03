"""
src/evaluate_models.py
───────────────────────
Standalone evaluation and report generation script.

Run this after training both models to:
  - Generate all figures
  - Print evaluation summaries
  - Verify saved metrics

Run with:
  python src/evaluate_models.py
"""

import sys
import pathlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.utils import (
    FULL_MODEL_DIR, URL_MODEL_DIR, METRICS_DIR, FIGURES_DIR,
    ensure_dirs, print_section, load_metrics,
)


def print_metrics_report(metrics_file: str, title: str) -> None:
    """Load and print a saved metrics JSON file."""
    print_section(title)
    try:
        data = load_metrics(metrics_file)
        print(f"Generated at : {data.get('generated_at', 'N/A')}")
        print(f"Model name   : {data.get('model_name', 'N/A')}")
        print()

        all_models = data.get("metrics", {}).get("all_models", [])
        if not all_models:
            print("[WARNING] No model results found in metrics file.")
            return

        header = f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
        print(header)
        print("─" * len(header))

        for r in all_models:
            auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "  N/A  "
            print(
                f"{r['model_name']:<25} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
                f"{r['recall']:>8.4f} {r['f1_score']:>8.4f} {auc:>9}"
            )

        best = data.get("metrics", {}).get("best_model", "N/A")
        print(f"\n  Best model: {best}")

    except FileNotFoundError as e:
        print(f"[WARNING] {e}")


def verify_saved_artifacts() -> None:
    """Check that all expected model files exist."""
    print_section("Artifact Verification")

    artifacts = {
        "Full model"       : FULL_MODEL_DIR / "best_model.pkl",
        "Full preprocessor": FULL_MODEL_DIR / "preprocessor.pkl",
        "Feature names"    : FULL_MODEL_DIR / "feature_names.pkl",
        "URL model"        : URL_MODEL_DIR  / "url_model.pkl",
        "URL preprocessor" : URL_MODEL_DIR  / "url_preprocessor.pkl",
        "Full metrics"     : METRICS_DIR    / "full_model_metrics.json",
        "URL metrics"      : METRICS_DIR    / "url_model_metrics.json",
    }

    all_ok = True
    for name, path in artifacts.items():
        status = "✅ Found" if path.exists() else "❌ MISSING"
        print(f"  {name:<25} {status}  ({path.name})")
        if not path.exists():
            all_ok = False

    if all_ok:
        print("\n[OK] All artifacts present.")
    else:
        print("\n[WARNING] Some artifacts are missing. Run the training scripts.")


def list_generated_figures() -> None:
    """Print a list of all generated figures."""
    print_section("Generated Figures")
    figs = sorted(FIGURES_DIR.glob("*.png"))
    if figs:
        for fig in figs:
            print(f"  {fig.name}")
    else:
        print("  No figures found yet. Run training scripts first.")


if __name__ == "__main__":
    ensure_dirs()
    verify_saved_artifacts()
    print_metrics_report("full_model_metrics.json", "Full Dataset Model Results")
    print_metrics_report("url_model_metrics.json",  "URL-Only Model Results")
    list_generated_figures()
