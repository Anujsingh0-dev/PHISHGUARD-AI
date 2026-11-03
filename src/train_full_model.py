"""
src/train_full_model.py
────────────────────────
Trains and evaluates ML models on the full PhiUSIIL dataset (MODE A).

What this script does:
  1. Loads preprocessed training and test data
  2. Trains three classifiers:
       - Logistic Regression
       - Decision Tree
       - Random Forest
  3. Evaluates each model on the test set
  4. Compares their performance
  5. Saves the best model
  6. Generates feature importance (for tree-based models)
  7. Saves a metrics JSON for use in the Streamlit dashboard

Run this AFTER preprocessing.py:
  python src/train_full_model.py

Expected output:
  - models/full_dataset_model/best_model.pkl
  - models/full_dataset_model/best_model_name.pkl
  - reports/metrics/full_model_metrics.json
  - reports/figures/full_model_confusion_matrix.png
  - reports/figures/full_model_roc_curve.png
  - reports/figures/full_model_feature_importance.png
  - reports/figures/full_model_comparison.png
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.utils import (
    PROCESSED_DIR, FULL_MODEL_DIR, FIGURES_DIR, METRICS_DIR,
    ensure_dirs, print_section, save_metrics,
)

# Consistent colour palette
COLORS = {
    "phishing"   : "#e74c3c",
    "legitimate" : "#2ecc71",
    "primary"    : "#3498db",
    "secondary"  : "#9b59b6",
    "accent"     : "#f39c12",
}


def load_processed_data() -> tuple:
    """
    Load the preprocessed numpy arrays saved by preprocessing.py.

    Returns
    -------
    X_train, X_test, y_train, y_test  (numpy arrays)
    """
    print_section("Loading Processed Data")

    required = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
    for fname in required:
        path = PROCESSED_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Run preprocessing first:  python src/preprocessing.py"
            )

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test  = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")

    print(f"[INFO] X_train : {X_train.shape}")
    print(f"[INFO] X_test  : {X_test.shape}")
    print(f"[INFO] y_train phishing   : {(y_train == 0).sum():,}")
    print(f"[INFO] y_train legitimate : {(y_train == 1).sum():,}")

    return X_train, X_test, y_train, y_test


def load_feature_names() -> list:
    """Load the feature names saved during preprocessing."""
    path = FULL_MODEL_DIR / "feature_names.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature names not found: {path}\n"
            f"Run preprocessing first."
        )
    return joblib.load(path)


def define_models() -> dict:
    """
    Define the three classifiers to train and compare.

    Each is a standard scikit-learn estimator with sensible defaults.
    We are NOT doing extensive hyperparameter tuning — this keeps
    the project understandable and avoids overfitting to test data.

    Returns
    -------
    dict : {model_name: estimator}
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter     = 1000,   # Give enough iterations to converge
            random_state = 42,
            n_jobs       = -1,     # Use all CPU cores
            C            = 1.0,    # Regularisation strength (default)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth    = 15,     # Limit depth to prevent overfitting
            random_state = 42,
            class_weight = "balanced",  # Handle class imbalance
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators = 100,    # 100 trees — good balance of speed and accuracy
            max_depth    = 15,
            random_state = 42,
            n_jobs       = -1,
            class_weight = "balanced",
        ),
    }
    return models


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    Evaluate a trained model and return a metrics dictionary.

    Positive class = 0 (Phishing) — we want to maximise phishing recall.

    Why phishing recall matters most:
    ----------------------------------
    A False Negative means a phishing site is predicted as legitimate.
    That is dangerous — the user might visit it and get scammed.
    A False Positive means a legitimate site is predicted as phishing.
    That is inconvenient but not dangerous.
    Therefore we pay extra attention to phishing recall.

    Parameters
    ----------
    model      : trained sklearn estimator
    X_test     : np.ndarray
    y_test     : np.ndarray
    model_name : str

    Returns
    -------
    dict with all metrics
    """
    y_pred = model.predict(X_test)

    # Probabilities (if supported)
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        y_proba = model.predict_proba(X_test)[:, 0]  # Probability of class 0 (phishing)
    else:
        y_proba = None

    # Metrics — pos_label=0 means phishing is treated as the positive class
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall    = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1        = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(
        y_test, y_pred,
        target_names=["Phishing (0)", "Legitimate (1)"],
        zero_division=0,
    )

    roc_auc = None
    pr_auc  = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc  = average_precision_score(y_test, y_proba, pos_label=0)
        except Exception:
            pass

    print(f"\n{'─'*50}")
    print(f"  Model: {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}  (phishing)")
    print(f"  Recall    : {recall:.4f}  (phishing)")
    print(f"  F1-score  : {f1:.4f}  (phishing)")
    if roc_auc:
        print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"\n{report}")

    return {
        "model_name" : model_name,
        "accuracy"   : round(float(accuracy),  4),
        "precision"  : round(float(precision), 4),
        "recall"     : round(float(recall),    4),
        "f1_score"   : round(float(f1),        4),
        "roc_auc"    : round(float(roc_auc), 4) if roc_auc else None,
        "pr_auc"     : round(float(pr_auc),  4) if pr_auc  else None,
        "confusion_matrix": cm.tolist(),
        "y_pred"     : y_pred,
        "y_proba"    : y_proba,
    }


def plot_confusion_matrix(cm: list, model_name: str, filename: str) -> None:
    """Save a confusion-matrix heatmap as a PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = "d",
        cmap       = "Blues",
        ax         = ax,
        xticklabels = ["Phishing", "Legitimate"],
        yticklabels = ["Phishing", "Legitimate"],
        linewidths  = 0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = FIGURES_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved → {out_path}")


def plot_roc_curves(results: list, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot ROC curves for all models on a single chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_list = [COLORS["primary"], COLORS["accent"], COLORS["phishing"]]

    for idx, result in enumerate(results):
        if result["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, result["y_proba"], pos_label=0)
            auc_val = result["roc_auc"] or 0.0
            ax.plot(
                fpr, tpr,
                color = colors_list[idx % len(colors_list)],
                lw    = 2,
                label = f"{result['model_name']} (AUC = {auc_val:.3f})",
            )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — All Models (Phishing as Positive Class)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = FIGURES_DIR / "full_model_roc_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] ROC curve saved → {out_path}")


def plot_model_comparison(results: list) -> None:
    """Bar chart comparing all models across key metrics."""
    metrics   = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels    = ["Accuracy", "Precision\n(Phishing)", "Recall\n(Phishing)", "F1-Score\n(Phishing)", "ROC-AUC"]
    model_names = [r["model_name"] for r in results]

    n_metrics = len(metrics)
    n_models  = len(results)
    x         = np.arange(n_metrics)
    width     = 0.25
    offsets   = np.linspace(-width, width, n_models)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_colors = [COLORS["primary"], COLORS["accent"], COLORS["phishing"]]

    for i, (result, color) in enumerate(zip(results, bar_colors)):
        values = [result.get(m) or 0 for m in metrics]
        ax.bar(
            x + offsets[i], values,
            width = width * 0.9,
            label = result["model_name"],
            color = color,
            alpha = 0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Full Dataset (PhiUSIIL)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out_path = FIGURES_DIR / "full_model_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Comparison chart saved → {out_path}")


def plot_feature_importance(model, feature_names: list, model_name: str) -> None:
    """Plot top-20 feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"[INFO] {model_name} does not support feature importance — skipping.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:20]   # Top 20

    top_names  = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        range(len(top_names)), top_values[::-1],
        color = COLORS["primary"], alpha = 0.85,
    )
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini Impurity Reduction)", fontsize=11)
    ax.set_title(f"Top 20 Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / f"full_model_feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature importance chart saved → {out_path}")


def select_best_model(results: list) -> dict:
    """
    Select the best model based on F1-score for phishing detection.

    We use F1-score (phishing) as the primary criterion because:
    - It balances precision and recall.
    - It better reflects real-world phishing detection quality.

    Parameters
    ----------
    results : list of evaluation result dicts

    Returns
    -------
    dict — the result dictionary of the best model
    """
    best = max(results, key=lambda r: r["f1_score"])
    print(f"\n[INFO] Best model by F1-score (phishing): {best['model_name']}")
    return best


def train_and_evaluate() -> None:
    """Main training function — runs the complete training and evaluation pipeline."""
    ensure_dirs()

    print_section("PHISHGUARD AI — Full Dataset Model Training")

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    feature_names = load_feature_names()

    models   = define_models()
    results  = []
    trained  = {}

    # Train and evaluate each model
    for name, model in models.items():
        print_section(f"Training: {name}")
        print(f"[INFO] Fitting {name}...")
        model.fit(X_train, y_train)
        print(f"[OK]  {name} trained.")

        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
        trained[name] = model

        # Save individual confusion matrix
        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            result["confusion_matrix"],
            name,
            f"cm_{safe_name}.png",
        )

    # Plot ROC curves for all models
    plot_roc_curves(results, X_test, y_test)

    # Plot model comparison
    plot_model_comparison(results)

    # Select and save best model
    best_result = select_best_model(results)
    best_model  = trained[best_result["model_name"]]

    # Plot feature importance for best model
    plot_feature_importance(best_model, feature_names, best_result["model_name"])

    # Save best model
    model_path = FULL_MODEL_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"[INFO] Best model saved → {model_path}")

    # Save best model name
    name_path = FULL_MODEL_DIR / "best_model_name.pkl"
    joblib.dump(best_result["model_name"], name_path)

    # Save all results as JSON (for Streamlit dashboard)
    # Remove numpy arrays before serialising
    serialisable_results = []
    for r in results:
        cleaned = {k: v for k, v in r.items() if k not in ("y_pred", "y_proba")}
        serialisable_results.append(cleaned)

    save_metrics(
        {"all_models": serialisable_results, "best_model": best_result["model_name"]},
        "full_model_metrics.json",
        model_name = "Full Dataset — PhiUSIIL",
    )

    # Print final summary table
    print_section("Model Comparison Summary")
    header = f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    print(header)
    print("─" * len(header))
    for r in results:
        auc = f"{r['roc_auc']:.4f}" if r["roc_auc"] else "  N/A  "
        print(
            f"{r['model_name']:<25} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} {r['f1_score']:>8.4f} {auc:>9}"
        )

    print(f"\n[DONE] Training complete. Best model: {best_result['model_name']}")


if __name__ == "__main__":
    train_and_evaluate()
