"""
src/train_url_model.py
───────────────────────
Trains a URL-only model (MODE B) for live URL prediction in the Streamlit app.

This is a SEPARATE model from the full-dataset model (MODE A).

Why a separate model?
─────────────────────
The full dataset contains features derived from:
  - The URL string (e.g. URL length, number of dots)
  - The actual webpage (e.g. HTML content, number of form elements, SSL details)

When a user pastes a URL into the Streamlit app, we only have the URL string.
We cannot download the webpage without opening the site — which is a security risk.
Therefore we train a model that uses ONLY URL-string features.

This model will have lower accuracy than the full model —
and we are completely honest about that in the app.

How this script works:
  1. Load the raw dataset (needs 'URL' and 'label' columns)
  2. Extract URL features from each URL using feature_extraction.py
  3. Train the same three classifiers on URL features only
  4. Save the best model and its metrics

Run this AFTER downloading the dataset:
  python src/train_url_model.py

Expected output:
  - models/url_only_model/url_model.pkl
  - models/url_only_model/url_model_name.pkl
  - reports/metrics/url_model_metrics.json
  - reports/figures/url_model_*.png
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute        import SimpleImputer
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, average_precision_score,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.utils import (
    DATASET_CSV, URL_MODEL_DIR, FIGURES_DIR, METRICS_DIR,
    TARGET_COLUMN, ensure_dirs, print_section, save_metrics,
)
from src.data_loader import load_or_download_dataset
from src.feature_extraction import extract_url_features, URL_FEATURE_NAMES

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Colour palette
COLORS = {
    "phishing"   : "#e74c3c",
    "legitimate" : "#2ecc71",
    "primary"    : "#3498db",
    "accent"     : "#f39c12",
}


def extract_features_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply extract_url_features() to every URL in the dataset.

    This creates the training features for the URL-only model.
    We use the same function that will be called during live prediction,
    so training and inference are guaranteed to use identical features.

    Parameters
    ----------
    df : pd.DataFrame — must contain a 'URL' column and 'label' column

    Returns
    -------
    pd.DataFrame — one row per URL, columns = URL_FEATURE_NAMES
    """
    if "URL" not in df.columns:
        raise ValueError(
            "Column 'URL' not found in the dataset.\n"
            "This is required for the URL-only model.\n"
            f"Available columns: {list(df.columns)}"
        )

    print(f"[INFO] Extracting URL features from {len(df):,} URLs...")
    print(f"[INFO] This may take a few minutes for large datasets...")

    feature_rows = []
    for url in tqdm(df["URL"], desc="Extracting URL features", unit="URL"):
        feature_rows.append(extract_url_features(str(url)))

    features_df = pd.DataFrame(feature_rows, columns=URL_FEATURE_NAMES)
    print(f"[INFO] URL features extracted: {features_df.shape}")
    return features_df


def evaluate_url_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """Evaluate a trained URL-only model. Mirrors train_full_model.evaluate_model()."""
    y_pred = model.predict(X_test)

    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        y_proba = model.predict_proba(X_test)[:, 0]
    else:
        y_proba = None

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
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            pass

    print(f"\n{'─'*50}")
    print(f"  URL Model: {model_name}")
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
        "confusion_matrix": cm.tolist(),
        "y_pred"     : y_pred,
        "y_proba"    : y_proba,
    }


def plot_url_confusion_matrix(cm: list, model_name: str) -> None:
    """Save confusion matrix for URL model."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        ax=ax,
        xticklabels=["Phishing", "Legitimate"],
        yticklabels=["Phishing", "Legitimate"],
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(f"URL Model — Confusion Matrix\n({model_name})", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = FIGURES_DIR / "url_model_confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] URL confusion matrix saved → {out_path}")


def plot_url_feature_importance(model, model_name: str) -> None:
    """Plot URL feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"[INFO] {model_name} does not support feature importance — skipping.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_names   = [URL_FEATURE_NAMES[i] for i in indices]
    top_values  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(top_names)), top_values[::-1],
        color=COLORS["primary"], alpha=0.85,
    )
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"URL Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out_path = FIGURES_DIR / "url_model_feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] URL feature importance saved → {out_path}")


def plot_url_roc(results: list, y_test: np.ndarray) -> None:
    """Plot ROC curves for all URL models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_list = [COLORS["primary"], COLORS["accent"], COLORS["phishing"]]

    for idx, result in enumerate(results):
        if result["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, result["y_proba"], pos_label=0)
            auc_val = result["roc_auc"] or 0.0
            ax.plot(fpr, tpr, color=colors_list[idx % 3], lw=2,
                    label=f"{result['model_name']} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("URL Model — ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = FIGURES_DIR / "url_model_roc_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] URL ROC curve saved → {out_path}")


def train_url_model() -> None:
    """Main function to train the URL-only model."""
    ensure_dirs()
    URL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print_section("PHISHGUARD AI — URL-Only Model Training (MODE B)")

    # Step 1: Load dataset (need URL column + label)
    df = load_or_download_dataset()

    # Step 2: Extract URL features
    X = extract_features_from_dataset(df)
    y = df[TARGET_COLUMN].reset_index(drop=True)

    print(f"\n[INFO] URL feature matrix shape: {X.shape}")
    print(f"[INFO] Class distribution — Phishing: {(y==0).sum():,} | Legitimate: {(y==1).sum():,}")

    # Step 3: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"\n[INFO] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # Step 4: Preprocessing pipeline — scale features
    preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Save the URL preprocessor
    joblib.dump(preprocessor, URL_MODEL_DIR / "url_preprocessor.pkl")
    print(f"[INFO] URL preprocessor saved.")

    # Step 5: Define and train models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, random_state=RANDOM_STATE, class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced",
        ),
    }

    results = []
    trained = {}

    for name, model in models.items():
        print_section(f"Training URL Model: {name}")
        model.fit(X_train_proc, y_train)
        print(f"[OK]  {name} trained.")

        result = evaluate_url_model(model, X_test_proc, y_test.values, name)
        results.append(result)
        trained[name] = model

    # Step 6: Plot curves
    plot_url_roc(results, y_test.values)

    # Step 7: Select best model
    best_result = max(results, key=lambda r: r["f1_score"])
    best_model  = trained[best_result["model_name"]]

    print(f"\n[INFO] Best URL model: {best_result['model_name']}")

    # Plot for best model
    plot_url_confusion_matrix(best_result["confusion_matrix"], best_result["model_name"])
    plot_url_feature_importance(best_model, best_result["model_name"])

    # Step 8: Save best model
    joblib.dump(best_model, URL_MODEL_DIR / "url_model.pkl")
    joblib.dump(best_result["model_name"], URL_MODEL_DIR / "url_model_name.pkl")
    print(f"[INFO] URL model saved → {URL_MODEL_DIR / 'url_model.pkl'}")

    # Step 9: Save metrics
    serialisable = [
        {k: v for k, v in r.items() if k not in ("y_pred", "y_proba")}
        for r in results
    ]
    save_metrics(
        {"all_models": serialisable, "best_model": best_result["model_name"],
         "feature_names": URL_FEATURE_NAMES},
        "url_model_metrics.json",
        model_name="URL-Only Model",
    )

    # Step 10: Print comparison
    print_section("URL Model Comparison Summary")
    header = f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    print(header)
    print("─" * len(header))
    for r in results:
        auc = f"{r['roc_auc']:.4f}" if r["roc_auc"] else "  N/A  "
        print(
            f"{r['model_name']:<25} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>8.4f} {r['f1_score']:>8.4f} {auc:>9}"
        )

    print(f"\n[DONE] URL model training complete.")
    print(f"\n⚠️  LIMITATION REMINDER:")
    print(f"    The URL-only model uses only {len(URL_FEATURE_NAMES)} URL-derived features.")
    print(f"    It cannot examine webpage content, SSL certificates, or hosting details.")
    print(f"    Its performance will be lower than the full-dataset model.")
    print(f"    Always treat predictions as probabilistic hints — not guarantees.")


if __name__ == "__main__":
    train_url_model()
