"""
src/preprocessing.py
────────────────────
Data preprocessing for the FULL DATASET model (Mode A).

What this file does:
  1. Loads the raw dataset
  2. Drops columns that should not be used as features
  3. Separates features (X) from the target (y)
  4. Saves a processed version for reuse
  5. Builds and returns the sklearn preprocessing pipeline

The preprocessing pipeline is fitted ONLY on training data (not the whole dataset).
This is critical to avoid "data leakage" — a situation where the model
accidentally sees test information during training.

Data Leakage (simple explanation):
  Imagine if a student sees the exam answer key while studying.
  They'd score 100% — but not because they truly learned.
  Data leakage is the same idea in ML:
  if preprocessing uses test data, evaluation scores look great
  but the model fails on truly new examples.
"""

import sys
import pathlib
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import shared utilities
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.utils import (
    DATASET_CSV, PROCESSED_DIR, FULL_MODEL_DIR, TARGET_COLUMN,
    ensure_dirs, print_section,
)
from src.data_loader import load_or_download_dataset

# ── Columns to always exclude from model input ────────────────────────────────
# FILENAME: a file path identifier — not a website feature
# DOMAIN, URL, TLD, Title: these are high-cardinality text strings.
#   Raw URLs and domain names cannot be fed directly into a numeric ML model.
#   They would need complex NLP or feature engineering to be useful.
#   In this project, URL-derived features are handled separately in MODE B.
COLUMNS_TO_DROP = [
    "FILENAME",   # Identifier — not present in this dataset but safe to list
    "URL",        # Raw text — handled by feature_extraction.py in URL model
    "Domain",     # Raw text hostname — 220,086 unique values, too high cardinality
    "TLD",        # Top-level domain string — 695 unique values, needs encoding
    "Title",      # Webpage title — raw text, requires NLP, 197,874 unique values
]

# Random state for reproducibility
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 20% for testing, 80% for training


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate the DataFrame into features (X) and target (y).

    Also drops columns that should not be used as model inputs.

    Parameters
    ----------
    df : pd.DataFrame — the full dataset

    Returns
    -------
    X : pd.DataFrame — input features
    y : pd.Series    — target labels (0 = phishing, 1 = legitimate)
    """
    print_section("Feature Preparation")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Separate target
    y = df[TARGET_COLUMN].copy()
    print(f"[INFO] Target column: '{TARGET_COLUMN}'")
    print(f"[INFO] Label 0 (Phishing)   : {(y == 0).sum():,}")
    print(f"[INFO] Label 1 (Legitimate) : {(y == 1).sum():,}")

    # Drop target and excluded columns from features
    cols_to_drop = [TARGET_COLUMN] + [
        col for col in COLUMNS_TO_DROP if col in df.columns
    ]
    X = df.drop(columns=cols_to_drop)

    # Identify and drop any remaining non-numeric columns
    # (After dropping the known text columns above, remaining should be numeric)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"[WARNING] Dropping additional non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    print(f"[INFO] Features shape: {X.shape[0]:,} rows × {X.shape[1]} columns")
    print(f"[INFO] Feature columns used: {list(X.columns)}")

    return X, y


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build a scikit-learn preprocessing Pipeline for numeric features.

    Steps:
      1. SimpleImputer  — fills any missing values with the column median
                          (median is better than mean when data has outliers)
      2. StandardScaler — standardises each feature to zero mean and unit variance
                          (required for Logistic Regression; harmless for trees)

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return pipeline


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Why stratify?
    -------------
    The dataset has ~57% legitimate and ~43% phishing.
    Without stratification, by pure chance the test set might have
    a different ratio — making evaluation misleading.
    Stratification ensures the class ratio is preserved in both splits.

    Parameters
    ----------
    X            : pd.DataFrame — features
    y            : pd.Series    — labels
    test_size    : float        — fraction for testing (0.20 = 20%)
    random_state : int          — seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    print_section("Train-Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        stratify     = y,          # Preserve class distribution
        random_state = random_state,
    )

    print(f"[INFO] Training set : {X_train.shape[0]:,} rows ({(1 - test_size)*100:.0f}%)")
    print(f"[INFO] Testing set  : {X_test.shape[0]:,} rows ({test_size*100:.0f}%)")
    print(f"[INFO] Train phishing   : {(y_train == 0).sum():,}")
    print(f"[INFO] Train legitimate : {(y_train == 1).sum():,}")
    print(f"[INFO] Test  phishing   : {(y_test == 0).sum():,}")
    print(f"[INFO] Test  legitimate : {(y_test == 1).sum():,}")

    return X_train, X_test, y_train, y_test


def run_preprocessing() -> dict:
    """
    Run the complete preprocessing pipeline:
      1. Load dataset
      2. Prepare features
      3. Split into train/test
      4. Fit the preprocessor on training data only
      5. Transform both train and test
      6. Save the processed arrays and fitted preprocessor

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test,
                    X_train_proc, X_test_proc,
                    preprocessor, feature_names
    """
    ensure_dirs()

    # Step 1: Load dataset
    print_section("Loading Dataset")
    df = load_or_download_dataset()

    # Step 2: Prepare features
    X, y = prepare_features(df)
    feature_names = list(X.columns)

    # Step 3: Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Build and fit preprocessor ON TRAINING DATA ONLY
    print_section("Fitting Preprocessor")
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    print("[INFO] Preprocessor fitted on training data.")

    # Step 5: Transform test data using the ALREADY FITTED preprocessor
    X_test_proc = preprocessor.transform(X_test)
    print("[INFO] Test data transformed.")

    # Step 6: Save processed data and the fitted pipeline
    print_section("Saving Processed Artifacts")

    preprocessor_path = FULL_MODEL_DIR / "preprocessor.pkl"
    FULL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, preprocessor_path)
    print(f"[INFO] Preprocessor saved → {preprocessor_path}")

    feature_names_path = FULL_MODEL_DIR / "feature_names.pkl"
    joblib.dump(feature_names, feature_names_path)
    print(f"[INFO] Feature names saved → {feature_names_path}")

    # Save processed numpy arrays for reuse
    np.save(PROCESSED_DIR / "X_train.npy", X_train_proc)
    np.save(PROCESSED_DIR / "X_test.npy",  X_test_proc)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.values)
    np.save(PROCESSED_DIR / "y_test.npy",  y_test.values)

    # Also save unprocessed for reference
    X_train.to_csv(PROCESSED_DIR / "X_train_raw.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test_raw.csv",   index=False)

    print(f"\n[DONE] Preprocessing complete.")
    print(f"       Training samples : {X_train_proc.shape[0]:,}")
    print(f"       Test samples     : {X_test_proc.shape[0]:,}")
    print(f"       Features         : {X_train_proc.shape[1]}")

    return {
        "X_train"       : X_train,
        "X_test"        : X_test,
        "y_train"       : y_train,
        "y_test"        : y_test,
        "X_train_proc"  : X_train_proc,
        "X_test_proc"   : X_test_proc,
        "preprocessor"  : preprocessor,
        "feature_names" : feature_names,
    }


if __name__ == "__main__":
    run_preprocessing()
