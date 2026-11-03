"""
src/data_loader.py
──────────────────
Responsible for downloading and loading the PhiUSIIL dataset.

Supports two modes:
  1. Automatic download via ucimlrepo (requires internet)
  2. Manual CSV loading (fallback when internet is unavailable)

Usage:
  python src/data_loader.py
"""

import sys
import pathlib
import pandas as pd

# ── Project root (one level above this file) ──────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Make sure the folders exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# The CSV file we will save after downloading
DATASET_CSV_PATH = RAW_DATA_DIR / "phiusiil_dataset.csv"

# UCI dataset ID for PhiUSIIL
UCI_DATASET_ID = 967


def download_dataset_via_ucimlrepo() -> pd.DataFrame:
    """
    Downloads the PhiUSIIL dataset directly from UCI using the ucimlrepo package.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with features and target column 'label'.

    Raises
    ------
    ImportError  : if ucimlrepo is not installed.
    Exception    : if the download fails for any reason.
    """
    print("[INFO] Attempting to download PhiUSIIL dataset via ucimlrepo...")

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "ucimlrepo is not installed. Run: pip install ucimlrepo"
        ) from exc

    try:
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch dataset from UCI (id={UCI_DATASET_ID}). "
            f"Check your internet connection. Error: {exc}"
        ) from exc

    # dataset.data.features → DataFrame of input columns
    # dataset.data.targets  → DataFrame with the label column
    features_df = dataset.data.features
    targets_df = dataset.data.targets

    if features_df is None or targets_df is None:
        raise ValueError(
            "ucimlrepo returned empty data. The dataset structure may have changed."
        )

    # Combine features and target into one DataFrame
    combined_df = pd.concat([features_df, targets_df], axis=1)

    print(f"[INFO] Dataset downloaded successfully.")
    print(f"[INFO] Shape: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")

    return combined_df


def load_dataset_from_csv(csv_path: pathlib.Path = DATASET_CSV_PATH) -> pd.DataFrame:
    """
    Loads the PhiUSIIL dataset from a local CSV file.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError : if the CSV does not exist at the given path.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at: {csv_path}\n"
            f"Please download it manually from:\n"
            f"  https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset\n"
            f"Then place the CSV file at: {csv_path}"
        )

    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def load_or_download_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Main entry point.

    First checks if a local CSV already exists.
    If not (or if force_download=True), downloads from UCI and saves locally.

    Parameters
    ----------
    force_download : bool
        Set to True to always re-download even if a local CSV exists.

    Returns
    -------
    pd.DataFrame
    """
    # If the CSV already exists locally, just load it
    if DATASET_CSV_PATH.exists() and not force_download:
        print(f"[INFO] Local dataset found. Loading from cache...")
        return load_dataset_from_csv(DATASET_CSV_PATH)

    # Try downloading from UCI
    try:
        df = download_dataset_via_ucimlrepo()

        # Save a local copy so we don't need to re-download every time
        df.to_csv(DATASET_CSV_PATH, index=False)
        print(f"[INFO] Dataset saved locally at: {DATASET_CSV_PATH}")
        return df

    except Exception as download_error:
        print(f"[WARNING] Automatic download failed: {download_error}")
        print("[INFO] Trying to load from local CSV as fallback...")
        return load_dataset_from_csv(DATASET_CSV_PATH)


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Runs basic sanity checks on the loaded DataFrame.

    Checks:
      - 'label' column exists
      - Label values are only 0 or 1
      - At least 1000 rows
      - At least 10 columns

    Returns True if all checks pass, False otherwise.
    """
    print("\n[INFO] Validating dataset...")

    issues = []

    if "label" not in df.columns:
        issues.append("Missing required column: 'label'")

    if "label" in df.columns:
        unique_labels = set(df["label"].unique())
        expected_labels = {0, 1}
        if not unique_labels.issubset(expected_labels):
            issues.append(
                f"Unexpected label values: {unique_labels}. Expected only 0 and 1."
            )

    if df.shape[0] < 1000:
        issues.append(f"Too few rows: {df.shape[0]}. Expected ~235,795.")

    if df.shape[1] < 10:
        issues.append(f"Too few columns: {df.shape[1]}. Expected ~55.")

    if issues:
        for issue in issues:
            print(f"[ERROR] Validation failed: {issue}")
        return False

    print("[OK] Dataset passed all validation checks.")
    return True


# ── Run directly to download and validate ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PHISHGUARD AI — Dataset Loader")
    print("=" * 60)

    force = "--force" in sys.argv

    df = load_or_download_dataset(force_download=force)
    is_valid = validate_dataset(df)

    if not is_valid:
        print("\n[ERROR] Dataset validation failed. Please check the dataset.")
        sys.exit(1)

    print(f"\n[DONE] Dataset is ready for use.")
    print(f"       Rows    : {df.shape[0]:,}")
    print(f"       Columns : {df.shape[1]}")
    print(f"       Label 0 (Phishing)   : {(df['label'] == 0).sum():,}")
    print(f"       Label 1 (Legitimate) : {(df['label'] == 1).sum():,}")
