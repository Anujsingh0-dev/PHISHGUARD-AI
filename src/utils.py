"""
src/utils.py
────────────
Shared utility functions used across the PHISHGUARD AI project.

Contents:
  - Project path constants
  - Logging helpers
  - File-saving helpers
  - Label mapping helpers

No ML logic lives here. This file only provides shared helpers.
"""

import pathlib
import json
import datetime

# ── Project Root ──────────────────────────────────────────────────────────────
# __file__ is the path of this file (src/utils.py)
# .parent      → src/
# .parent.parent → PHISHGUARD-AI/   (project root)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── Sub-directories ───────────────────────────────────────────────────────────
DATA_DIR         = PROJECT_ROOT / "data"
RAW_DATA_DIR     = DATA_DIR / "raw"
PROCESSED_DIR    = DATA_DIR / "processed"
MODELS_DIR       = PROJECT_ROOT / "models"
FULL_MODEL_DIR   = MODELS_DIR / "full_dataset_model"
URL_MODEL_DIR    = MODELS_DIR / "url_only_model"
REPORTS_DIR      = PROJECT_ROOT / "reports"
FIGURES_DIR      = REPORTS_DIR / "figures"
METRICS_DIR      = REPORTS_DIR / "metrics"

# ── Dataset File ──────────────────────────────────────────────────────────────
DATASET_CSV      = RAW_DATA_DIR / "phiusiil_dataset.csv"

# ── Target Column ─────────────────────────────────────────────────────────────
TARGET_COLUMN    = "label"

# Label convention used throughout this project:
#   0 → Phishing  (the "positive" class for detection purposes)
#   1 → Legitimate
LABEL_MAP = {0: "Phishing", 1: "Legitimate"}


def ensure_dirs() -> None:
    """Create all required project directories if they do not already exist."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DIR,
        FULL_MODEL_DIR,
        URL_MODEL_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def label_to_text(label: int) -> str:
    """
    Convert numeric label to human-readable text.

    Parameters
    ----------
    label : int
        0 for phishing, 1 for legitimate.

    Returns
    -------
    str
        'Phishing' or 'Legitimate'.
    """
    return LABEL_MAP.get(int(label), f"Unknown({label})")


def save_metrics(metrics: dict, filename: str, model_name: str = "") -> pathlib.Path:
    """
    Save a metrics dictionary as a JSON file in the metrics folder.

    Parameters
    ----------
    metrics   : dict  — the evaluation results to save
    filename  : str   — e.g. 'full_model_metrics.json'
    model_name: str   — optional label added to the JSON

    Returns
    -------
    pathlib.Path — path where the file was saved
    """
    ensure_dirs()
    output_path = METRICS_DIR / filename

    payload = {
        "model_name"  : model_name,
        "generated_at": datetime.datetime.now().isoformat(),
        "metrics"     : metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"[INFO] Metrics saved → {output_path}")
    return output_path


def load_metrics(filename: str) -> dict:
    """
    Load a previously saved metrics JSON file.

    Parameters
    ----------
    filename : str — e.g. 'full_model_metrics.json'

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError if the file does not exist.
    """
    path = METRICS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}\n"
            f"Run the training script first to generate evaluation results."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header to the terminal."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
