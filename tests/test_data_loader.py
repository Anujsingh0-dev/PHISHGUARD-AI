"""
tests/test_data_loader.py
──────────────────────────
Tests for dataset loading functions (src/data_loader.py).

Run with:
  pytest tests/test_data_loader.py -v
"""

import sys
import pathlib
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data_loader import validate_dataset, load_dataset_from_csv
from src.utils import DATASET_CSV


class TestValidateDataset:
    """Test the dataset validation function."""

    def _make_df(self, rows=5000, label_values=None):
        """Create a minimal valid DataFrame for testing."""
        import numpy as np
        if label_values is None:
            label_values = [0, 1] * (rows // 2)
        data = {f"feature_{i}": np.random.rand(rows) for i in range(10)}
        data["label"] = label_values[:rows]
        return pd.DataFrame(data)

    def test_valid_dataset_passes(self):
        """A proper DataFrame should pass validation."""
        df = self._make_df(5000)
        assert validate_dataset(df) is True

    def test_missing_label_column_fails(self):
        """Dataset without 'label' column should fail."""
        df = self._make_df(5000)
        df = df.drop(columns=["label"])
        assert validate_dataset(df) is False

    def test_too_few_rows_fails(self):
        """Dataset with fewer than 1000 rows should fail."""
        df = self._make_df(100)
        assert validate_dataset(df) is False

    def test_too_few_columns_fails(self):
        """Dataset with fewer than 10 columns should fail."""
        import numpy as np
        df = pd.DataFrame({
            "f1": range(5000),
            "f2": range(5000),
            "label": [0, 1] * 2500,
        })
        assert validate_dataset(df) is False

    def test_invalid_labels_fail(self):
        """Labels other than 0 or 1 should fail validation."""
        df = self._make_df(5000, label_values=[0, 1, 2, 3] * 1250)
        assert validate_dataset(df) is False


class TestLoadFromCsv:
    """Test CSV loading."""

    def test_missing_csv_raises_filenotfounderror(self):
        """Loading a non-existent CSV should raise FileNotFoundError."""
        bad_path = pathlib.Path("/tmp/definitely_does_not_exist_12345.csv")
        with pytest.raises(FileNotFoundError):
            load_dataset_from_csv(bad_path)

    def test_local_csv_loads_if_exists(self):
        """If the dataset CSV exists locally, it should load successfully."""
        if not DATASET_CSV.exists():
            pytest.skip("Dataset CSV not available — skipping local CSV test.")
        df = load_dataset_from_csv(DATASET_CSV)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "label" in df.columns
