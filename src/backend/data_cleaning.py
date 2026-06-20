"""
src/backend/data_cleaning.py
──────────────────────────────────────────────────────────────
Data cleaning module extracted directly from the notebook.

Notebook section: "Data Cleaning"
  - Check null values
  - Remove duplicate rows
  - Fill missing numeric values with column means
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.backend.utils import get_logger

logger = get_logger(__name__)


def load_raw_data(csv_path: Path | str) -> pd.DataFrame:
    """
    Load the raw UP rainfall CSV from *csv_path*.

    Args:
        csv_path: Path to ``UP_rainfall_dataset.csv``.

    Returns:
        Raw DataFrame (565 210 rows × 20 columns before cleaning).

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Reading CSV from %s …", path)
    df = pd.read_csv(path)
    logger.info("  Loaded shape: %s", df.shape)
    return df


def check_nulls(df: pd.DataFrame) -> pd.Series:
    """
    Return the null-value count for every column.

    Args:
        df: Any DataFrame.

    Returns:
        pandas Series – null count per column.
    """
    null_counts = df.isnull().sum()
    logger.info("Null counts:\n%s", null_counts)
    return null_counts


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows in-place and report how many were removed.

    Args:
        df: Input DataFrame.

    Returns:
        De-duplicated DataFrame (same object, modified in-place).
    """
    before = len(df)
    df.drop_duplicates(inplace=True)
    removed = before - len(df)
    logger.info("Removed %d duplicate rows. Remaining: %d", removed, len(df))
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill any remaining numeric NaN values with the column mean.

    This matches the notebook cell::

        df.fillna(df.mean(numeric_only=True), inplace=True)

    Args:
        df: DataFrame that may contain NaN values.

    Returns:
        DataFrame with numeric NaNs filled (same object, modified in-place).
    """
    before = int(df.isnull().sum().sum())
    df.fillna(df.mean(numeric_only=True), inplace=True)
    after = int(df.isnull().sum().sum())
    logger.info(
        "fill_missing_values: %d NaN → %d NaN (filled %d cells)",
        before, after, before - after,
    )
    return df


def clean(csv_path: Path | str) -> pd.DataFrame:
    """
    Full cleaning pipeline: load → remove duplicates → fill NaNs.

    This is the convenience function used by ``train.py`` and
    ``feature_engineering.py``.

    Args:
        csv_path: Path to the raw CSV.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = load_raw_data(csv_path)
    check_nulls(df)
    remove_duplicates(df)
    fill_missing_values(df)
    logger.info("Cleaning complete. Final shape: %s", df.shape)
    return df
