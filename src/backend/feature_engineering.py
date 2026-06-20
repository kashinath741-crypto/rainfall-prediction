"""
src/backend/feature_engineering.py
──────────────────────────────────────────────────────────────
Feature engineering module extracted directly from the notebook.

Notebook section: "Feature Engineering"
  - Create DATE from YEAR / MO / DY
  - Extract MONTH and DAY_OF_YEAR from DATE
  - Compute numeric SEASON
  - Compute TEMP_RANGE, HUMIDITY_DEW_DIFF, WIND_INTENSITY
  - One-hot encode DISTRICT
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.backend.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual feature builders
# ─────────────────────────────────────────────────────────────────────────────

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the YEAR / MO / DY columns into a DATE column and derive:
      - MONTH       – calendar month (1–12)
      - DAY_OF_YEAR – day number within the year (1–366)
      - SEASON      – numeric season label  (1 = winter … 4 = monsoon)

    The helper DATE column is **not** dropped here; call
    :func:`drop_date_column` or pass ``drop_date=True`` to
    :func:`engineer_all` if you want it removed.

    Args:
        df: DataFrame containing YEAR, MO, DY columns.

    Returns:
        Same DataFrame with DATE, MONTH, DAY_OF_YEAR, SEASON columns added.
    """
    df["DATE"] = pd.to_datetime(
        df[["YEAR", "MO", "DY"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day"}
        )
    )
    df["MONTH"]       = df["DATE"].dt.month
    df["DAY_OF_YEAR"] = df["DATE"].dt.dayofyear
    # Notebook formula: (month % 12 + 3) // 3
    df["SEASON"]      = (df["MONTH"] % 12 + 3) // 3
    logger.debug("Date features added: MONTH, DAY_OF_YEAR, SEASON")
    return df


def add_meteorological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive three composite meteorological features:
      - TEMP_RANGE        = T2M_MAX − T2M_MIN
      - HUMIDITY_DEW_DIFF = RH2M    − T2MDEW
      - WIND_INTENSITY    = WS50M   × WSC

    Args:
        df: DataFrame that already contains the raw meteorological columns.

    Returns:
        Same DataFrame with three new columns appended.
    """
    df["TEMP_RANGE"]        = df["T2M_MAX"] - df["T2M_MIN"]
    df["HUMIDITY_DEW_DIFF"] = df["RH2M"]    - df["T2MDEW"]
    df["WIND_INTENSITY"]    = df["WS50M"]   * df["WSC"]
    logger.debug(
        "Meteorological features added: TEMP_RANGE, HUMIDITY_DEW_DIFF, WIND_INTENSITY"
    )
    return df


def encode_district(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot encode the DISTRICT column using :func:`pandas.get_dummies`.

    Args:
        df:         DataFrame containing a DISTRICT string column.
        drop_first: Whether to drop the first dummy to avoid multicollinearity.
                    Must be ``True`` during training to match the saved model.

    Returns:
        DataFrame with DISTRICT replaced by binary dummy columns.
    """
    before_cols = df.shape[1]
    df = pd.get_dummies(df, columns=["DISTRICT"], drop_first=drop_first)
    added = df.shape[1] - before_cols
    logger.debug("DISTRICT one-hot encoded: %d new columns added.", added)
    return df


def drop_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the temporary DATE column if it exists."""
    if "DATE" in df.columns:
        df.drop(columns=["DATE"], inplace=True)
        logger.debug("Dropped DATE column.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def engineer_all(df: pd.DataFrame, drop_date: bool = True) -> pd.DataFrame:
    """
    Apply the complete feature-engineering pipeline in the order used by the
    notebook:

    1. :func:`add_date_features`
    2. :func:`add_meteorological_features`
    3. :func:`encode_district`
    4. Drop DATE column (optional)

    Args:
        df:        Cleaned raw DataFrame.
        drop_date: Whether to remove the helper DATE column before returning.

    Returns:
        Fully engineered DataFrame ready for model training or inference.
    """
    logger.info("Running full feature-engineering pipeline …")
    df = add_date_features(df)
    df = add_meteorological_features(df)
    df = encode_district(df)
    if drop_date:
        df = drop_date_column(df)
    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df
