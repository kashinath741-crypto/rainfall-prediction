"""
src/backend/preprocessing.py
──────────────────────────────────────────────────────────────
Preprocessing module – train/test split and feature/target separation.

Notebook section: "Prepare X/y and split"
  - Separate features (X) from target (y = PRECTOTCORR)
  - Perform stratified train/test split
  - (No StandardScaler is used in this project because Random Forest is
    scale-invariant; the module is provided for completeness and future use.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.backend.utils import get_logger

logger = get_logger(__name__)

# Column to predict
TARGET_COLUMN: str = "PRECTOTCORR"


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate the feature matrix X and target vector y.

    Args:
        df:         Fully engineered DataFrame (includes target column).
        target_col: Name of the rainfall target column.

    Returns:
        (X, y) tuple.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info("Feature matrix X: %s  |  Target y: %s", X.shape, y.shape)
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TRAIN_TEST_SPLIT_SIZE,
    random_state: int = config.RF_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a random train/test split.

    Args:
        X:            Feature matrix.
        y:            Target vector.
        test_size:    Fraction reserved for testing (default from config).
        random_state: RNG seed for reproducibility (default from config).

    Returns:
        (X_train, X_test, y_train, y_test) tuple.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Train: %d samples  |  Test: %d samples", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


def preprocess(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function: split features/target then split train/test.

    Args:
        df: Fully engineered DataFrame.

    Returns:
        (X_train, X_test, y_train, y_test) tuple.
    """
    X, y = split_features_target(df)
    return make_train_test_split(X, y)
