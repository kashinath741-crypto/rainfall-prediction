"""
train.py  –  Standalone model training script.

Replicates the notebook pipeline in a production-quality, executable script:
  1. Load and clean the dataset.
  2. Feature engineering.
  3. Train/test split.
  4. Train a Random Forest Regressor.
  5. Evaluate (MAE, R²).
  6. Save model + feature list to disk.

Run from the project root:
    python train.py [--data PATH] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Allow config to be imported from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading & cleaning
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the raw UP rainfall CSV and apply basic cleaning:
      - Remove duplicate rows.
      - Fill missing numeric values with column means.

    Args:
        path: Absolute path to ``UP_rainfall_dataset.csv``.

    Returns:
        Cleaned DataFrame with 20 original columns.
    """
    logger.info("Loading dataset from %s …", path)
    df = pd.read_csv(path)
    logger.info("  Shape: %s", df.shape)

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("  Dropped %d duplicate rows.", before - len(df))

    # Fill remaining numeric NaNs with column means
    df.fillna(df.mean(numeric_only=True), inplace=True)

    logger.info("  Null values remaining: %d", df.isnull().sum().sum())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that improve model performance:
      - DATE                 – parsed from YEAR / MO / DY columns
      - MONTH                – calendar month extracted from DATE
      - DAY_OF_YEAR          – day number within the year (1-366)
      - SEASON               – numeric season label (1-4)
      - TEMP_RANGE           – T2M_MAX − T2M_MIN
      - HUMIDITY_DEW_DIFF    – RH2M − T2MDEW
      - WIND_INTENSITY       – WS50M × WSC

    The temporary DATE column is dropped afterwards; the DISTRICT column is
    one-hot encoded via ``pd.get_dummies``.

    Args:
        df: Cleaned raw DataFrame.

    Returns:
        Transformed DataFrame ready for model training.
    """
    logger.info("Engineering features …")

    df["DATE"] = pd.to_datetime(
        df[["YEAR", "MO", "DY"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day"}
        )
    )
    df["MONTH"]       = df["DATE"].dt.month
    df["DAY_OF_YEAR"] = df["DATE"].dt.dayofyear
    df["SEASON"]      = (df["MONTH"] % 12 + 3) // 3

    df["TEMP_RANGE"]        = df["T2M_MAX"] - df["T2M_MIN"]
    df["HUMIDITY_DEW_DIFF"] = df["RH2M"]    - df["T2MDEW"]
    df["WIND_INTENSITY"]    = df["WS50M"]   * df["WSC"]

    # One-hot encode DISTRICT (drop_first avoids multicollinearity)
    df = pd.get_dummies(df, columns=["DISTRICT"], drop_first=True)

    # Drop the helper DATE column
    df.drop(columns=["DATE"], inplace=True)

    logger.info("  Feature count after engineering: %d", df.shape[1])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> tuple[RandomForestRegressor, list[str], dict]:
    """
    Split the data, train a Random Forest Regressor, and evaluate it.

    Args:
        df: Fully engineered DataFrame (includes target column PRECTOTCORR).

    Returns:
        Tuple of (fitted model, feature list, metrics dict).
        Metrics dict contains 'mae' and 'r2'.
    """
    target = "PRECTOTCORR"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = config.TRAIN_TEST_SPLIT_SIZE,
        random_state= config.RF_RANDOM_STATE,
    )
    logger.info(
        "Train size: %d  |  Test size: %d", len(X_train), len(X_test)
    )

    model = RandomForestRegressor(
        n_estimators = config.RF_N_ESTIMATORS,
        random_state = config.RF_RANDOM_STATE,
        n_jobs       = config.RF_N_JOBS,
    )
    logger.info("Training Random Forest …")
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2":  r2_score(y_test, y_pred),
    }
    logger.info("  MAE : %.4f", metrics["mae"])
    logger.info("  R²  : %.4f", metrics["r2"])

    return model, list(X_train.columns), metrics


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(
    model: RandomForestRegressor,
    features: list[str],
    output_dir: Path,
) -> None:
    """
    Serialise the trained model and feature list to *output_dir*.

    Args:
        model:      Fitted RandomForestRegressor.
        features:   Ordered list of training feature names.
        output_dir: Directory where artifacts will be written (created if absent).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path    = output_dir / "rainfall_rf_model.joblib"
    features_path = output_dir / "model_features.joblib"

    joblib.dump(model,    model_path)
    joblib.dump(features, features_path)

    logger.info("✅  Model saved    → %s", model_path)
    logger.info("✅  Features saved → %s", features_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the UP Rainfall Random Forest model."
    )
    parser.add_argument(
        "--data",
        type    = Path,
        default = config.DATASET_PATH,
        help    = "Path to UP_rainfall_dataset.csv  (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type    = Path,
        default = config.MODELS_DIR,
        help    = "Directory to save model artifacts  (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df       = load_dataset(args.data)
    df       = engineer_features(df)
    model, features, metrics = train(df)
    save_artifacts(model, features, args.output_dir)

    logger.info(
        "Training finished.  MAE=%.4f  R²=%.4f", metrics["mae"], metrics["r2"]
    )


if __name__ == "__main__":
    main()
