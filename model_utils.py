"""
model_utils.py  –  Utility functions for model management.

Provides helpers to:
  - Inspect model metadata (feature importances, parameters).
  - Verify that saved artifacts are loadable.
  - Generate a quick evaluation summary from a test set.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.backend.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_artifacts() -> dict[str, bool]:
    """
    Check that the expected model artifacts exist on disk and can be loaded.

    Returns:
        dict with keys 'model' and 'features', each mapped to True/False.
    """
    status: dict[str, bool] = {"model": False, "features": False}
    for key, path in [("model", config.MODEL_PATH), ("features", config.FEATURES_PATH)]:
        if path.exists():
            try:
                joblib.load(path)
                status[key] = True
                logger.info("[OK] %s verified at %s", key, path)
            except Exception as exc:
                logger.warning("[WARN] %s exists but failed to load: %s", key, exc)
        else:
            logger.warning("[MISSING] %s not found at %s", key, path)
    return status


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importances(top_n: int = 20) -> pd.DataFrame:
    """
    Return a DataFrame of the top *top_n* feature importances from the saved
    Random Forest model, sorted descending.

    Args:
        top_n: Number of top features to return.

    Returns:
        DataFrame with columns ['feature', 'importance'].
    """
    model    = joblib.load(config.MODEL_PATH)
    features = joblib.load(config.FEATURES_PATH)

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    return pd.DataFrame(
        {
            "feature":    [features[i] for i in indices],
            "importance": importances[indices],
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model metadata
# ─────────────────────────────────────────────────────────────────────────────

def get_model_info() -> dict[str, Any]:
    """
    Return a dict summarising the saved model's configuration and the
    feature space it was trained on.

    Returns:
        dict with keys: n_estimators, max_depth, n_features, feature_names,
        model_version, model_path.
    """
    model    = joblib.load(config.MODEL_PATH)
    features = joblib.load(config.FEATURES_PATH)

    return {
        "n_estimators":  model.n_estimators,
        "max_depth":     model.max_depth,
        "n_features":    len(features),
        "feature_names": features,
        "model_version": config.MODEL_VERSION,
        "model_path":    str(config.MODEL_PATH),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_csv(csv_path: Path) -> dict[str, float]:
    """
    Load the model, process the dataset at *csv_path* using the same
    pipeline as training, and return test-set evaluation metrics.

    This is useful for monitoring model drift over time.

    Args:
        csv_path: Path to the raw UP rainfall CSV.

    Returns:
        dict with 'mae' and 'r2'.
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    # Inline the training pipeline to avoid circular imports
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    df["DATE"]      = pd.to_datetime(
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
    df = pd.get_dummies(df, columns=["DISTRICT"], drop_first=True)
    df.drop(columns=["DATE"], inplace=True)

    model    = joblib.load(config.MODEL_PATH)
    features = joblib.load(config.FEATURES_PATH)

    X = df[[c for c in features if c in df.columns]]
    y = df["PRECTOTCORR"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config.TRAIN_TEST_SPLIT_SIZE,
        random_state=config.RF_RANDOM_STATE,
    )

    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2":  float(r2_score(y_test, y_pred)),
    }
    logger.info("Evaluation – MAE: %.4f  R²: %.4f", metrics["mae"], metrics["r2"])
    return metrics


if __name__ == "__main__":
    import sys
    # Force UTF-8 on Windows to avoid cp1252 issues
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("\n=== Artifact verification ===")
    status = verify_artifacts()
    print(status)

    print("\n=== Model info ===")
    try:
        info = get_model_info()
        for k, v in info.items():
            if k != "feature_names":
                print(f"  {k}: {v}")
        print(f"  feature_names[:5]: {info['feature_names'][:5]}")
    except FileNotFoundError:
        print("  (model not yet trained)")

    print("\n=== Top-10 feature importances ===")
    try:
        df_imp = get_feature_importances(10)
        print(df_imp.to_string(index=False))
    except FileNotFoundError:
        print("  (model not yet trained)")
