"""
src/backend/model_loader.py
──────────────────────────────────────────────────────────────
Thin wrapper around joblib that loads trained model artifacts.

Provides a single cached load function so the heavy model file is
read from disk only once per process lifetime.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.backend.utils import get_logger

logger = get_logger(__name__)

# Module-level cache (avoids reloading on every request)
_cache: dict[str, Any] = {}


def load_model() -> Any:
    """
    Load and cache the trained Random Forest Regressor.

    Returns:
        The sklearn model object.

    Raises:
        FileNotFoundError: If the model file does not exist on disk.
    """
    if "model" not in _cache:
        path = config.MODEL_PATH
        logger.info("Loading model from %s …", path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run  python train.py  first to generate the artifact."
            )
        _cache["model"] = joblib.load(path)
        logger.info("Model loaded successfully.")
    return _cache["model"]


def load_features() -> list[str]:
    """
    Load and cache the ordered list of feature names used during training.

    Returns:
        List of column-name strings.

    Raises:
        FileNotFoundError: If the features file does not exist on disk.
    """
    if "features" not in _cache:
        path = config.FEATURES_PATH
        logger.info("Loading feature list from %s …", path)
        if not path.exists():
            raise FileNotFoundError(
                f"Feature list file not found: {path}\n"
                "Run  python train.py  first to generate the artifact."
            )
        _cache["features"] = joblib.load(path)
        logger.info("Feature list loaded (%d features).", len(_cache["features"]))
    return _cache["features"]


def reload_all() -> None:
    """
    Force a fresh reload of all cached artifacts (useful after retraining).
    """
    _cache.clear()
    load_model()
    load_features()
    logger.info("All artifacts reloaded.")
