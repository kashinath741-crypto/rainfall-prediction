"""
config.py  –  Central configuration for the Rainfall Prediction project.

All tuneable constants, paths, and environment-based settings live here.
Import this module wherever configuration is needed; never hard-code paths
or magic numbers in other modules.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Project root
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent   # <project root>

# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR: Path       = BASE_DIR / "artifacts" / "models"
PLOTS_DIR: Path        = BASE_DIR / "artifacts" / "plots"
MODEL_PATH: Path       = MODELS_DIR / "rainfall_rf_model.joblib"
FEATURES_PATH: Path    = MODELS_DIR / "model_features.joblib"

# Raw data (read-only; large file kept out of models/)
DATA_DIR: Path         = BASE_DIR               # dataset lives in project root
DATASET_PATH: Path     = DATA_DIR / "UP_rainfall_dataset.csv"

# SQLite prediction log
DB_DIR: Path           = BASE_DIR / "data"
DB_PATH: Path          = DB_DIR   / "predictions.db"

# ─────────────────────────────────────────────────────────────────────────────
# Model training hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
RF_N_ESTIMATORS: int   = 100
RF_RANDOM_STATE: int   = 42
RF_N_JOBS: int         = -1          # use all available cores

TRAIN_TEST_SPLIT_SIZE: float = 0.2   # fraction used for test set

# ─────────────────────────────────────────────────────────────────────────────
# Model metadata
# ─────────────────────────────────────────────────────────────────────────────
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")

# ─────────────────────────────────────────────────────────────────────────────
# Rainfall classification thresholds  (mm / day)
# ─────────────────────────────────────────────────────────────────────────────
HEAVY_RAIN_THRESHOLD: float    = 10.0   # > 10 mm  → heavy
MODERATE_RAIN_THRESHOLD: float = 2.0    # > 2  mm  → moderate / light
# ≤ 2 mm → clear / negligible

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
