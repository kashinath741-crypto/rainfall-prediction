"""
src/backend/predictor.py  –  Model loading, input validation, feature
                              engineering, and prediction logic.

All logic here mirrors the notebook exactly so that predictions produced
by the application are identical to those generated during training.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.backend.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# District lookup tables
# ─────────────────────────────────────────────────────────────────────────────

#: (latitude, longitude) for every district in the dataset.
DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "Aligarh":              (27.88, 78.08),
    "Ambedkarnagar":        (26.43, 82.53),
    "Amethi":               (26.15, 81.81),
    "Auraiya":              (26.46, 79.52),
    "Ayodhya":              (26.79, 82.20),
    "Azamgarh":             (26.07, 83.18),
    "Ballia":               (25.75, 84.15),
    "Banda":                (25.48, 80.34),
    "Barabanki":            (26.93, 81.18),
    "Deoria":               (26.50, 83.78),
    "Etah":                 (27.55, 78.67),
    "Etawah":               (26.78, 79.02),
    "Farrukhabad":          (27.39, 79.58),
    "Fatehpur":             (25.93, 80.81),
    "Firozabad":            (27.15, 78.39),
    "Gautam Buddha Nagar":  (28.57, 77.38),
    "Ghaziabad":            (28.67, 77.44),
    "Ghazipur":             (25.58, 83.58),
    "Gonda":                (27.13, 81.96),
    "Gorakhpur":            (26.76, 83.37),
    "Hamirpur":             (25.96, 80.15),
    "Hapur":                (28.73, 77.78),
    "Hardoi":               (27.39, 80.13),
    "Hathras":              (27.59, 78.05),
    "Jalaun":               (26.15, 79.33),
    "Jaunpur":              (25.74, 82.69),
    "Pilibhit":             (28.63, 79.80),
    "Pratapgarh":           (25.90, 81.99),
    "Prayagraj":            (25.45, 81.84),
    "Rampur":               (28.81, 79.02),
    "Saharanpur":           (29.97, 77.55),
    "Sambhal":              (28.59, 78.56),
    "Shahajanpur":          (27.88, 79.91),
    "Shravasti":            (27.52, 81.93),
    "Sonbhadra":            (24.68, 83.07),
    "Sultanpur":            (26.27, 82.07),
    "Varanasi":             (25.32, 82.97),
    "Amroha":               (28.90, 78.47),
    "Baghpat":              (28.94, 77.22),
    "Bahraich":             (27.57, 81.60),
    "Balrampur":            (27.43, 82.18),
    "Bareilly":             (28.36, 79.42),
    "Basti":                (26.80, 82.73),
    "Bhadohi":              (25.39, 82.57),
    "Bijnor":               (29.37, 78.14),
    "Budaun":               (28.04, 79.12),
    "Bulandshahr":          (28.41, 77.85),
    "Chandauli":            (25.27, 83.27),
    "Chitrakoot":           (25.20, 80.90),
    "Jhansi":               (25.45, 78.57),
    "Kannauj":              (27.06, 79.92),
    "Kanpur":               (26.46, 80.33),
    "Kasganj":              (27.81, 78.65),
    "Kausambhi":            (25.52, 81.38),
    "Kushinagar":           (26.74, 83.89),
    "Lakhimpur":            (27.95, 80.78),
    "Lalitpur":             (24.69, 78.42),
    "Lucknow":              (26.85, 80.95),
    "Maharajganj":          (27.13, 83.56),
    "Mahoba":               (25.29, 79.87),
    "Mainpuri":             (27.23, 79.02),
    "Mathura":              (27.49, 77.67),
    "Mau":                  (25.95, 83.56),
    "Meerut":               (28.98, 77.71),
    "Mirzapur":             (25.14, 82.57),
    "Moradabad":            (28.84, 78.78),
    "Muzaffarnagar":        (29.47, 77.70),
    "Raebareli":            (26.23, 81.24),
    "Sant Kabir Nagar":     (26.79, 83.05),
    "Shamli":               (29.45, 77.31),
    "Siddharth Nagar":      (27.29, 83.07),
    "Sitapur":              (27.56, 80.68),
}

#: Maps district display-name → one-hot column name used during training.
DISTRICT_COLUMN_MAP: dict[str, str] = {
    "Aligarh":             "DISTRICT_Aligarh",
    "Ambedkarnagar":       "DISTRICT_Ambedkarnagar",
    "Amethi":              "DISTRICT_Amethi",
    "Auraiya":             "DISTRICT_Auraiya",
    "Ayodhya":             "DISTRICT_Ayodhya",
    "Azamgarh":            "DISTRICT_Azamgarh",
    "Ballia":              "DISTRICT_Ballia",
    "Banda":               "DISTRICT_Banda",
    "Barabanki":           "DISTRICT_Barabanki",
    "Deoria":              "DISTRICT_Deoria",
    "Etah":                "DISTRICT_Etah",
    "Etawah":              "DISTRICT_Etawah",
    "Farrukhabad":         "DISTRICT_Farrukhabad",
    "Fatehpur":            "DISTRICT_Fatehpur",
    "Firozabad":           "DISTRICT_Firozabad",
    "Gautam Buddha Nagar": "DISTRICT_Gautam Buddha Nagar",
    "Ghaziabad":           "DISTRICT_Ghaziabad",
    "Ghazipur":            "DISTRICT_Ghazipur",
    "Gonda":               "DISTRICT_Gonda",
    "Gorakhpur":           "DISTRICT_Gorakhpur",
    "Hamirpur":            "DISTRICT_Hamirpur",
    "Hapur":               "DISTRICT_Hapur",
    "Hardoi":              "DISTRICT_Hardoi",
    "Hathras":             "DISTRICT_Hathras",
    "Jalaun":              "DISTRICT_Jalaun",
    "Jaunpur":             "DISTRICT_Jaunpur",
    "Pilibhit":            "DISTRICT_Pilibhit",
    "Pratapgarh":          "DISTRICT_Pratapgarh",
    "Prayagraj":           "DISTRICT_Prayagraj",
    "Rampur":              "DISTRICT_Rampur ",
    "Saharanpur":          "DISTRICT_Saharanpur ",
    "Sambhal":             "DISTRICT_Sambhal",
    "Shahajanpur":         "DISTRICT_Shahajanpur",
    "Shravasti":           "DISTRICT_Shravasti",
    "Sonbhadra":           "DISTRICT_Sonbhadra",
    "Sultanpur":           "DISTRICT_Sultanpur",
    "Varanasi":            "DISTRICT_Varanasi",
    "Amroha":              "DISTRICT_amroha",
    "Baghpat":             "DISTRICT_baghpat",
    "Bahraich":            "DISTRICT_bahraich",
    "Balrampur":           "DISTRICT_balrampur",
    "Bareilly":            "DISTRICT_bareilly",
    "Basti":               "DISTRICT_basti",
    "Bhadohi":             "DISTRICT_bhadohi ",
    "Bijnor":              "DISTRICT_bijnor",
    "Budaun":              "DISTRICT_budaun",
    "Bulandshahr":         "DISTRICT_bulandshahr",
    "Chandauli":           "DISTRICT_chandauli",
    "Chitrakoot":          "DISTRICT_chitrakoot",
    "Jhansi":              "DISTRICT_jhansi",
    "Kannauj":             "DISTRICT_kannauj",
    "Kanpur":              "DISTRICT_kanpur",
    "Kasganj":             "DISTRICT_kasganj",
    "Kausambhi":           "DISTRICT_kausambhi",
    "Kushinagar":          "DISTRICT_kushinagar",
    "Lakhimpur":           "DISTRICT_lakhimpur",
    "Lalitpur":            "DISTRICT_lalipur",
    "Lucknow":             "DISTRICT_lucknow",
    "Maharajganj":         "DISTRICT_maharajganj",
    "Mahoba":              "DISTRICT_mahoba",
    "Mainpuri":            "DISTRICT_mainpuri",
    "Mathura":             "DISTRICT_mathura",
    "Mau":                 "DISTRICT_mau",
    "Meerut":              "DISTRICT_meerut",
    "Mirzapur":            "DISTRICT_mirzpur",
    "Moradabad":           "DISTRICT_moradabad",
    "Muzaffarnagar":       "DISTRICT_muzaffarnagar",
    "Raebareli":           "DISTRICT_raebareli",
    "Sant Kabir Nagar":    "DISTRICT_sant kabir nagar",
    "Shamli":              "DISTRICT_shamli",
    "Siddharth Nagar":     "DISTRICT_siddharth nagar",
    "Sitapur":             "DISTRICT_sitapur",
}

#: Typical meteorological defaults per Indian season.
SEASON_DEFAULTS: dict[str, dict[str, float]] = {
    "monsoon": {
        "RH2M": 85, "T2MDEW": 22, "QV2M": 15, "PS": 99,
        "WS50M": 5, "T2MWET": 26, "WD50M": 200,
        "T2M_MAX": 33, "T2M_MIN": 26,
        "ALLSKY_SFC_UV_INDEX": 4, "TS": 32, "PSC": 90, "WSC": 5.5,
    },
    "post-monsoon": {
        "RH2M": 65, "T2MDEW": 14, "QV2M": 9, "PS": 100,
        "WS50M": 3, "T2MWET": 18, "WD50M": 150,
        "T2M_MAX": 30, "T2M_MIN": 18,
        "ALLSKY_SFC_UV_INDEX": 5, "TS": 28, "PSC": 91, "WSC": 3.5,
    },
    "winter": {
        "RH2M": 70, "T2MDEW": 8, "QV2M": 6, "PS": 101,
        "WS50M": 3, "T2MWET": 10, "WD50M": 120,
        "T2M_MAX": 22, "T2M_MIN": 10,
        "ALLSKY_SFC_UV_INDEX": 3, "TS": 18, "PSC": 92, "WSC": 3,
    },
    "pre-monsoon": {
        "RH2M": 40, "T2MDEW": 14, "QV2M": 8, "PS": 99,
        "WS50M": 5, "T2MWET": 22, "WD50M": 250,
        "T2M_MAX": 40, "T2M_MIN": 26,
        "ALLSKY_SFC_UV_INDEX": 8, "TS": 40, "PSC": 89, "WSC": 5.5,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> tuple[Any, list[str]]:
    """
    Load the trained Random Forest model and its expected feature list from
    disk.

    Returns:
        (model, feature_names)  –  the sklearn estimator and ordered list of
        feature column names.

    Raises:
        FileNotFoundError: If the model files are not found on disk.
        RuntimeError:      On any other loading error.
    """
    logger.info("Loading model from %s", config.MODEL_PATH)
    try:
        model    = joblib.load(config.MODEL_PATH)
        features = joblib.load(config.FEATURES_PATH)
        logger.info("Model loaded successfully. Feature count: %d", len(features))
        return model, features
    except FileNotFoundError as exc:
        logger.error("Model file not found: %s", exc)
        raise
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        raise RuntimeError(f"Model loading error: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_input(district: str, year: int, month: int, day: int) -> None:
    """
    Validate user-supplied inputs before attempting a prediction.

    Args:
        district: Name of the UP district.
        year:     Calendar year (must be ≥ 2000).
        month:    Month number [1, 12].
        day:      Day number   [1, 31].

    Raises:
        ValueError: If any input is invalid.
    """
    if district not in DISTRICT_COORDS:
        raise ValueError(
            f"Unknown district '{district}'. "
            f"Must be one of: {sorted(DISTRICT_COORDS)}"
        )
    if year < 2000:
        raise ValueError(f"Year must be ≥ 2000, got {year}.")
    if not 1 <= month <= 12:
        raise ValueError(f"Month must be in [1, 12], got {month}.")
    if not 1 <= day <= 31:
        raise ValueError(f"Day must be in [1, 31], got {day}.")


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  (mirrors notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_input(
    district: str,
    year: int,
    month: int,
    day: int,
    expected_features: list[str],
) -> pd.DataFrame:
    """
    Build and engineer a single-row DataFrame that matches the feature space
    the trained model expects.

    The feature engineering pipeline replicates every step from the training
    notebook:
    - Date-based features  (MONTH, DAY_OF_YEAR, SEASON)
    - Derived meteorological features  (TEMP_RANGE, HUMIDITY_DEW_DIFF, WIND_INTENSITY)
    - One-hot encoding of the DISTRICT column

    Args:
        district:          Name of the Uttar Pradesh district.
        year:              Calendar year of the forecast date.
        month:             Month number [1, 12].
        day:               Day number [1, 31].
        expected_features: Ordered list of feature names used during training.

    Returns:
        A single-row DataFrame aligned with *expected_features*.
    """
    from src.backend.utils import get_season  # local import to avoid circular

    season   = get_season(month)
    defaults = SEASON_DEFAULTS[season]
    lat, lon = DISTRICT_COORDS.get(district, (26.85, 80.95))  # fallback: Lucknow

    raw: dict[str, Any] = {
        "YEAR":                year,
        "MO":                  month,
        "DY":                  day,
        "RH2M":                defaults["RH2M"],
        "T2MDEW":              defaults["T2MDEW"],
        "QV2M":                defaults["QV2M"],
        "PS":                  defaults["PS"],
        "WS50M":               defaults["WS50M"],
        "T2MWET":              defaults["T2MWET"],
        "WD50M":               defaults["WD50M"],
        "T2M_MAX":             defaults["T2M_MAX"],
        "T2M_MIN":             defaults["T2M_MIN"],
        "ALLSKY_SFC_UV_INDEX": defaults["ALLSKY_SFC_UV_INDEX"],
        "TS":                  defaults["TS"],
        "PSC":                 defaults["PSC"],
        "WSC":                 defaults["WSC"],
        "LATITUDE":            lat,
        "LONGITUDE":           lon,
        "DISTRICT":            district,
    }

    df = pd.DataFrame([raw])

    # ── Date-based feature engineering ──────────────────────────────────────
    df["DATE"] = pd.to_datetime(
        df[["YEAR", "MO", "DY"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day"}
        )
    )
    df["MONTH"]       = df["DATE"].dt.month
    df["DAY_OF_YEAR"] = df["DATE"].dt.dayofyear
    df["SEASON"]      = (df["MONTH"] % 12 + 3) // 3

    # ── Derived meteorological features ─────────────────────────────────────
    df["TEMP_RANGE"]        = df["T2M_MAX"]  - df["T2M_MIN"]
    df["HUMIDITY_DEW_DIFF"] = df["RH2M"]     - df["T2MDEW"]
    df["WIND_INTENSITY"]    = df["WS50M"]    * df["WSC"]

    # ── Drop helper columns ──────────────────────────────────────────────────
    df = df.drop(columns=["DATE", "DISTRICT"])

    # ── One-hot encode the district ──────────────────────────────────────────
    # Initialise all district dummy columns to 0 then set the correct one.
    district_col = DISTRICT_COLUMN_MAP.get(district)
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    if district_col and district_col in expected_features:
        df[district_col] = 1

    # ── Align column order with training features ────────────────────────────
    df = df[expected_features]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model: Any,
    df_input: pd.DataFrame,
) -> float:
    """
    Run the model on a pre-processed feature DataFrame and return the
    predicted daily rainfall in millimetres (clipped to ≥ 0).

    Args:
        model:    Trained sklearn estimator.
        df_input: Single-row DataFrame aligned with the model feature space.

    Returns:
        Predicted rainfall in mm (float, ≥ 0).

    Raises:
        RuntimeError: On unexpected prediction errors.
    """
    try:
        raw_pred = model.predict(df_input)[0]
        prediction = float(max(0.0, raw_pred))
        logger.debug("Raw prediction: %.4f  →  clipped: %.4f", raw_pred, prediction)
        return prediction
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise RuntimeError(f"Prediction error: {exc}") from exc
