"""
src/backend/main.py  –  Thin orchestration layer.

Wires together model loading, input validation, feature preprocessing,
prediction, and database logging into a single callable used by the
Streamlit frontend.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.backend.database import save_prediction
from src.backend.predictor import (
    DISTRICT_COORDS,
    load_model,
    predict,
    preprocess_input,
    validate_input,
)
from src.backend.utils import classify_rainfall, get_logger, get_season

logger = get_logger(__name__)

# Cache the model in module-level variables so it is only loaded once per
# process (Streamlit's st.cache_resource is handled in the frontend layer).
_MODEL: Any = None
_FEATURES: list[str] = []


def _ensure_model_loaded() -> None:
    """Lazy-load the model the first time a prediction is requested."""
    global _MODEL, _FEATURES
    if _MODEL is None:
        _MODEL, _FEATURES = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction(
    district: str,
    forecast_date: date,
) -> dict[str, Any]:
    """
    End-to-end prediction pipeline.

    Steps:
    1. Validate inputs.
    2. Ensure the model is loaded.
    3. Engineer features (mirrors notebook preprocessing).
    4. Run the Random Forest to get a rainfall estimate.
    5. Classify the result (clear / moderate / heavy).
    6. Persist the prediction to the SQLite log.
    7. Return a rich result dict for the frontend.

    Args:
        district:      Name of the Uttar Pradesh district.
        forecast_date: Python ``date`` object for the desired day.

    Returns:
        dict with keys:
            prediction_mm  (float)   – rainfall amount in mm
            category       (str)     – 'clear', 'moderate', or 'heavy'
            headline       (str)     – short descriptive label
            description    (str)     – explanatory text
            icon           (str)     – emoji for the result
            css_class      (str)     – CSS class for styling
            season         (str)     – detected season
            defaults       (dict)    – meteorological defaults used
            lat, lon       (float)   – district coordinates
            db_row_id      (int)     – id of the inserted DB record

    Raises:
        ValueError:    On invalid inputs.
        FileNotFoundError: If model artifacts are missing from disk.
        RuntimeError:  On prediction failure.
    """
    year, month, day = forecast_date.year, forecast_date.month, forecast_date.day

    logger.info(
        "Prediction request: district=%s  date=%s-%02d-%02d",
        district, year, month, day,
    )

    # Step 1 – validate
    validate_input(district, year, month, day)

    # Step 2 – load model (once)
    _ensure_model_loaded()

    # Step 3 – feature engineering
    df_input = preprocess_input(district, year, month, day, _FEATURES)

    # Step 4 – predict
    prediction_mm = predict(_MODEL, df_input)

    # Step 5 – classify
    classification = classify_rainfall(prediction_mm)

    # Step 6 – persist
    season = get_season(month)
    lat, lon = DISTRICT_COORDS.get(district, (26.85, 80.95))

    db_row_id = save_prediction(
        district      = district,
        forecast_date = forecast_date.isoformat(),
        season        = season,
        prediction_mm = prediction_mm,
        category      = classification["category"],
        latitude      = lat,
        longitude     = lon,
    )

    # Step 7 – build response
    from src.backend.predictor import SEASON_DEFAULTS
    result: dict[str, Any] = {
        "prediction_mm": prediction_mm,
        "season":        season,
        "defaults":      SEASON_DEFAULTS[season],
        "lat":           lat,
        "lon":           lon,
        "db_row_id":     db_row_id,
        **classification,
    }

    logger.info(
        "Prediction complete: %.2f mm [%s] for %s on %s (db_id=%d)",
        prediction_mm, classification["category"],
        district, forecast_date.isoformat(), db_row_id,
    )

    return result
