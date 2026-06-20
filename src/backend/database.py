"""
src/backend/database.py  –  SQLite persistence layer.

Stores every prediction request together with the user's inputs,
the model's output, and a timestamp.  Uses only the standard-library
`sqlite3` module — no external ORM required.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow imports from the project root regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.backend.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,          -- ISO-8601 UTC
    district        TEXT    NOT NULL,
    forecast_date   TEXT    NOT NULL,          -- YYYY-MM-DD
    season          TEXT    NOT NULL,
    prediction_mm   REAL    NOT NULL,
    category        TEXT    NOT NULL,          -- clear / moderate / heavy
    model_version   TEXT    NOT NULL,
    latitude        REAL,
    longitude       REAL
);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Ensure the SQLite database and *predictions* table exist.
    Creates the data directory if it does not yet exist.
    """
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
    logger.info("Database ready at %s", config.DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction(
    district: str,
    forecast_date: str,          # "YYYY-MM-DD"
    season: str,
    prediction_mm: float,
    category: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> int:
    """
    Persist one prediction record to the database.

    Args:
        district:      Name of the UP district.
        forecast_date: ISO date string "YYYY-MM-DD".
        season:        Season label (e.g. 'monsoon').
        prediction_mm: Model output in millimetres.
        category:      Classification label ('clear', 'moderate', 'heavy').
        latitude:      Decimal latitude of the district (optional).
        longitude:     Decimal longitude of the district (optional).

    Returns:
        The auto-generated row *id* for the inserted record.
    """
    init_db()          # idempotent – creates table if missing

    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with sqlite3.connect(config.DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (timestamp, district, forecast_date, season,
                 prediction_mm, category, model_version, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                district,
                forecast_date,
                season,
                round(prediction_mm, 4),
                category,
                config.MODEL_VERSION,
                latitude,
                longitude,
            ),
        )
        conn.commit()
        row_id: int = cursor.lastrowid  # type: ignore[assignment]

    logger.info(
        "Prediction saved (id=%d): %s / %s -> %.2f mm [%s]",
        row_id, district, forecast_date, prediction_mm, category,
    )
    return row_id


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────

def fetch_recent_predictions(limit: int = 20) -> list[dict[str, Any]]:
    """
    Return the most recent *limit* prediction records, newest first.

    Args:
        limit: Maximum number of rows to retrieve.

    Returns:
        List of dicts; each dict corresponds to one database row.
    """
    init_db()

    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def fetch_all_predictions() -> list[dict[str, Any]]:
    """Return every prediction record, newest first."""
    init_db()

    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC"
        ).fetchall()

    return [dict(row) for row in rows]
