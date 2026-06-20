"""
src/backend/utils.py  –  Shared utility helpers.

Provides logging setup and lightweight reusable functions used across
the backend (predictor, database, etc.).
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Any

import config


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = __name__) -> logging.Logger:
    """Return a consistently configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Use UTF-8 stream to handle emoji/Unicode on Windows terminals
        stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace'
        ) if hasattr(sys.stdout, 'buffer') else sys.stdout
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(config.LOG_LEVEL)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Rainfall classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def classify_rainfall(amount_mm: float) -> dict[str, Any]:
    """
    Convert a numeric rainfall prediction (mm) into a human-readable
    classification dict.

    Args:
        amount_mm: Predicted rainfall in millimetres per day (≥ 0).

    Returns:
        dict with keys: category (str), headline (str), description (str),
                        icon (str), css_class (str).
    """
    if amount_mm > config.HEAVY_RAIN_THRESHOLD:
        return {
            "category": "heavy",
            "headline": "Heavy Rainfall Expected",
            "description": "Carry an umbrella and stay alert for flooding alerts.",
            "icon": "⛈️",
            "css_class": "result-heavy",
        }
    if amount_mm > config.MODERATE_RAIN_THRESHOLD:
        return {
            "category": "moderate",
            "headline": "Light to Moderate Showers",
            "description": "A light jacket or umbrella would be handy today.",
            "icon": "🌦️",
            "css_class": "result-moderate",
        }
    return {
        "category": "clear",
        "headline": "Clear / Negligible Rain",
        "description": "Enjoy the day — minimal chance of rain.",
        "icon": "☀️",
        "css_class": "result-clear",
    }


def get_season(month: int) -> str:
    """
    Map a calendar month (1-12) to an Indian meteorological season name.

    Args:
        month: Integer in [1, 12].

    Returns:
        One of: 'monsoon', 'post-monsoon', 'winter', 'pre-monsoon'.
    """
    if month in (6, 7, 8, 9):
        return "monsoon"
    if month in (10, 11):
        return "post-monsoon"
    if month in (12, 1, 2):
        return "winter"
    return "pre-monsoon"
