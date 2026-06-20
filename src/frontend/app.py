"""
src/frontend/app.py  –  Flask web application.

Routes
──────
GET  /              → index page (prediction form)
POST /predict       → run prediction, return result page
GET  /history       → prediction history page
GET  /api/districts → JSON list of all districts (for autocomplete)
GET  /health        → JSON health-check endpoint

Run locally:
    python src/frontend/app.py

Or with the Flask CLI:
    set FLASK_APP=src/frontend/app.py
    flask run --debug
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

# ── Project-root imports ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config
from src.backend.database import fetch_recent_predictions
from src.backend.main import run_prediction
from src.backend.predictor import DISTRICT_COORDS

# ── Flask setup ──────────────────────────────────────────────────────────────
TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR   = Path(__file__).parent / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = os.getenv("SECRET_KEY", "rainfall-predictor-dev-key-2025")

# Sorted district list (used in templates)
ALL_DISTRICTS: list[str] = sorted(DISTRICT_COORDS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Render the main prediction form."""
    today = date.today().isoformat()
    return render_template(
        "index.html",
        districts=ALL_DISTRICTS,
        today=today,
        title="UP Rainfall Predictor",
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle the prediction form submission.

    Reads *district* and *forecast_date* from the POST body,
    runs the model, then renders the result page.
    """
    district      = request.form.get("district", "").strip()
    date_str      = request.form.get("forecast_date", "").strip()

    # ── Input validation ─────────────────────────────────────────────────────
    errors: list[str] = []

    if not district:
        errors.append("Please select a district.")
    elif district not in DISTRICT_COORDS:
        errors.append(f"Unknown district: '{district}'.")

    forecast_date: date | None = None
    if not date_str:
        errors.append("Please select a forecast date.")
    else:
        try:
            forecast_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            errors.append("Invalid date format. Please use the date picker.")

    if errors:
        today = date.today().isoformat()
        return render_template(
            "index.html",
            districts=ALL_DISTRICTS,
            today=today,
            title="UP Rainfall Predictor",
            errors=errors,
            selected_district=district,
            selected_date=date_str,
        ), 422

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        result = run_prediction(district, forecast_date)  # type: ignore[arg-type]
    except FileNotFoundError:
        return render_template(
            "error.html",
            title="Model Not Found",
            message=(
                "The trained model artifacts could not be located. "
                "Please run <code>python train.py</code> first, then restart the server."
            ),
        ), 500
    except Exception as exc:
        return render_template(
            "error.html",
            title="Prediction Error",
            message=str(exc),
        ), 500

    return render_template(
        "result.html",
        title="Prediction Result",
        district=district,
        forecast_date=forecast_date.strftime("%d %B %Y"),
        season=result["season"].replace("-", " ").title(),
        prediction_mm=round(result["prediction_mm"], 2),
        category=result["category"],
        headline=result["headline"],
        description=result["description"],
        icon=result["icon"],
        css_class=result["css_class"],
        defaults=result["defaults"],
        lat=result["lat"],
        lon=result["lon"],
        db_row_id=result["db_row_id"],
    )


@app.route("/history", methods=["GET"])
def history():
    """Render the prediction history page (last 50 records)."""
    records = fetch_recent_predictions(limit=50)
    return render_template(
        "history.html",
        title="Prediction History",
        records=records,
    )


@app.route("/api/districts", methods=["GET"])
def api_districts():
    """Return a JSON array of all district names."""
    return jsonify(ALL_DISTRICTS)


@app.route("/health", methods=["GET"])
def health():
    """Lightweight health-check for uptime monitoring."""
    model_ok    = config.MODEL_PATH.exists()
    features_ok = config.FEATURES_PATH.exists()
    return jsonify(
        {
            "status":       "ok" if (model_ok and features_ok) else "degraded",
            "model_ready":  model_ok,
            "features_ready": features_ok,
            "version":      config.MODEL_VERSION,
        }
    ), 200 if (model_ok and features_ok) else 503


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if sys.platform.startswith('win') and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    print(f"\n🌧️  UP Rainfall Predictor running at http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
