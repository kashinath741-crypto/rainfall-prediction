"""
inference.py  –  Command-line inference tool.

Runs the trained model on a single (district, date) pair and prints
the result, optionally logging it to the SQLite database.

Usage:
    python inference.py --district Lucknow --date 2025-07-15
    python inference.py --district Varanasi --date 2025-01-01 --no-db
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Allow config / src imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backend.main import run_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UP Rainfall Predictor – command-line inference."
    )
    parser.add_argument(
        "--district",
        required=True,
        help="Name of the Uttar Pradesh district (e.g. 'Lucknow').",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Forecast date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        default=False,
        help="Skip saving the prediction to the SQLite database.",
    )
    return parser.parse_args()


def main() -> None:
    # Force UTF-8 output on Windows terminals
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    args = parse_args()

    # Parse date
    try:
        forecast_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        print(f"[ERROR]  Invalid date format: '{args.date}'. Expected YYYY-MM-DD.")
        sys.exit(1)

    try:
        result = run_prediction(args.district, forecast_date)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR]  {exc}")
        sys.exit(1)

    # ── Pretty-print the result ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  {result['headline']}")
    print("=" * 60)
    print(f"  District     : {args.district}")
    print(f"  Date         : {forecast_date.strftime('%d %B %Y')}")
    print(f"  Season       : {result['season'].replace('-', ' ').title()}")
    print(f"  Prediction   : {result['prediction_mm']:.2f} mm / day")
    print(f"  Category     : {result['category']}")
    print(f"  Description  : {result['description']}")
    if not args.no_db:
        print(f"  DB record id : {result['db_row_id']}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
