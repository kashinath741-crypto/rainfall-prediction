# 🌧️ UP Rainfall Predictor

An AI-powered daily rainfall prediction system for all 75 districts of Uttar Pradesh, India.  
Trained on 565 K+ records of NASA POWER climate data using a Random Forest Regressor (MAE ≈ 1.26 mm, R² ≈ 0.77).

---

## 📁 Project Structure

```
Rainfall Prediction/
│
├── notebook/
│   └── Rainfall_Model_FINAL.ipynb     ← Original source notebook
│
├── artifacts/                         ← Model artifacts & plots
│   ├── models/                        ← Trained model files
│   │   ├── rainfall_rf_model.joblib
│   │   └── model_features.joblib
│   └── plots/                         ← Saved evaluation plots
│
├── data/                              ← SQLite prediction log (auto-created)
│   └── predictions.db
│
├── src/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py                    ← Orchestration layer
│   │   ├── data_cleaning.py           ← Load & clean dataset
│   │   ├── feature_engineering.py     ← Derive new features
│   │   ├── preprocessing.py           ← Train/test split helpers
│   │   ├── model_loader.py            ← Cached joblib loader
│   │   ├── predictor.py               ← Feature engineering for inference
│   │   ├── database.py                ← SQLite persistence
│   │   └── utils.py                   ← Logging, classification helpers
│   │
│   └── frontend/
│       ├── __init__.py
│       ├── app.py                     ← Flask web application
│       ├── templates/
│       │   ├── index.html             ← Prediction form
│       │   ├── result.html            ← Prediction result
│       │   ├── history.html           ← Prediction history table
│       │   └── error.html             ← Error page
│       └── static/
│           ├── css/
│           │   └── style.css          ← Glassmorphism dark theme
│           └── js/
│               └── rain.js            ← Animated rain background
│
├── config.py                          ← Central configuration (paths, hyper-params)
├── train.py                           ← Model training script
├── inference.py                       ← CLI prediction tool
├── model_utils.py                     ← Artifact verification & inspection
├── app.py                             ← Streamlit alternative frontend
│
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd "Rainfall Prediction"

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
# Edit .env if needed (SECRET_KEY, PORT, etc.)
```

### 3. Run the application (Windows Quick Start)

Double-click `run.bat` in the project root, or execute it in your terminal:
```bash
run.bat
```
This launcher is smart:
- **Activates Virtual Environment:** Automatically detects and loads `venv` or `.venv` if they exist.
- **Checks Dependencies:** Runs a quick import check to verify if all required modules are installed. If any dependencies are missing, it runs `pip install -r requirements.txt`; otherwise, it skips package installation.
- **Checks Model Artifacts:** Verifies if the trained model files exist under `artifacts/models/`. If they exist, it skips training; otherwise, it runs `python train.py` to train the models from the dataset.
- **Interface Selection Menu:** Prompts you to run either the **Flask Web Application**, **Streamlit Web Dashboard**, or the **CLI Predictor**, with a 10-second default timeout to run the Flask application.

### 4. Manual execution (Standard/Multi-platform)

If you prefer to run steps manually, or are on macOS/Linux:

#### A. Train the Model
```bash
python train.py
# Saves to: artifacts/models/rainfall_rf_model.joblib and artifacts/models/model_features.joblib
```

#### B. Launch Flask App
```bash
python src/frontend/app.py
# Open http://127.0.0.1:5000
```

#### C. Launch Streamlit App
```bash
streamlit run app.py
# Open http://localhost:8501
```

#### D. CLI Predictions
```bash
python inference.py --district Lucknow --date 2025-07-15
```

---

## 🧠 Model Details

| Attribute        | Value                              |
|------------------|------------------------------------|
| Algorithm        | Random Forest Regressor            |
| n_estimators     | 100                                |
| Target variable  | PRECTOTCORR (mm/day)               |
| Train/test split | 80 / 20                            |
| MAE (test)       | ≈ 1.26 mm                          |
| R² (test)        | ≈ 0.77                             |
| Features used    | 13 raw + 6 engineered + 74 dummies |

### Feature Engineering

| Feature            | Formula                       |
|--------------------|-------------------------------|
| MONTH              | extracted from DATE           |
| DAY_OF_YEAR        | extracted from DATE           |
| SEASON             | `(MONTH % 12 + 3) // 3`      |
| TEMP_RANGE         | `T2M_MAX − T2M_MIN`           |
| HUMIDITY_DEW_DIFF  | `RH2M − T2MDEW`               |
| WIND_INTENSITY     | `WS50M × WSC`                 |
| DISTRICT_*         | one-hot encoding (74 columns) |

### Rainfall Classification

| Range      | Category | Label                       |
|------------|----------|-----------------------------|
| > 10 mm    | heavy    | Heavy Rainfall Expected     |
| 2 – 10 mm  | moderate | Light to Moderate Showers   |
| ≤ 2 mm     | clear    | Clear / Negligible Rain     |

---

## 🌐 Flask API Reference

| Endpoint         | Method | Description                        |
|------------------|--------|------------------------------------|
| `/`              | GET    | Prediction form                    |
| `/predict`       | POST   | Run prediction → result page       |
| `/history`       | GET    | Last 50 predictions                |
| `/api/districts` | GET    | JSON list of all districts         |
| `/health`        | GET    | Model readiness health-check       |

---

## 🗄️ Database Schema

```sql
CREATE TABLE predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,     -- ISO-8601 UTC
    district      TEXT NOT NULL,
    forecast_date TEXT NOT NULL,     -- YYYY-MM-DD
    season        TEXT NOT NULL,
    prediction_mm REAL NOT NULL,
    category      TEXT NOT NULL,     -- clear | moderate | heavy
    model_version TEXT NOT NULL,
    latitude      REAL,
    longitude     REAL
);
```

---

## 🛠️ Development

```bash
# Verify model artifacts are loadable
python model_utils.py

# Re-train from scratch
python train.py --data UP_rainfall_dataset.csv --output-dir artifacts/models/

# Evaluate current model on dataset
python -c "
from model_utils import evaluate_on_csv
from pathlib import Path
print(evaluate_on_csv(Path('UP_rainfall_dataset.csv')))
"
```

---

## 🚀 Deployment

### Streamlit Cloud
1. Push the repository to GitHub.
2. Create a new app on [share.streamlit.io](https://share.streamlit.io).
3. Set the main file to `app.py`.
4. Add `MODEL_VERSION` as a secret if needed.
5. **Important:** commit your `artifacts/models/` directory (or use `train.py` in a startup script).

### Render (Flask)
1. Create a **Web Service** and connect your GitHub repo.
2. Build command: `pip install -r requirements.txt && python train.py`
3. Start command: `python src/frontend/app.py`
4. Set environment variables (`SECRET_KEY`, `PORT=10000`, `FLASK_DEBUG=false`).

### Local production
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "src.frontend.app:app"
```

---

## 📊 Data Source

NASA POWER (Prediction Of Worldwide Energy Resources) daily climate data for all 75 districts of Uttar Pradesh, India, spanning **2005 – 2025**.

Key variables: `RH2M`, `T2MDEW`, `QV2M`, `PS`, `WS50M`, `T2MWET`, `WD50M`, `T2M_MAX`, `T2M_MIN`, `ALLSKY_SFC_UV_INDEX`, `TS`, `PSC`, `WSC`, `LATITUDE`, `LONGITUDE`

Target: `PRECTOTCORR` – bias-corrected total precipitation (mm/day)

---

## 📄 License

MIT License. See `LICENSE` for details.
