# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
from datetime import datetime
import config


# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UP Rainfall Predictor",
    page_icon="🌧️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  –  glassmorphism dark theme
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root & Background ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 40%, #24243e 100%);
        min-height: 100vh;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header {visibility: hidden;}

    /* ── Animated rain drops (pure CSS) ── */
    @keyframes drop {
        0%   { transform: translateY(-100vh); opacity: 0.8; }
        100% { transform: translateY(110vh);  opacity: 0.2; }
    }
    .rain-container {
        position: fixed; top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none; z-index: 0;
        overflow: hidden;
    }
    .rain-drop {
        position: absolute; top: 0;
        width: 1px; border-radius: 50px;
        background: linear-gradient(transparent, rgba(120,200,255,0.6));
        animation: drop linear infinite;
    }

    /* ── Main container ── */
    .block-container {
        max-width: 740px !important;
        margin: auto !important;
        padding: 1rem 2rem 4rem !important;
        position: relative; z-index: 1;
    }

    /* ── Hero Header ── */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
        animation: fadeSlideIn 0.8s ease both;
    }
    .hero-icon {
        font-size: 4.5rem;
        display: block;
        animation: bounce 2.5s ease-in-out infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50%       { transform: translateY(-12px); }
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #64b3f4, #c2e9fb, #a1c4fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.4rem 0 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(200, 220, 255, 0.65);
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
    }
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Section divider label ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: rgba(160,200,255,0.7);
        margin: 0.5rem 0 0.8rem;
        padding-left: 2px;
    }

    /* ── Widget label overrides ── */
    .stDateInput label,
    .stSelectbox label {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: rgba(210, 230, 255, 0.9) !important;
        letter-spacing: 0.3px !important;
    }

    /* ── Date input box ── */
    .stDateInput input {
        background: rgba(255,255,255,0.09) !important;
        border: 1.5px solid rgba(100,180,255,0.35) !important;
        border-radius: 12px !important;
        color: #e8f0ff !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        caret-color: #64b3f4 !important;
    }
    .stDateInput input:focus {
        border-color: rgba(100,180,255,0.75) !important;
        box-shadow: 0 0 0 3px rgba(100,180,255,0.18) !important;
        outline: none !important;
    }

    /* ── Selectbox: the visible trigger button ── */
    [data-baseweb="select"] > div:first-child {
        background: rgba(255,255,255,0.09) !important;
        border: 1.5px solid rgba(100,180,255,0.35) !important;
        border-radius: 12px !important;
        color: #e8f0ff !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        min-height: 44px !important;
    }
    [data-baseweb="select"] > div:first-child:hover {
        border-color: rgba(100,180,255,0.65) !important;
    }
    /* The text inside the select trigger */
    [data-baseweb="select"] span,
    [data-baseweb="select"] div[class*="ValueContainer"] *,
    [data-baseweb="select"] div[class*="singleValue"],
    [data-baseweb="select"] input {
        color: #e8f0ff !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    /* Dropdown arrow ── */
    [data-baseweb="select"] svg {
        fill: rgba(150,200,255,0.8) !important;
    }
    /* Dropdown list popup */
    [data-baseweb="menu"],
    [data-baseweb="popover"] {
        background: #16163a !important;
        border: 1px solid rgba(100,180,255,0.25) !important;
        border-radius: 12px !important;
    }
    [data-baseweb="option"] {
        background: transparent !important;
        color: rgba(200,220,255,0.85) !important;
        font-size: 0.95rem !important;
    }
    [data-baseweb="option"]:hover,
    [data-baseweb="option"][aria-selected="true"] {
        background: rgba(100,180,255,0.18) !important;
        color: #ffffff !important;
    }

    /* ── Predict button ── */
    .stButton > button {
        width: 100%;
        padding: 0.9rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        border: none;
        border-radius: 14px;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #06063a;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s, filter 0.2s;
        box-shadow: 0 6px 24px rgba(79, 172, 254, 0.4);
        margin-top: 0.75rem;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(79, 172, 254, 0.55);
        filter: brightness(1.08);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ── Result card ── */
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: fadeSlideIn 0.6s ease both;
        margin-top: 1.2rem;
    }
    .result-heavy {
        background: linear-gradient(135deg, rgba(20,60,140,0.6), rgba(0,90,200,0.4));
        border: 1px solid rgba(100,180,255,0.45);
        box-shadow: 0 8px 40px rgba(0,120,255,0.3);
    }
    .result-moderate {
        background: linear-gradient(135deg, rgba(40,90,20,0.6), rgba(80,160,30,0.35));
        border: 1px solid rgba(160,220,80,0.45);
        box-shadow: 0 8px 40px rgba(100,180,40,0.3);
    }
    .result-clear {
        background: linear-gradient(135deg, rgba(180,110,10,0.5), rgba(240,170,20,0.3));
        border: 1px solid rgba(255,200,80,0.45);
        box-shadow: 0 8px 40px rgba(240,170,20,0.3);
    }
    .result-icon {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: bounce 2.5s ease-in-out infinite;
    }
    .result-label {
        font-size: 1rem;
        font-weight: 500;
        color: rgba(220,240,255,0.75);
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .result-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1px;
        text-shadow: 0 2px 20px rgba(100,200,255,0.5);
        margin: 0.3rem 0;
    }
    .result-desc {
        font-size: 0.9rem;
        color: rgba(200,220,255,0.6);
        margin-top: 0.4rem;
    }
    .chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
        justify-content: center;
    }
    .chip {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 50px;
        padding: 0.3rem 0.9rem;
        font-size: 0.78rem;
        color: rgba(200,220,255,0.8);
        font-weight: 500;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(100,150,255,0.15) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: rgba(140,165,210,0.4);
        font-size: 0.75rem;
        margin-top: 2.5rem;
        letter-spacing: 0.5px;
    }
    </style>

    <!-- Animated rain drops -->
    <div class="rain-container" id="rain"></div>
    <script>
    (function(){
        var c = document.getElementById('rain');
        if (!c) return;
        for (var i = 0; i < 30; i++) {
            var d = document.createElement('div');
            d.className = 'rain-drop';
            d.style.left   = Math.random()*100 + '%';
            d.style.height = (Math.random()*60 + 40) + 'px';
            d.style.animationDuration  = (Math.random()*2 + 1.5) + 's';
            d.style.animationDelay     = (Math.random()*3) + 's';
            d.style.opacity = Math.random()*0.5 + 0.2;
            c.appendChild(d);
        }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# DISTRICT DATA  (lat / lon / seasonal defaults)
# ─────────────────────────────────────────────
SEASON_DEFAULTS = {
    "monsoon":      {"RH2M":85,"T2MDEW":22,"QV2M":15,"PS":99, "WS50M":5,"T2MWET":26,"WD50M":200,"T2M_MAX":33,"T2M_MIN":26,"ALLSKY_SFC_UV_INDEX":4,"TS":32,"PSC":90,"WSC":5.5},
    "post-monsoon": {"RH2M":65,"T2MDEW":14,"QV2M":9, "PS":100,"WS50M":3,"T2MWET":18,"WD50M":150,"T2M_MAX":30,"T2M_MIN":18,"ALLSKY_SFC_UV_INDEX":5,"TS":28,"PSC":91,"WSC":3.5},
    "winter":       {"RH2M":70,"T2MDEW":8, "QV2M":6, "PS":101,"WS50M":3,"T2MWET":10,"WD50M":120,"T2M_MAX":22,"T2M_MIN":10,"ALLSKY_SFC_UV_INDEX":3,"TS":18,"PSC":92,"WSC":3},
    "pre-monsoon":  {"RH2M":40,"T2MDEW":14,"QV2M":8, "PS":99, "WS50M":5,"T2MWET":22,"WD50M":250,"T2M_MAX":40,"T2M_MIN":26,"ALLSKY_SFC_UV_INDEX":8,"TS":40,"PSC":89,"WSC":5.5},
}

def get_season(month: int) -> str:
    if month in (6, 7, 8, 9):  return "monsoon"
    if month in (10, 11):       return "post-monsoon"
    if month in (12, 1, 2):     return "winter"
    return "pre-monsoon"

DISTRICT_COORDS = {
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

_DISTRICT_COLUMN_MAP = {
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

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    model    = joblib.load(config.MODEL_PATH)
    features = joblib.load(config.FEATURES_PATH)
    return model, features

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <span class="hero-icon">🌧️</span>
        <h1>UP Rainfall Predictor</h1>
        <p>AI-powered daily rainfall forecast for Uttar Pradesh districts</p>
        <p style="font-size: 0.85rem; color: rgba(200, 220, 255, 0.4); margin-top: 0.5rem; margin-bottom: 0;">
            Trained on NASA POWER climate data from 2005 to 2025
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Loading AI model …"):
    try:
        model, expected_features = load_assets()
    except FileNotFoundError:
        st.error(
            f"⚠️ Model files not found! Make sure model files exist at {config.MODELS_DIR}"
        )
        st.stop()

# ─────────────────────────────────────────────
# INPUT WIDGETS  (no wrapping div – Streamlit
# renders widgets outside custom HTML tags)
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">📍 Forecast Details</p>', unsafe_allow_html=True)

col_date, col_dist = st.columns(2)
with col_date:
    select_date = st.date_input(
        "📅  Forecast Date",
        value=datetime.now().date(),
        min_value=datetime(2000, 1, 1).date(),
        max_value=datetime(2030, 12, 31).date(),
    )
with col_dist:
    districts_sorted = sorted(DISTRICT_COORDS.keys())
    selected_district = st.selectbox(
        "🗺️  Select District",
        options=districts_sorted,
        index=districts_sorted.index("Lucknow"),
    )

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔮  Predict Rainfall", use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_btn:
    season   = get_season(select_date.month)
    defaults = SEASON_DEFAULTS[season]
    lat, lon = DISTRICT_COORDS.get(selected_district, (26.85, 80.95))

    raw_input = {
        "YEAR":                select_date.year,
        "MO":                  select_date.month,
        "DY":                  select_date.day,
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
        "DISTRICT":            selected_district,
    }

    df_input = pd.DataFrame([raw_input])

    # Feature engineering (mirrors training notebook)
    df_input["DATE"]              = pd.to_datetime(
        df_input[["YEAR","MO","DY"]].rename(columns={"YEAR":"year","MO":"month","DY":"day"})
    )
    df_input["MONTH"]             = df_input["DATE"].dt.month
    df_input["DAY_OF_YEAR"]       = df_input["DATE"].dt.dayofyear
    df_input["SEASON"]            = (df_input["MONTH"] % 12 + 3) // 3
    df_input["TEMP_RANGE"]        = df_input["T2M_MAX"] - df_input["T2M_MIN"]
    df_input["HUMIDITY_DEW_DIFF"] = df_input["RH2M"]    - df_input["T2MDEW"]
    df_input["WIND_INTENSITY"]    = df_input["WS50M"]   * df_input["WSC"]
    df_input = df_input.drop(columns=["DATE"])

    district_col = _DISTRICT_COLUMN_MAP.get(selected_district)
    df_input = df_input.drop(columns=["DISTRICT"])

    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
    if district_col and district_col in expected_features:
        df_input[district_col] = 1

    df_input = df_input[expected_features]

    with st.spinner("Running AI forecast …"):
        prediction = model.predict(df_input)[0]
        prediction = max(0.0, prediction)

    # ── Result ──
    date_str     = select_date.strftime("%d %B %Y")
    season_label = season.replace("-", " ").title()

    if prediction > 10.0:
        css_class   = "result-heavy"
        icon        = "⛈️"
        headline    = "Heavy Rainfall Expected"
        description = "Carry an umbrella and stay alert for flooding alerts."
    elif prediction > 2.0:
        css_class   = "result-moderate"
        icon        = "🌦️"
        headline    = "Light to Moderate Showers"
        description = "A light jacket or umbrella would be handy today."
    else:
        css_class   = "result-clear"
        icon        = "☀️"
        headline    = "Clear / Negligible Rain"
        description = "Enjoy the day — minimal chance of rain."

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <span class="result-icon">{icon}</span>
            <div class="result-label">{headline}</div>
            <div class="result-value">{prediction:.2f}&nbsp;<span style="font-size:1.3rem;font-weight:400;opacity:.8">mm</span></div>
            <div class="result-desc">{description}</div>
            <div class="chips-row">
                <span class="chip">📅 {date_str}</span>
                <span class="chip">📍 {selected_district}</span>
                <span class="chip">🌿 {season_label}</span>
                <span class="chip">🌡️ {defaults['T2M_MAX']}°C / {defaults['T2M_MIN']}°C</span>
                <span class="chip">💧 {defaults['RH2M']}% humidity</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div class="footer">Powered by Random Forest AI · UP Climate Data · Built with Streamlit</div>',
    unsafe_allow_html=True,
)