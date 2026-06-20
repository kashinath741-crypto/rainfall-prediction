@echo off
setlocal enabledelayedexpansion

:: Set console layout and title
title UP Rainfall Predictor Launcher
color 0B

echo ============================================================
echo            🌧️  UP RAINFALL PREDICTOR STARTUP  🌧️
echo ============================================================
echo.

:: 1. Check Python Launcher
where py >nul 2>&1
if !errorlevel! equ 0 (
    set PY_CMD=py
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set PY_CMD=python
    ) else (
        echo [ERROR] Python was not found on your system path.
        echo Please install Python 3.9 or higher and check Add Python to PATH.
        pause
        exit /b 1
    )
)

:: 2. Check Virtual Environment
if exist venv\Scripts\activate.bat (
    echo [INFO] Found virtual environment venv. Activating...
    call venv\Scripts\activate.bat
    set PY_CMD=python
)
if exist .venv\Scripts\activate.bat (
    echo [INFO] Found virtual environment .venv. Activating...
    call .venv\Scripts\activate.bat
    set PY_CMD=python
)

:: 3. Check dependencies
echo [INFO] Verifying installed dependencies...
%PY_CMD% -c "import flask, pandas, numpy, sklearn, joblib, dotenv, streamlit" >nul 2>&1
if !errorlevel! neq 0 (
    echo [INFO] Installing required dependencies...
    %PY_CMD% -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install dependencies. Please check your internet connection.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Requirements already satisfied. Skipping installation.
)

:: 4. Check model files
if exist "artifacts\models\rainfall_rf_model.joblib" (
    if exist "artifacts\models\model_features.joblib" (
        echo [INFO] Model files found in artifacts\models. Skipping model training.
        goto :select_app
    )
)

echo [INFO] Trained model artifacts not found. Starting training...
echo        This may take a moment to train the Random Forest Regressor on the NASA POWER dataset.
%PY_CMD% train.py
if !errorlevel! neq 0 (
    echo [ERROR] Model training failed. Make sure UP_rainfall_dataset.csv exists in the root directory.
    pause
    exit /b 1
)

:select_app
echo.
echo ============================================================
echo Select which interface to run:
echo [1] Flask Web Application - Default, Premium Glassmorphism UI
echo [2] Streamlit Web Application - Alternative Dashboard UI
echo [3] CLI Predictor - Quick Command Line Interface
echo [4] Exit
echo ============================================================
echo.

:: Choice utility is built-in. Sets ERRORLEVEL: 1 for option 1, 2 for option 2, etc.
choice /c 1234 /t 10 /d 1 /m "Select option (1-4)"
set OPTION=%errorlevel%

if %OPTION% equ 1 (
    echo.
    echo [INFO] Launching Flask Web Application...
    echo        Access the application at http://127.0.0.1:5000
    echo        Press Ctrl+C in this window to stop the server.
    echo.
    %PY_CMD% src\frontend\app.py
    echo.
    echo [INFO] Application has stopped.
    pause
)
if %OPTION% equ 2 (
    echo.
    echo [INFO] Launching Streamlit Web Application...
    echo.
    %PY_CMD% -m streamlit run app.py
    echo.
    echo [INFO] Application has stopped.
    pause
)
if %OPTION% equ 3 (
    echo.
    echo [INFO] Launching CLI Predictor...
    set /p DISTRICT="Enter District Name (e.g., Lucknow, Varanasi, Prayagraj): "
    set /p FORDATE="Enter Forecast Date (YYYY-MM-DD): "
    echo.
    %PY_CMD% inference.py --district "!DISTRICT!" --date "!FORDATE!"
    echo.
    pause
    goto :select_app
)
if %OPTION% equ 4 (
    echo Exiting. Goodbye!
    pause
)
