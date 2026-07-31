@echo off
REM ============================================
REM  Student Risk Prediction - Windows setup
REM  Creates a virtual environment and installs
REM  all dependencies. Run once after cloning.
REM ============================================
setlocal
cd /d "%~dp0"

if not exist venv (
    echo [1/3] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/3] Virtual environment already exists.
)

echo [2/3] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

echo [3/3] Installing dependencies from requirements.txt...
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Setup complete! Launch the app with:
echo     run.bat
echo or manually:
echo     venv\Scripts\streamlit.exe run app.py
echo.
pause
