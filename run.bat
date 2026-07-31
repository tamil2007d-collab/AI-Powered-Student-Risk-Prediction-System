@echo off
REM ============================================
REM  Launch the Student Risk Prediction app
REM ============================================
cd /d "%~dp0"

if not exist venv (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

venv\Scripts\streamlit.exe run app.py
