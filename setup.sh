#!/usr/bin/env bash
# ============================================
#  Student Risk Prediction - macOS/Linux setup
#  Creates a virtual environment and installs
#  all dependencies. Run once after cloning:
#      bash setup.sh
# ============================================
set -e
cd "$(dirname "$0")"

echo "[1/3] Creating virtual environment..."
python3 -m venv venv

echo "[2/3] Upgrading pip..."
venv/bin/pip install --upgrade pip

echo "[3/3] Installing dependencies from requirements.txt..."
venv/bin/pip install -r requirements.txt

echo
echo "Setup complete! Launch the app with:"
echo "    venv/bin/streamlit run app.py"
