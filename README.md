# 🎓 AI-Powered Student Risk Prediction System

An end-to-end machine learning application that predicts **at-risk students** from academic and engagement features, explains every prediction with **SHAP**, and suggests **targeted interventions** — all inside a modern, dark-themed Streamlit dashboard.

> Hackathon-ready · Explainable AI · Production-structured

---

## ✨ Features

- 📊 **Interactive Dashboard** — KPI cards, risk distribution, attendance & marks charts, correlation heatmap, confusion matrix
- 🔮 **Single-Student Prediction** — risk level, confidence %, SHAP waterfall + force plot, tailored interventions
- 🧬 **Explainable AI (SHAP)** — global summary plot and per-prediction reasoning
- 🚦 **Intervention Engine** — High / Medium / Low risk recommendation tiers
- 📁 **CSV Upload & Export** — load your own data, download full predictions
- 🧠 **Model Comparison** — Random Forest vs Decision Tree vs Logistic Regression, best model auto-selected
- 🕘 **Prediction History**, 🔍 **Search Student**, 🎚️ **Risk Filtering**, 🔄 **One-click Retraining**
- 🛡️ **Ethics Page** — privacy, bias mitigation, transparency, human-in-the-loop design

---

## 🏗️ Project Structure

```
student-risk-prediction/
│
├── app.py                 # Streamlit dashboard (main entry point)
├── train.py               # Training + model comparison + saving
├── predict.py             # Inference + SHAP explanations
├── utils.py               # Data cleaning, preprocessing, recommendations
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── data/
│     students.csv         # Sample dataset (1,000 students)
│
├── models/
│     model.pkl            # Trained best model + metadata
│     metrics.json         # Model comparison metrics
│
├── notebooks/
│     EDA.ipynb            # Exploratory Data Analysis notebook
│
├── assets/
│     logo.png             # App logo
│
└── .streamlit/
      config.toml          # Dark theme configuration
```

---

## 🛠️ Installation

> **No need to copy/paste any `venv` folder — that's machine-specific.**
> The repo ships with `requirements.txt` plus one-click setup scripts that rebuild
> the environment for you in seconds.

```bash
# 1. Clone the repository
git clone https://github.com/tamil2007d-collab/AI-Powered-Student-Risk-Prediction-System.git
cd AI-Powered-Student-Risk-Prediction-System

# 2. Automatic setup (recommended)
setup.bat              # Windows  (double-click, or run in a terminal)
bash setup.sh          # macOS / Linux
```

### Manual setup (optional)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# 1. Train the model (creates models/model.pkl)
python train.py

# 2. Launch the dashboard
streamlit run app.py          # or double-click run.bat on Windows
```

The app opens at `http://localhost:8501`. If `model.pkl` is missing, the app auto-trains on first load.

---

## 🗂️ Dataset Schema

| Column              | Type      | Description                          |
|---------------------|-----------|--------------------------------------|
| Student_ID          | str       | Unique identifier                    |
| Attendance          | 0–100     | % of classes attended                |
| Assignment_Score    | 0–100     | Average assignment marks             |
| Quiz_Score          | 0–100     | Average quiz marks                   |
| Midterm             | 0–100     | Midterm exam score                   |
| Final               | 0–100     | Final exam score                     |
| GPA                 | 0–4       | Grade point average                  |
| LMS_Logins          | int       | Learning-platform logins             |
| Time_Spent          | hours     | Time spent on the LMS                |
| Discussion_Posts    | int       | Forum posts contributed              |
| Late_Submissions    | int       | Number of late submissions           |
| Teacher_Feedback    | cat       | Poor / Fair / Good / Excellent       |
| Course_Completion   | 0–100     | % of the course completed            |
| Previous_Result     | 0/1       | Passed previous year (1 = pass)      |
| **Risk**            | **0/1**   | **Target: 0 = Safe, 1 = At Risk**    |

---

## 🔄 Project Workflow

```
1. DATA CLEANING      → drop duplicates · fill missing · cap outliers (IQR) · encode categories
2. EDA                → distributions · correlation heatmap · risk & GPA analysis
3. MODEL TRAINING     → Random Forest vs Decision Tree vs Logistic Regression
4. MODEL EVALUATION   → accuracy · precision · recall · F1 · ROC-AUC · confusion matrix
5. EXPLAINABLE AI     → SHAP feature importance · waterfall · force · summary plots
6. INTERVENTION       → High / Medium / Low risk recommendations
7. STREAMLIT UI       → dashboard · prediction · insights · ethics
```

---

## 🧠 Model Selection

| Model               | Why                                     |
|---------------------|-----------------------------------------|
| Random Forest       | Robust, handles non-linearity, native feature importance (usually best) |
| Decision Tree       | Interpretable baseline                   |
| Logistic Regression | Simple linear benchmark                  |

The **best model by ROC-AUC** is saved to `models/model.pkl` with all preprocessing metadata, so the dashboard and API behave consistently.

---

## 📸 Screenshots

> *Add screenshots here after running the app*

| Dashboard | Single Prediction | SHAP Explanation |
|-----------|-------------------|------------------|
| `assets/dashboard.png` | `assets/predict.png` | `assets/shap.png` |

---

## 🧪 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python 3.12
- **ML:** scikit-learn
- **Explainability:** SHAP
- **Visualization:** Plotly, Matplotlib
- **Data:** pandas, numpy

---

## 🚧 Future Improvements

- [ ] Deep-learning tabular models (XGBoost, LightGBM, CatBoost)
- [ ] Time-series risk tracking across terms
- [ ] Multi-class risk scoring instead of binary thresholding
- [ ] Fairness auditing (demographic parity / equalised odds)
- [ ] REST API wrapper (FastAPI) for integration with SIS
- [ ] Database persistence (SQLite / Postgres) for prediction history
- [ ] Email / SMS alerts to mentors when a student crosses a risk threshold
- [ ] Docker packaging for one-command deployment

---

## ⚠️ Ethical Note

This system is a **decision-support tool only**. Predictions are probabilities that must be combined with professional judgement. Never use automated risk scores to discipline students — use them to route support and care.

---

## 📜 License

Free to use for academic and hackathon purposes.
