"""app.py — AI-Powered Student Risk Prediction System (Streamlit front-end).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import predict
import train
import utils

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "students.csv"
MODEL_FILE = BASE_DIR / "models" / "model.pkl"
LOGO_FILE = BASE_DIR / "assets" / "logo.png"

st.set_page_config(
    page_title="AI Student Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CAT_COLORS = {"High Risk": "#ff4d6d", "Medium Risk": "#ffb020", "Low Risk": "#2dd4bf"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

.stApp {
    font-family: 'Inter', -apple-system, sans-serif;
    background:
        radial-gradient(1200px 600px at 85% -5%, rgba(76,110,245,.16), transparent 55%),
        radial-gradient(1000px 500px at -10% 10%, rgba(123,47,247,.12), transparent 55%),
        #0b1020;
    color: #e6edf7;
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; }
h1, h2, h3 { letter-spacing: -0.02em; }

.grad-header {
    background: linear-gradient(135deg, #1f6feb 0%, #6c4dff 55%, #b32fff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-weight: 800;
}

.kpi-card {
    background: linear-gradient(160deg, rgba(30,41,82,.75), rgba(18,24,52,.85));
    border: 1px solid rgba(120,150,255,.18);
    border-radius: 16px; padding: 18px 20px; height: 100%;
    box-shadow: 0 8px 28px rgba(0,0,0,.35);
}
.kpi-icon { font-size: 26px; }
.kpi-value { font-size: 30px; font-weight: 800; line-height: 1.1; }
.kpi-label { font-size: 13px; color: #9fb2d9; margin-top: 2px; }
.kpi-sub { font-size: 12px; margin-top: 6px; color: #6f84b3; }

.badge {
    display: inline-block; padding: 8px 22px; border-radius: 999px;
    font-weight: 700; font-size: 17px;
}
.badge-high   { background: rgba(255,77,109,.16); color: #ff7b94; border: 1px solid rgba(255,77,109,.45); }
.badge-medium { background: rgba(255,176,32,.14); color: #ffc85c; border: 1px solid rgba(255,176,32,.45); }
.badge-low    { background: rgba(45,212,191,.14); color: #5eead4; border: 1px solid rgba(45,212,191,.45); }

.rec-card {
    background: linear-gradient(160deg, rgba(31,111,235,.12), rgba(28,20,60,.6));
    border: 1px solid rgba(120,150,255,.22);
    border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;
}
.section-title { font-size: 18px; font-weight: 700; margin: 20px 0 10px; color: #cfe0ff; }

.pulse { animation: pulse 2.2s infinite; }
@keyframes pulse { 0%{opacity:1;} 50%{opacity:.55;} 100%{opacity:1;} }

.stButton>button {
    background: linear-gradient(135deg, #1f6feb, #6c4dff);
    border: none; border-radius: 10px; color: #fff; font-weight: 600;
    padding: .5rem 1rem;
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(76,110,245,.45); }

div[data-testid="stDataFrame"] {
    border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(120,150,255,.15);
}

.ethical-card {
    background: linear-gradient(160deg, rgba(30,41,82,.6), rgba(18,24,52,.75));
    border: 1px solid rgba(120,150,255,.18);
    border-radius: 16px; padding: 20px; margin-bottom: 14px; height: 100%;
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def init_state():
    st.session_state.setdefault("model_version", 0)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("raw_data", None)
    st.session_state.setdefault("df_pred", None)
    st.session_state.setdefault("nav", "📊 Dashboard")


@st.cache_data(show_spinner=False)
def load_sample():
    utils.ensure_sample_data(DATA_FILE)
    return utils.load_data(DATA_FILE)


@st.cache_resource(show_spinner=False)
def load_artifact(version: int):
    return predict.load_model(MODEL_FILE)


def get_raw() -> pd.DataFrame:
    return st.session_state.raw_data if st.session_state.raw_data is not None else load_sample()


def get_artifact():
    art = load_artifact(st.session_state.model_version)
    if art is None:
        st.warning("No trained model found — training a fresh model now…")
        art, _, _ = train.train(save=True)
        st.session_state.model_version += 1
        load_artifact.clear()
        st.success("Model ready!")
    return art


def get_predictions() -> pd.DataFrame:
    if st.session_state.df_pred is None:
        st.session_state.df_pred = predict.predict_batch(get_artifact(), get_raw())
    return st.session_state.df_pred


def lyt(fig, h: int = 420):
    fig.update_layout(
        height=h,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,28,60,.55)",
        font=dict(color="#e6edf7"),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def kpi_card(icon: str, value: str, label: str, sub: str, accent: str = "#ffffff") -> str:
    return (
        f"<div class='kpi-card'>"
        f"<div class='kpi-icon'>{icon}</div>"
        f"<div class='kpi-value' style='color:{accent}'>{value}</div>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-sub'>{sub}</div>"
        f"</div>"
    )


def render_kpis(df: pd.DataFrame):
    total = len(df)
    high = int((df["Risk_Category"] == "High Risk").sum())
    med = int((df["Risk_Category"] == "Medium Risk").sum())
    low = int((df["Risk_Category"] == "Low Risk").sum())
    rate = high / total * 100 if total else 0.0
    avg_gpa = df["GPA"].mean() if "GPA" in df.columns else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("👥", f"{total:,}", "Total Students", f"Avg GPA {avg_gpa:.2f} / 4.0"), unsafe_allow_html=True)
    c2.markdown(kpi_card("🚨", f"{high}", "High Risk", f"{rate:.1f}% of cohort", accent="#ff7b94"), unsafe_allow_html=True)
    c3.markdown(kpi_card("⚠️", f"{med}", "Medium Risk", "Needs attention", accent="#ffc85c"), unsafe_allow_html=True)
    c4.markdown(kpi_card("✅", f"{low}", "Low Risk", "On track", accent="#5eead4"), unsafe_allow_html=True)


def attendance_hist(df):
    fig = px.histogram(
        df,
        x="Attendance",
        color="Risk_Category",
        nbins=25,
        title="Attendance Distribution by Risk",
        color_discrete_map=CAT_COLORS,
    )
    return lyt(fig)


def risk_pie(df):
    counts = df["Risk_Category"].value_counts()
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        hole=0.55,
        title="Risk Distribution",
        color=counts.index,
        color_discrete_map=CAT_COLORS,
    )
    fig.update_traces(textinfo="label+percent")
    return lyt(fig, 430)


def marks_dist(df):
    fig = go.Figure()
    for col, color in [("Midterm", "#4da3ff"), ("Final", "#b32fff")]:
        fig.add_trace(
            go.Violin(
                y=df[col], name=col, box_visible=True, meanline_visible=True, line_color=color
            )
        )
    fig.update_layout(title="Marks Distribution (Midterm vs Final)")
    return lyt(fig)


def feature_importance(fi: dict):
    s = pd.Series(fi).sort_values()
    fig = px.bar(
        s,
        orientation="h",
        title="Feature Importance (Best Model)",
        color=s.values,
        color_continuous_scale=["#2dd4bf", "#6c4dff", "#ffb020"],
    )
    fig.update_layout(coloraxis_showscale=False)
    return lyt(fig)


def gpa_by_risk(df):
    g = df.groupby("Risk_Category")["GPA"].mean().reindex(["High Risk", "Medium Risk", "Low Risk"])
    fig = px.bar(
        x=g.index, y=g.values, color=g.index, color_discrete_map=CAT_COLORS,
        title="Average GPA by Risk Level", labels={"x": "", "y": "Avg GPA"},
    )
    return lyt(fig)


def completion_by_risk(df):
    g = df.groupby("Risk_Category")["Course_Completion"].mean().reindex(["High Risk", "Medium Risk", "Low Risk"])
    fig = px.bar(
        x=g.index, y=g.values, color=g.index, color_discrete_map=CAT_COLORS,
        title="Avg Course Completion by Risk Level", labels={"x": "", "y": "Avg %"},
    )
    return lyt(fig)


def feedback_by_risk(df):
    tab = df.groupby(["Teacher_Feedback", "Risk_Category"]).size().reset_index(name="count")
    fig = px.bar(
        tab, x="Teacher_Feedback", y="count", color="Risk_Category",
        color_discrete_map=CAT_COLORS, barmode="stack",
        title="Students by Teacher Feedback & Risk",
    )
    return lyt(fig)


def conf_matrix(cm, model_name):
    fig = px.imshow(
        np.array(cm), text_auto=True, color_continuous_scale="Blues",
        x=["Predicted: Safe", "Predicted: At Risk"],
        y=["Actual: Safe", "Actual: At Risk"],
        title=f"Confusion Matrix — {model_name}",
    )
    fig.update_layout(coloraxis_showscale=False)
    return lyt(fig, 400)


def corr_heat(df):
    cols = [c for c in utils.NUMERICAL_FEATURES if c in df.columns]
    if utils.TARGET in df.columns:
        cols = cols + [utils.TARGET]
    d = df[cols].copy()
    if utils.TARGET in d.columns:
        d[utils.TARGET] = d[utils.TARGET].astype(int)
    fig = px.imshow(
        d.corr(), text_auto=".2f", color_continuous_scale="RdYlBu_r",
        aspect="auto", title="Feature Correlation Heatmap", zmin=-1, zmax=1,
    )
    return lyt(fig, 720)


def risk_color(v):
    if v == "High Risk":
        return "background-color: rgba(255,77,109,.18)"
    if v == "Medium Risk":
        return "background-color: rgba(255,176,32,.16)"
    return "background-color: rgba(45,212,191,.14)"


def show_pred_table(df: pd.DataFrame):
    cols = [c for c in ["Student_ID", "Attendance", "GPA", "Final", "Course_Completion", "Risk_Probability", "Risk_Category"] if c in df.columns]
    view = df[cols].copy()

    f1, f2 = st.columns([1, 2])
    with f1:
        cats = ["High Risk", "Medium Risk", "Low Risk"]
        sel = st.multiselect("Filter by risk level", cats, default=cats)
    with f2:
        query = st.text_input("🔍 Search by Student ID", "")

    if sel:
        view = view[view["Risk_Category"].isin(sel)]
    if query:
        view = view[view["Student_ID"].astype(str).str.contains(query, case=False, na=False)]

    styler = view.style.format({"Risk_Probability": "{:.1%}"})
    try:
        styler = styler.map(risk_color, subset=["Risk_Category"])
    except AttributeError:
        styler = styler.applymap(risk_color, subset=["Risk_Category"])

    st.dataframe(styler, width="stretch", height=380)
    st.caption(f"Showing {len(view):,} of {len(df):,} students")


def show_history():
    if st.session_state.history:
        st.markdown("<div class='section-title'>🕘 Prediction History</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch", height=220)
        if st.button("🧹 Clear History"):
            st.session_state.history = []
            rerun()


def show_result(art, features, result):
    cat = result["category"]
    badge = {"High Risk": "badge-high", "Medium Risk": "badge-medium", "Low Risk": "badge-low"}[cat]

    st.markdown("<div class='section-title'>🎯 Prediction Result</div>", unsafe_allow_html=True)
    r1, r2 = st.columns([1, 1])
    with r1:
        st.markdown(f"<span class='badge {badge}'>🎯 {cat}</span>", unsafe_allow_html=True)
    with r2:
        st.metric("Confidence", f"{result['confidence']:.1f}%", delta="Model certainty")
    st.progress(int(result["probability"] * 100), text=f"At-risk probability: {result['probability']:.1%}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-title'>🧠 Why this prediction? (SHAP)</div>", unsafe_allow_html=True)
        try:
            base, values, _ = predict.shap_for_row(art, result["X_row"])
            st.pyplot(predict.waterfall_figure(base, values, result["X_row"].values[0]))
            try:
                html = predict.force_plot_html(base, values, result["X_row"].values[0])
                st.components.v1.html(html, height=200, scrolling=True)
            except Exception:
                st.info("Interactive force plot is unavailable for this model.")
        except Exception as exc:
            st.warning(f"SHAP explanation could not be generated: {exc}")
    with col_r:
        st.markdown("<div class='section-title'>💡 Recommended Interventions</div>", unsafe_allow_html=True)
        for icon, title, desc in result["recommendations"]:
            st.markdown(
                f"<div class='rec-card'><b>{icon} {title}</b>"
                f"<br/><span style='color:#9fb2d9;font-size:13px'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    st.session_state.history.append(
        {
            "Student_ID": features.get("Student_ID", "Manual Entry"),
            "Attendance": features["Attendance"],
            "GPA": features["GPA"],
            "Final": features["Final"],
            "Probability": round(result["probability"], 4),
            "Category": cat,
        }
    )
    show_history()


def page_dashboard():
    art = get_artifact()
    df = get_predictions()
    best = art["best"]
    fi = art["metrics"][best]["feature_importance"]
    cm = art["metrics"][best]["confusion_matrix"]

    st.markdown("<h1 class='grad-header'>📊 Risk Dashboard</h1>", unsafe_allow_html=True)
    st.caption(
        f"Best model: **{best}** · Trained on {art.get('n_samples', '?')} students · "
        f"At-risk rate {art.get('risk_rate', 0):.1%} · {art.get('trained_at', '')[:16]}"
    )

    render_kpis(df)

    st.markdown("<div class='section-title'>📈 Overview Charts</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(attendance_hist(df), width="stretch")
    with c2:
        st.plotly_chart(risk_pie(df), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(marks_dist(df), width="stretch")
    with c4:
        st.plotly_chart(feature_importance(fi), width="stretch")

    st.markdown("<div class='section-title'>🧪 Model Validation</div>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(conf_matrix(cm, best), width="stretch")
    with c6:
        st.plotly_chart(corr_heat(df), width="stretch")

    st.markdown("<div class='section-title'>👨‍🎓 Student Predictions</div>", unsafe_allow_html=True)
    show_pred_table(df)


def page_predict():
    art = get_artifact()
    df = get_predictions()
    p = "pred"

    st.markdown("<h1 class='grad-header'>🔮 Predict a Student</h1>", unsafe_allow_html=True)
    st.caption("Enter a student's details. The model returns risk level, confidence, a SHAP explanation and interventions.")

    ids = [str(x) for x in df["Student_ID"].tolist()]
    top = st.columns([3, 1])
    with top[0]:
        picked = st.selectbox("Load an existing student (auto-fill the form)", [""] + ids, key=f"{p}_pick")
    with top[1]:
        st.write("")
        auto = st.button("🪄 Auto-fill", width="stretch")

    defaults = {
        f"{p}_attendance": 80.0, f"{p}_assignment": 70.0, f"{p}_quiz": 68.0,
        f"{p}_midterm": 65.0, f"{p}_final": 65.0, f"{p}_gpa": 3.0,
        f"{p}_lms": 40, f"{p}_time": 60.0, f"{p}_posts": 15,
        f"{p}_late": 2, f"{p}_completion": 80.0, f"{p}_feedback": "Good", f"{p}_prev": "Pass",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if auto and picked:
        row = df[df["Student_ID"].astype(str) == picked].iloc[0]
        st.session_state[f"{p}_attendance"] = float(row["Attendance"])
        st.session_state[f"{p}_assignment"] = float(row["Assignment_Score"])
        st.session_state[f"{p}_quiz"] = float(row["Quiz_Score"])
        st.session_state[f"{p}_midterm"] = float(row["Midterm"])
        st.session_state[f"{p}_final"] = float(row["Final"])
        st.session_state[f"{p}_gpa"] = float(row["GPA"])
        st.session_state[f"{p}_lms"] = int(row["LMS_Logins"])
        st.session_state[f"{p}_time"] = float(row["Time_Spent"])
        st.session_state[f"{p}_posts"] = int(row["Discussion_Posts"])
        st.session_state[f"{p}_late"] = int(row["Late_Submissions"])
        st.session_state[f"{p}_completion"] = float(row["Course_Completion"])
        st.session_state[f"{p}_feedback"] = str(row["Teacher_Feedback"])
        st.session_state[f"{p}_prev"] = "Pass" if int(row["Previous_Result"]) == 1 else "Fail"
        rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("📋 Attendance (%)", 0.0, 100.0, key=f"{p}_attendance", step=1.0)
        st.number_input("📝 Assignment Score", 0.0, 100.0, key=f"{p}_assignment", step=1.0)
        st.number_input("❓ Quiz Score", 0.0, 100.0, key=f"{p}_quiz", step=1.0)
        st.number_input("📄 Midterm Marks", 0.0, 100.0, key=f"{p}_midterm", step=1.0)
        st.number_input("🎯 Final Marks", 0.0, 100.0, key=f"{p}_final", step=1.0)
    with c2:
        st.number_input("⭐ GPA (0–4)", 0.0, 4.0, key=f"{p}_gpa", step=0.01)
        st.number_input("💻 LMS Logins", 0, 200, key=f"{p}_lms", step=1)
        st.number_input("⏱️ Time Spent (hours)", 0.0, 300.0, key=f"{p}_time", step=0.5)
        st.number_input("💬 Discussion Posts", 0, 80, key=f"{p}_posts", step=1)
        st.number_input("⚠️ Late Submissions", 0, 20, key=f"{p}_late", step=1)
    with c3:
        st.selectbox("👩‍🏫 Teacher Feedback", ["Poor", "Fair", "Good", "Excellent"], key=f"{p}_feedback")
        st.selectbox("📚 Previous Result", ["Pass", "Fail"], key=f"{p}_prev")
        st.number_input("✅ Course Completion (%)", 0.0, 100.0, key=f"{p}_completion", step=1.0)
        predict_clicked = st.button("🔮 Predict Risk", type="primary", width="stretch")

    if predict_clicked:
        features = {
            "Student_ID": picked or "Manual Entry",
            "Attendance": st.session_state[f"{p}_attendance"],
            "Assignment_Score": st.session_state[f"{p}_assignment"],
            "Quiz_Score": st.session_state[f"{p}_quiz"],
            "Midterm": st.session_state[f"{p}_midterm"],
            "Final": st.session_state[f"{p}_final"],
            "GPA": st.session_state[f"{p}_gpa"],
            "LMS_Logins": st.session_state[f"{p}_lms"],
            "Time_Spent": st.session_state[f"{p}_time"],
            "Discussion_Posts": st.session_state[f"{p}_posts"],
            "Late_Submissions": st.session_state[f"{p}_late"],
            "Teacher_Feedback": st.session_state[f"{p}_feedback"],
            "Course_Completion": st.session_state[f"{p}_completion"],
            "Previous_Result": 1 if st.session_state[f"{p}_prev"] == "Pass" else 0,
        }
        prog = st.progress(0, text="Starting analysis…")
        steps = ["Loading model", "Cleaning & encoding features", "Computing prediction", "Running SHAP explanation", "Building interventions"]
        for i, msg in enumerate(steps, start=1):
            time.sleep(0.15)
            prog.progress(int(i / len(steps) * 100), text=msg)
        prog.progress(100, text="Done")

        result = predict.predict_single(art, features)
        show_result(art, features, result)


def page_insights():
    art = get_artifact()
    df = get_predictions()
    best = art["best"]

    st.markdown("<h1 class='grad-header'>📈 Insights & EDA</h1>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🧭 Data Overview</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%")
    m2.metric("Average GPA", f"{df['GPA'].mean():.2f}")
    m3.metric("Avg Course Completion", f"{df['Course_Completion'].mean():.1f}%")
    m4.metric("Avg Discussion Posts", f"{df['Discussion_Posts'].mean():.1f}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(attendance_hist(df), width="stretch")
    with c2:
        st.plotly_chart(risk_pie(df), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(marks_dist(df), width="stretch")
    with c4:
        st.plotly_chart(gpa_by_risk(df), width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(completion_by_risk(df), width="stretch")
    with c6:
        st.plotly_chart(feedback_by_risk(df), width="stretch")

    st.markdown("<div class='section-title'>🔗 Correlation Heatmap</div>", unsafe_allow_html=True)
    st.plotly_chart(corr_heat(df), width="stretch")

    st.markdown("<div class='section-title'>⚙️ Model Performance</div>", unsafe_allow_html=True)
    comp = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
                "ROC-AUC": round(m["roc_auc"], 4),
            }
            for name, m in art["metrics"].items()
        ]
    )
    comp[""] = comp["Model"].map({best: "⭐ BEST"})
    st.dataframe(comp, width="stretch", hide_index=True)

    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(conf_matrix(art["metrics"][best]["confusion_matrix"], best), width="stretch")
    with c8:
        rep = art["metrics"][best]["classification_report"]
        rows = [
            {
                "Class": k,
                "Precision": v["precision"],
                "Recall": v["recall"],
                "F1": v["f1-score"],
                "Support": v["support"],
            }
            for k, v in rep.items()
            if isinstance(v, dict)
        ]
        st.markdown("<div class='section-title'>📋 Classification Report</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("<div class='section-title'>🧬 SHAP Global Explanation (Summary)</div>", unsafe_allow_html=True)
    with st.spinner("Computing SHAP values…"):
        X_all, _ = utils.clean_data(get_raw(), prep=art["prep"])
        X_all = X_all[utils.ALL_FEATURES]
        if art.get("scaler") is not None:
            X_all = pd.DataFrame(art["scaler"].transform(X_all), columns=utils.ALL_FEATURES)
        sample = X_all.sample(min(120, len(X_all)), random_state=42)
        st.pyplot(predict.shap_summary(art, sample))


def page_ethics():
    st.markdown("<h1 class='grad-header'>🛡️ AI Ethics & Responsible Use</h1>", unsafe_allow_html=True)
    st.caption("Prediction is a decision-support tool — final decisions always stay with humans.")

    cards = [
        ("🔒", "Privacy Protection", "Student data is processed locally and never leaves your machine. The app supports fully offline deployment, and predictions can be exported without raw personal identifiers."),
        ("⚖️", "Bias Mitigation", "The system reports class-level precision, recall and F1 so skewed behaviour toward any group can be detected. Monitor fairness across demographic segments before acting on outputs."),
        ("🔍", "Transparency", "Every prediction is explained with SHAP values — feature-by-feature reasons for the risk score — so teachers can see exactly why a student was flagged."),
        ("🤝", "Fair AI", "Machine predictions are probabilities, not verdicts. They are combined with teacher judgement, context and empathy to avoid automated labelling of students."),
        ("🧑‍🏫", "Human Decision Support", "This system never expels, fails or penalises a student automatically. It only highlights who may benefit from early intervention and support."),
        ("🛡️", "Intervention, not Punishment", "High-risk flags always map to supportive actions — mentoring, study groups and monitoring — never to punitive measures."),
    ]
    cols = st.columns(3)
    for i, (icon, title, body) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(
                f"<div class='ethical-card'><div style='font-size:30px'>{icon}</div>"
                f"<div style='font-weight:700;font-size:17px;margin:6px 0'>{title}</div>"
                f"<div style='color:#9fb2d9;font-size:13.5px'>{body}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title'>⚠️ Model Limitations</div>", unsafe_allow_html=True)
    st.warning(
        "Risk scores are trained on historical academic patterns and may carry hidden biases. "
        "Always validate predictions with in-person context before any conversation with a student or family."
    )


def page_about():
    st.markdown("<h1 class='grad-header'>ℹ️ About the Project</h1>", unsafe_allow_html=True)
    readme = BASE_DIR / "README.md"
    if readme.exists():
        st.markdown(readme.read_text(encoding="utf-8"))
    else:
        st.info("README.md not found.")


def build_sidebar():
    with st.sidebar:
        if LOGO_FILE.exists():
            st.image(str(LOGO_FILE), width=140)
        st.markdown("<h3 style='text-align:center;margin-top:0'>Student Risk AI</h3>", unsafe_allow_html=True)
        st.markdown("---")
        st.session_state.nav = st.radio(
            "Navigate",
            ["📊 Dashboard", "🔮 Predict a Student", "📈 Insights & EDA", "🛡️ AI Ethics", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("### 📂 Dataset")
        uploaded = st.file_uploader("Upload students CSV / Excel", type=["csv", "xlsx"])
        if uploaded is not None:
            st.session_state.raw_data = utils.load_data(uploaded)
            st.session_state.df_pred = None
            st.success(f"Loaded {len(st.session_state.raw_data):,} students")
            rerun()
        if st.button("↩️ Reset to sample dataset"):
            st.session_state.raw_data = None
            st.session_state.df_pred = None
            rerun()

        st.markdown("### 🧠 Model")
        if st.button("🔄 Retrain Model"):
            with st.spinner("Training Random Forest, Decision Tree & Logistic Regression…"):
                _, _, best_name = train.train(save=True)
            st.session_state.model_version += 1
            st.session_state.df_pred = None
            load_artifact.clear()
            st.success(f"Retrained — best: {best_name}")
            rerun()

        st.markdown("### ⬇️ Export")
        if st.session_state.df_pred is not None:
            csv_data = st.session_state.df_pred.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV", csv_data, file_name="predictions.csv", mime="text/csv"
            )

        st.markdown("---")
        st.caption("AI-Powered Student Risk Prediction · v1.0")


def main():
    init_state()
    inject_css()
    build_sidebar()

    if st.session_state.nav == "📊 Dashboard":
        page_dashboard()
    elif st.session_state.nav == "🔮 Predict a Student":
        page_predict()
    elif st.session_state.nav == "📈 Insights & EDA":
        page_insights()
    elif st.session_state.nav == "🛡️ AI Ethics":
        page_ethics()
    else:
        page_about()


if __name__ == "__main__":
    main()
