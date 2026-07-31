"""utils.py — Shared helpers for the AI-Powered Student Risk Prediction System.

Contains synthetic dataset generation, cleaning, preprocessing, evaluation
helpers and the intervention / recommendation engine used by train.py,
predict.py and app.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

TARGET = "Risk"
STUDENT_ID = "Student_ID"

NUMERICAL_FEATURES = [
    "Attendance",
    "Assignment_Score",
    "Quiz_Score",
    "Midterm",
    "Final",
    "GPA",
    "LMS_Logins",
    "Time_Spent",
    "Discussion_Posts",
    "Late_Submissions",
    "Course_Completion",
]

CATEGORICAL_FEATURES = ["Teacher_Feedback", "Previous_Result"]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

FEEDBACK_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
FEEDBACK_INV = {v: k for k, v in FEEDBACK_MAP.items()}

RISK_LABELS = {0: "Safe", 1: "At Risk"}


def generate_sample_data(n: int = 1000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a realistic synthetic dataset of students for the demo.

    Academic and engagement features are generated from a latent ability
    variable, then a weighted risk score produces the binary Risk target.
    """
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n)

    assignment = (66 + 13 * ability + rng.normal(0, 8, n)).clip(0, 100)
    quiz = (63 + 12 * ability + rng.normal(0, 9, n)).clip(0, 100)
    midterm = (61 + 15 * ability + rng.normal(0, 12, n)).clip(0, 100)
    final = (59 + 17 * ability + rng.normal(0, 13, n)).clip(0, 100)
    gpa = (2.65 + 0.55 * ability + rng.normal(0, 0.35, n)).clip(0, 4)
    attendance = (81 + 9 * ability + rng.normal(0, 10, n)).clip(20, 100)
    completion = (83 + 9 * ability + rng.normal(0, 12, n)).clip(0, 100)
    lms = (42 + 12 * ability + rng.normal(0, 14, n)).clip(0, 100).round()
    time_spent = (62 + 19 * ability + rng.normal(0, 26, n)).clip(0, 200)
    posts = (21 + 8 * ability + rng.normal(0, 10, n)).clip(0, 60).round()
    late = np.clip(np.round(7 - 2.5 * ability + rng.normal(0, 2.5, n)), 0, 15)
    previous = (rng.random(n) < (0.72 + 0.15 * ability)).astype(int)

    feedback_num = ability + rng.normal(0, 0.5, n)
    feedback = np.select(
        [feedback_num > 0.8, feedback_num > 0.15, feedback_num > -0.5],
        ["Excellent", "Good", "Fair"],
        default="Poor",
    )

    marks_avg = (assignment + quiz + midterm + final) / 4
    fb_risk = np.select(
        [feedback == "Poor", feedback == "Fair", feedback == "Good"],
        [1.0, 0.66, 0.33],
        default=0.0,
    )

    risk_score = (
        0.20 * (1 - attendance / 100)
        + 0.20 * (1 - marks_avg / 100)
        + 0.15 * (1 - gpa / 4)
        + 0.10 * (1 - completion / 100)
        + 0.08 * (late / 15)
        + 0.05 * (1 - lms / 100)
        + 0.05 * (1 - time_spent / 200)
        + 0.05 * (1 - posts / 60)
        + 0.07 * fb_risk
        + 0.05 * (1 - previous)
    )
    risk = (risk_score + rng.normal(0, 0.055, n) > 0.46).astype(int)

    df = pd.DataFrame(
        {
            STUDENT_ID: [f"STU{i:04d}" for i in range(1, n + 1)],
            "Attendance": attendance.round(1),
            "Assignment_Score": assignment.round(1),
            "Quiz_Score": quiz.round(1),
            "Midterm": midterm.round(1),
            "Final": final.round(1),
            "GPA": gpa.round(2),
            "LMS_Logins": lms.astype(int),
            "Time_Spent": time_spent.round(1),
            "Discussion_Posts": posts.astype(int),
            "Late_Submissions": late.astype(int),
            "Teacher_Feedback": feedback,
            "Course_Completion": completion.round(1),
            "Previous_Result": previous,
            TARGET: risk,
        }
    )

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_nan = int(n * 0.03)
    df.loc[rng.choice(n, size=n_nan, replace=False), "Time_Spent"] = np.nan
    df.loc[rng.choice(n, size=n_nan, replace=False), "Discussion_Posts"] = np.nan
    df.loc[rng.choice(n, size=int(n * 0.01), replace=False), "Final"] = np.nan

    df = pd.concat([df, df.head(8)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    return df


def load_data(path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def clean_data(df: pd.DataFrame, prep: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean a raw dataframe.

    Removes duplicates, fills missing values, caps outliers with the IQR
    method and encodes categorical columns. When ``prep`` is None the
    cleaning parameters are fitted and returned so they can be re-used at
    prediction time.
    """
    data = df.copy()
    if prep is None:
        prep = {"fill": {}, "bounds": {}, "feedback_map": FEEDBACK_MAP}

    cols = [c for c in data.columns if c != STUDENT_ID]
    data = data.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)

    if prep["fill"]:
        for col in NUMERICAL_FEATURES:
            if col in data.columns:
                data[col] = data[col].fillna(prep["fill"][col])
        for col in CATEGORICAL_FEATURES:
            if col in data.columns:
                data[col] = data[col].fillna(prep["fill"][col])
    else:
        for col in NUMERICAL_FEATURES:
            if col in data.columns:
                med = data[col].median()
                prep["fill"][col] = float(med)
                data[col] = data[col].fillna(med)
        for col in CATEGORICAL_FEATURES:
            if col in data.columns:
                mode = data[col].mode().iloc[0]
                prep["fill"][col] = mode
                data[col] = data[col].fillna(mode)

    if prep["bounds"]:
        for col, (lo, hi) in prep["bounds"].items():
            if col in data.columns:
                data[col] = data[col].clip(lo, hi)
    else:
        for col in NUMERICAL_FEATURES:
            if col in data.columns:
                q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
                iqr = q3 - q1
                lo = max(q1 - 1.5 * iqr, data[col].min())
                hi = min(q3 + 1.5 * iqr, data[col].max())
                prep["bounds"][col] = (float(lo), float(hi))
                data[col] = data[col].clip(lo, hi)

    fmap = prep.get("feedback_map", FEEDBACK_MAP)
    data["Teacher_Feedback"] = data["Teacher_Feedback"].map(fmap).fillna(1).astype(int)
    data["Previous_Result"] = data["Previous_Result"].replace({"Pass": 1, "Fail": 0}).astype(int)

    if TARGET in data.columns:
        data[TARGET] = data[TARGET].astype(int)

    return data, prep


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Stratified train/test split on the cleaned feature set."""
    X = df[ALL_FEATURES]
    y = df[TARGET]
    stratify = y if len(np.unique(y)) > 1 else None
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)


def get_risk_category(prob: float) -> str:
    """Map a risk probability to a human readable category."""
    if prob >= 0.70:
        return "High Risk"
    if prob >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def get_recommendations(prob: float) -> list[tuple[str, str, str]]:
    """Intervention engine — returns (icon, title, description) tuples."""
    if prob >= 0.70:
        return [
            ("👥", "Meet Mentor", "Schedule a one-on-one meeting with an academic mentor this week."),
            ("🏠", "Parent Meeting", "Arrange a parent-teacher meeting to align on a support plan."),
            ("📝", "Extra Assignments", "Provide additional practice assignments to reinforce weak topics."),
            ("📊", "Weekly Monitoring", "Track attendance, LMS activity and marks on a weekly basis."),
        ]
    if prob >= 0.40:
        return [
            ("🎯", "Weekly Practice", "Complete a weekly practice quiz to close knowledge gaps."),
            ("📅", "Improve Attendance", "Target a minimum of 85% attendance for the next month."),
            ("👩‍🏫", "Join Study Groups", "Participate in peer study groups and discussion forums."),
        ]
    return [
        ("⭐", "Continue Performance", "Keep up the excellent work and consistent study routine."),
        ("🚀", "Advanced Learning", "Explore advanced modules, electives and research opportunities."),
    ]


def ensure_sample_data(path: Path, n: int = 1000) -> None:
    """Generate the sample dataset on first run if it does not exist."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_sample_data(n=n).to_csv(path, index=False)
