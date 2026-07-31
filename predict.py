"""predict.py — Inference helpers for the risk prediction model.

Loads the saved artifact, predicts risk probabilities for single students
or whole batches, and produces SHAP explanations (waterfall, force plot,
beeswarm summary) for explainable AI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

import utils

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "model.pkl"


def load_model(model_file=MODEL_FILE):
    """Load the saved artifact dictionary, or None if missing."""
    path = Path(model_file)
    if not path.exists():
        return None
    return joblib.load(path)


def to_features_df(data) -> pd.DataFrame:
    """Convert a dict or DataFrame into a feature-only DataFrame."""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    for col in utils.ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[utils.ALL_FEATURES]


def _prepare_x(artifact: dict, data) -> pd.DataFrame:
    """Apply the fitted cleaning pipeline (and scaler) to new input."""
    df = to_features_df(data)
    df, _ = utils.clean_data(df, prep=artifact["prep"])
    X = df[utils.ALL_FEATURES]
    if artifact.get("scaler") is not None:
        X = pd.DataFrame(artifact["scaler"].transform(X), columns=utils.ALL_FEATURES)
    return X


def predict_single(artifact: dict, data: dict) -> dict:
    """Predict risk for one student and return probability + category."""
    X = _prepare_x(artifact, data)
    proba = float(artifact["model"].predict_proba(X)[:, 1][0])
    return {
        "probability": proba,
        "category": utils.get_risk_category(proba),
        "confidence": max(proba, 1 - proba) * 100,
        "recommendations": utils.get_recommendations(proba),
        "X_row": X,
    }


def predict_batch(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Score a full dataset and append probability / category columns."""
    base = df.copy()
    cols = [c for c in base.columns if c != utils.STUDENT_ID]
    base = base.drop_duplicates(subset=cols, keep="first")
    X = _prepare_x(artifact, base)
    proba = artifact["model"].predict_proba(X)[:, 1]
    out = base.reset_index(drop=True).copy()
    out["Risk_Probability"] = np.round(proba, 4)
    out["Predicted_Risk"] = (proba >= 0.5).astype(int)
    out["Risk_Category"] = out["Risk_Probability"].map(utils.get_risk_category)
    return out


def _explainer(artifact: dict, X):
    """Pick the right SHAP explainer for the chosen model."""
    model = artifact["model"]
    if artifact["type"] == "Logistic Regression":
        return shap.LinearExplainer(model, X)
    return shap.TreeExplainer(model)


def _positive_class(base, sv):
    """Normalise SHAP outputs to the positive (at-risk) class."""
    if isinstance(sv, list):
        if isinstance(base, (list, tuple, np.ndarray)) and len(base) > 1:
            base = base[1]
        sv = sv[1]
    base = float(np.atleast_1d(base)[0])
    return base, np.asarray(sv)


def shap_for_row(artifact: dict, X_row: pd.DataFrame):
    """SHAP values for a single prediction row."""
    explainer = _explainer(artifact, X_row)
    sv = explainer.shap_values(X_row)
    base, values = _positive_class(explainer.expected_value, sv)
    return base, values[0], explainer


def waterfall_figure(base: float, values: np.ndarray, data_row: np.ndarray):
    """Matplotlib waterfall figure explaining a single prediction."""
    import matplotlib.pyplot as plt

    exp = shap.Explanation(
        values=values,
        base_values=base,
        data=np.asarray(data_row).reshape(-1),
        feature_names=utils.ALL_FEATURES,
    )
    fig, _ = plt.subplots(figsize=(9, 4.4))
    shap.plots.waterfall(exp, max_display=14, show=False)
    return fig


def force_plot_html(base: float, values: np.ndarray, data_row: np.ndarray) -> str:
    """Interactive SHAP force plot as self-contained HTML."""
    vis = shap.force_plot(
        base,
        values,
        np.asarray(data_row).reshape(-1),
        feature_names=utils.ALL_FEATURES,
        matplotlib=False,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as tmp:
        shap.save_html(tmp.name, vis)
        return Path(tmp.name).read_text(encoding="utf-8")


def shap_summary(artifact: dict, X: pd.DataFrame):
    """Beeswarm summary plot on a sample of rows."""
    import matplotlib.pyplot as plt

    explainer = _explainer(artifact, X)
    sv = explainer.shap_values(X)
    base, values = _positive_class(explainer.expected_value, sv)
    exp = shap.Explanation(
        values=values,
        base_values=base,
        data=np.asarray(X),
        feature_names=utils.ALL_FEATURES,
    )
    shap.plots.beeswarm(exp, max_display=14, show=False)
    fig = plt.gcf()
    plt.close()
    return fig
