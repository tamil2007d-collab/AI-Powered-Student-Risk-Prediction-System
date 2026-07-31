"""train.py — Train, compare and save the best risk prediction model.

Trains a Random Forest, Decision Tree and Logistic Regression on the
cleaned student data, evaluates each model, selects the best by ROC-AUC
and persists it (plus preprocessing metadata) to models/model.pkl.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import utils

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "students.csv"
MODEL_FILE = BASE_DIR / "models" / "model.pkl"
METRICS_FILE = BASE_DIR / "models" / "metrics.json"


def get_models() -> dict:
    rs = utils.RANDOM_STATE
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=rs,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=4, random_state=rs
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1500, solver="liblinear", random_state=rs
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute the standard classification metrics."""
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(auc),
    }


def compute_feature_importance(model, X: pd.DataFrame, name: str) -> dict:
    """Model-agnostic feature importance (coefs for linear, gain for trees)."""
    if name == "Logistic Regression":
        return {c: float(abs(coef)) for c, coef in zip(X.columns, model.coef_[0])}
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        return {c: float(v) for c, v in zip(X.columns, fi)}
    return {c: 0.0 for c in X.columns}


def print_report(results: dict, best_name: str) -> None:
    print("\n" + "=" * 78)
    print("MODEL COMPARISON  (test set)")
    print("=" * 78)
    print(f"{'Model':<22}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}")
    print("-" * 78)
    for name, m in results.items():
        star = "   <-- BEST" if name == best_name else ""
        print(
            f"{name:<22}{m['accuracy']:>8.3f}{m['precision']:>8.3f}"
            f"{m['recall']:>8.3f}{m['f1']:>8.3f}{m['roc_auc']:>8.3f}{star}"
        )
    print("=" * 78)


def train(dataset_path=None, save: bool = True):
    """Run the full training pipeline and return the saved artifact."""
    utils.ensure_sample_data(DATA_FILE)
    df = utils.load_data(dataset_path or DATA_FILE)

    df_clean, prep = utils.clean_data(df)
    X, y = df_clean[utils.ALL_FEATURES], df_clean[utils.TARGET]
    X_train, X_test, y_train, y_test = utils.split_data(df_clean)

    models = get_models()
    scaler = StandardScaler()
    results: dict = {}
    fitted: dict = {}

    for name, model in models.items():
        t0 = time.time()
        if name == "Logistic Regression":
            Xtr, Xte = scaler.fit_transform(X_train), scaler.transform(X_test)
        else:
            Xtr, Xte = X_train, X_test
        model.fit(Xtr, y_train)

        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        m = evaluate(y_test, y_pred, y_prob)
        m["training_time"] = round(time.time() - t0, 3)
        m["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        m["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
        m["feature_importance"] = compute_feature_importance(model, X_test, name)
        results[name] = m
        fitted[name] = model

    best_name = max(results, key=lambda n: (results[n]["roc_auc"], results[n]["accuracy"]))
    best_model = fitted[best_name]

    artifact = {
        "model": best_model,
        "type": best_name,
        "features": utils.ALL_FEATURES,
        "prep": prep,
        "scaler": scaler if best_name == "Logistic Regression" else None,
        "metrics": results,
        "best": best_name,
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_samples": int(len(df_clean)),
        "risk_rate": float(y.mean()),
    }

    if save:
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, MODEL_FILE)
        with open(METRICS_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    name: {
                        k: v
                        for k, v in m.items()
                        if k not in ("classification_report", "confusion_matrix", "feature_importance")
                    }
                    for name, m in results.items()
                },
                fh,
                indent=2,
            )
        print(f"\nSaved model          -> {MODEL_FILE}")
        print(f"Saved metrics        -> {METRICS_FILE}")
        print(f"Best model           -> {best_name}")
        print(f"Training rows        -> {artifact['n_samples']:,}")
        print(f"At-risk label rate   -> {artifact['risk_rate']:.1%}")

    print_report(results, best_name)
    return artifact, results, best_name


if __name__ == "__main__":
    train(save=True)
