import json
import os
from typing import Tuple

import numpy as np
import pandas as pd

from .config import AGG_MONTHLY_FILE, RISK_RANKING_FILE, MODEL_METRICS_FILE

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create feature matrix X and binary target y.
    Target: community-month is high risk if incident_count in top quartile.
    """
    df = df.copy()
    threshold = df["incident_count"].quantile(0.75)
    df["high_risk"] = (df["incident_count"] >= threshold).astype(int)

    feature_cols = [c for c in df.columns if c.startswith("z_")]
    # Include lagged incident counts as feature
    df = df.sort_values(["community_area", "month_start"]) 
    df["incident_count_lag1"] = df.groupby("community_area")["incident_count"].shift(1).fillna(0)
    feature_cols += ["incident_count_lag1"]

    # add second lag
    df["incident_count_lag2"] = df.groupby("community_area")["incident_count"].shift(2).fillna(0)
    feature_cols += ["incident_count_lag2"]

    X = df[feature_cols].fillna(0)
    y = df["high_risk"]
    return X, y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _logistic_train(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 300) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        z = X @ w
        p = _sigmoid(z)
        grad = X.T @ (p - y) / X.shape[0]
        w -= lr * grad
    return w


def train_and_evaluate() -> dict:
    """
    Train a simple logistic regression (numpy) and save metrics and risk rankings.
    """
    df = pd.read_csv(AGG_MONTHLY_FILE, parse_dates=["month_start"])
    Xdf, y = _build_features(df)
    # Train/test split
    idx = np.arange(len(Xdf))
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.75 * len(idx))
    tr, te = idx[:split], idx[split:]
    X_train = Xdf.iloc[tr].to_numpy()
    y_train = y.iloc[tr].to_numpy()
    X_test = Xdf.iloc[te].to_numpy()
    y_test = y.iloc[te].to_numpy()

    # Standardize columns
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-9
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    w = _logistic_train(X_train, y_train, lr=0.1, epochs=500)
    y_pred_prob = _sigmoid(X_test @ w)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Compute simple metrics
    accuracy = float((y_pred == y_test).mean())
    precision = float(( (y_pred & y_test).sum() / (y_pred.sum() + 1e-9) ))
    recall = float(( (y_pred & y_test).sum() / (y_test.sum() + 1e-9) ))
    f1 = float(2 * precision * recall / (precision + recall + 1e-9))
    report = {
        "accuracy": accuracy,
        "macro avg": {"precision": precision, "recall": recall, "f1-score": f1},
    }

    # Risk ranking by community area using latest month entries
    latest = df.groupby("community_area").apply(lambda g: g.sort_values("month_start").tail(1)).reset_index(drop=True)
    X_latest, _ = _build_features(latest)
    Xl = (X_latest.to_numpy() - mean) / std
    latest["risk_score"] = _sigmoid(Xl @ w)
    ranking = latest[["community_area", "risk_score", "incident_count"]].sort_values("risk_score", ascending=False)
    ranking.to_csv(RISK_RANKING_FILE, index=False)

    os.makedirs(os.path.dirname(MODEL_METRICS_FILE), exist_ok=True)
    with open(MODEL_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def train_random_forest() -> dict:
    if not SKLEARN_AVAILABLE:
        return {"available": False}
    df = pd.read_csv(AGG_MONTHLY_FILE, parse_dates=["month_start"])
    Xdf, y = _build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(Xdf, y, test_size=0.25, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    # simple metrics
    acc = float((y_pred == y_test).mean())
    # risk score for latest month
    latest = df.groupby("community_area").apply(lambda g: g.sort_values("month_start").tail(1)).reset_index(drop=True)
    X_latest, _ = _build_features(latest)
    latest["rf_risk_score"] = rf.predict_proba(X_latest)[:, 1]
    # Write back rf risk to monthly file (align by community_area + month_start)
    df = df.merge(
        latest[["community_area", "month_start", "rf_risk_score"]],
        on=["community_area", "month_start"],
        how="left",
    )
    df.to_csv(AGG_MONTHLY_FILE, index=False)
    return {"accuracy": acc, "available": True}
