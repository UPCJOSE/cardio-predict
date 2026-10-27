"""Entrenamiento: baseline (regresión logística) + XGBoost con ajuste de hiperparámetros."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    N_CV_FOLDS,
    N_ITER_SEARCH,
    RANDOM_STATE,
    TARGET_COLUMN,
)


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_parquet(DATA_PROCESSED / "train.parquet")
    y = train[TARGET_COLUMN]
    X = train.drop(columns=[TARGET_COLUMN])
    return X, y


def build_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


def tune_logistic(X, y, cv) -> tuple[Pipeline, dict]:
    pipe = build_logistic()
    param_dist = {
        "clf__C": np.logspace(-3, 2, 20),
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=min(N_ITER_SEARCH, 12),
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X, y)
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_auc": float(search.best_score_),
    }


def tune_xgboost(X, y, cv, scale_pos_weight: float) -> tuple[XGBClassifier, dict]:
    model = build_xgboost(scale_pos_weight)
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.5],
        "reg_lambda": [1.0, 5.0, 10.0],
    }
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X, y)
    return search.best_estimator_, {
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "best_cv_auc": float(search.best_score_),
        "scale_pos_weight": float(scale_pos_weight),
    }


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def run() -> dict:
    X, y = load_train()
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("=== Baseline: Regresión logística ===")
    log_model, log_info = tune_logistic(X, y, cv)
    print("=== Modelo propuesto: XGBoost ===")
    xgb_model, xgb_info = tune_xgboost(X, y, cv, scale_pos_weight)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(log_model, MODELS_DIR / "logistic.joblib")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")

    # Probabilidades en train solo para sanity-check (no decide).
    log_auc_train = float(roc_auc_score(y, log_model.predict_proba(X)[:, 1]))
    xgb_auc_train = float(roc_auc_score(y, xgb_model.predict_proba(X)[:, 1]))

    experiment = {
        "random_state": RANDOM_STATE,
        "n_cv_folds": N_CV_FOLDS,
        "n_iter_search": N_ITER_SEARCH,
        "class_balance_train": {"neg": neg, "pos": pos},
        "logistic": {**log_info, "train_auc": log_auc_train},
        "xgboost": {**xgb_info, "train_auc": xgb_auc_train},
    }
    (MODELS_DIR / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    print(json.dumps(experiment, indent=2))
    return experiment


if __name__ == "__main__":
    run()
