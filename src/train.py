"""Entrenamiento: baseline (regresión logística) + XGBoost.

Ajusta hiperparámetros con RandomizedSearchCV sobre el hold-out de validación
(15 %). El modelo final se reentrena solo en el 70 % de train.
Guarda models/logistic.joblib, models/xgboost.joblib y models/experiment.json.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    N_ITER_SEARCH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)


def _load_xy(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Carga un parquet (train/val/test) y separa X e y."""
    frame = pd.read_parquet(DATA_PROCESSED / f"{name}.parquet")
    y = frame[TARGET_COLUMN]
    X = frame.drop(columns=[TARGET_COLUMN])
    return X, y


def load_train_val() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carga train (70 %) y val (15 %) para tunear sin tocar el test."""
    X_train, y_train = _load_xy("train")
    X_val, y_val = _load_xy("val")
    return X_train, y_train, X_val, y_val


def build_logistic() -> Pipeline:
    """Logística con datos estandarizados y pesos de clase (desbalance ~9:1)."""
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
    """XGBoost binario. scale_pos_weight ≈ n_neg/n_pos para el desbalance."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


def _predefined_cv(n_train: int, n_val: int) -> PredefinedSplit:
    """Marca train como entrenamiento fijo y val como único fold de prueba."""
    test_fold = np.concatenate(
        [np.full(n_train, -1, dtype=int), np.zeros(n_val, dtype=int)]
    )
    return PredefinedSplit(test_fold)


def tune_logistic(X_train, y_train, X_val, y_val) -> tuple[Pipeline, dict]:
    """Busca el mejor C maximizando AUC-ROC en el conjunto de validación."""
    pipe = build_logistic()
    param_dist = {
        "clf__C": np.logspace(-3, 2, 20),
    }
    X_all = pd.concat([X_train, X_val], axis=0)
    y_all = pd.concat([y_train, y_val], axis=0)
    cv = _predefined_cv(len(X_train), len(X_val))
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=min(N_ITER_SEARCH, 12),
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
        verbose=1,
    )
    search.fit(X_all, y_all)
    best = build_logistic().set_params(**search.best_params_)
    best.fit(X_train, y_train)
    return best, {
        "best_params": search.best_params_,
        "best_val_auc": float(search.best_score_),
    }


def tune_xgboost(
    X_train, y_train, X_val, y_val, scale_pos_weight: float
) -> tuple[XGBClassifier, dict]:
    """Prueba combinaciones aleatorias y elige por AUC-ROC en validación."""
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
    X_all = pd.concat([X_train, X_val], axis=0)
    y_all = pd.concat([y_train, y_val], axis=0)
    cv = _predefined_cv(len(X_train), len(X_val))
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
        verbose=1,
    )
    search.fit(X_all, y_all)
    best = build_xgboost(scale_pos_weight).set_params(**search.best_params_)
    best.fit(X_train, y_train)
    return best, {
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "best_val_auc": float(search.best_score_),
        "scale_pos_weight": float(scale_pos_weight),
    }


def _jsonable(v):
    """Convierte numpy.int64/float64 a tipos nativos para poder guardar JSON."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def run() -> dict:
    """Entrena ambos modelos, los serializa y anota hiperparámetros y AUC de train/val."""
    X_train, y_train, X_val, y_val = load_train_val()
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    print("=== Baseline: Regresión logística ===")
    log_model, log_info = tune_logistic(X_train, y_train, X_val, y_val)
    print("=== Modelo propuesto: XGBoost ===")
    xgb_model, xgb_info = tune_xgboost(
        X_train, y_train, X_val, y_val, scale_pos_weight
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(log_model, MODELS_DIR / "logistic.joblib")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")

    # Probabilidades en train/val solo para sanity-check (no decide el test).
    log_auc_train = float(roc_auc_score(y_train, log_model.predict_proba(X_train)[:, 1]))
    xgb_auc_train = float(roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1]))
    log_auc_val = float(roc_auc_score(y_val, log_model.predict_proba(X_val)[:, 1]))
    xgb_auc_val = float(roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1]))

    experiment = {
        "random_state": RANDOM_STATE,
        "split": {"train": TRAIN_SIZE, "val": VAL_SIZE, "test": TEST_SIZE},
        "n_iter_search": N_ITER_SEARCH,
        "class_balance_train": {"neg": neg, "pos": pos},
        "logistic": {
            **log_info,
            "train_auc": log_auc_train,
            "val_auc": log_auc_val,
        },
        "xgboost": {
            **xgb_info,
            "train_auc": xgb_auc_train,
            "val_auc": xgb_auc_val,
        },
    }
    (MODELS_DIR / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    print(json.dumps(experiment, indent=2))
    return experiment


if __name__ == "__main__":
    run()
