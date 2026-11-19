"""Reentrenamiento solo de regresión logística (panel Admin).

Mide tiempos, guarda logistic.joblib, actualiza métricas del bloque logístico
y escribe reports/training_run.json para el dashboard.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ADMIN_SETTINGS_JSON,
    FEATURE_COLUMNS,
    MODELS_DIR,
    N_ITER_SEARCH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_SIZE,
    TRAINING_RUN_JSON,
    VAL_SIZE,
)
from src.data_prep import save_processed, split_data
from src.evaluate import (
    best_threshold_for_sensitivity,
    bootstrap_auc,
    high_risk_threshold,
    metrics_at_threshold,
)


def build_logistic() -> Pipeline:
    """Logística con estandarización (misma base que src.train)."""
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


def _predefined_cv(n_train: int, n_val: int) -> PredefinedSplit:
    test_fold = np.concatenate(
        [np.full(n_train, -1, dtype=int), np.zeros(n_val, dtype=int)]
    )
    return PredefinedSplit(test_fold)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _format_duration(seconds: float) -> dict:
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    secs = total - minutes * 60
    return {
        "seconds_total": round(total, 3),
        "minutes": minutes,
        "seconds": round(secs, 3),
        "human": f"{minutes} min {secs:05.2f} s" if minutes else f"{secs:.2f} s",
    }


def save_admin_settings(settings: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ADMIN_SETTINGS_JSON.write_text(
        json.dumps(_jsonable(settings), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_admin_settings() -> dict:
    if not ADMIN_SETTINGS_JSON.exists():
        return {
            "train_pct": int(TRAIN_SIZE * 100),
            "val_pct": int(VAL_SIZE * 100),
            "test_pct": int(TEST_SIZE * 100),
            "tune_C": True,
            "C": 1.0,
            "class_weight": "balanced",
            "n_iter_search": min(N_ITER_SEARCH, 12),
        }
    return json.loads(ADMIN_SETTINGS_JSON.read_text(encoding="utf-8"))


def _fit_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    tune_C: bool,
    C: float,
    class_weight: str | dict,
    n_iter_search: int,
) -> tuple[Any, dict, float]:
    """Entrena (con o sin búsqueda de C) y devuelve modelo, info y segundos."""
    t0 = time.perf_counter()
    base = build_logistic()
    base.set_params(clf__class_weight=class_weight)

    search_curve: list[dict] = []
    if tune_C:
        param_dist = {"clf__C": np.logspace(-3, 2, 20)}
        X_all = pd.concat([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0)
        cv = _predefined_cv(len(X_train), len(X_val))
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=max(1, int(n_iter_search)),
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=False,
            verbose=0,
        )
        search.fit(X_all, y_all)
        best = build_logistic().set_params(
            clf__class_weight=class_weight, **search.best_params_
        )
        best.fit(X_train, y_train)

        # Curva tipo MATLAB: error de clasificación ≈ 1 − AUC en cada intento.
        mean_auc = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
        errors = 1.0 - mean_auc
        running_min = np.minimum.accumulate(errors)
        c_tried = [
            float(p.get("clf__C", float("nan")))
            for p in search.cv_results_["params"]
        ]
        for i, (err, min_err, c_val, auc) in enumerate(
            zip(errors, running_min, c_tried, mean_auc), start=1
        ):
            search_curve.append(
                {
                    "iteration": i,
                    "C": c_val,
                    "val_auc": float(auc),
                    "classification_error": float(err),
                    "min_classification_error": float(min_err),
                }
            )
        info = {
            "tuned": True,
            "best_params": _jsonable(search.best_params_),
            "best_val_auc_search": float(search.best_score_),
            "n_iter_search": int(n_iter_search),
            "search_curve": search_curve,
            "min_observed_classification_error": float(running_min[-1]),
        }
    else:
        best = build_logistic().set_params(clf__class_weight=class_weight, clf__C=float(C))
        best.fit(X_train, y_train)
        info = {
            "tuned": False,
            "best_params": {"clf__C": float(C)},
            "n_iter_search": 0,
            "search_curve": [],
        }

    elapsed = time.perf_counter() - t0
    return best, info, elapsed


def retrain_logistic_from_frame(
    df: pd.DataFrame,
    *,
    train_pct: int = 70,
    val_pct: int = 15,
    test_pct: int = 15,
    tune_C: bool = True,
    C: float = 1.0,
    class_weight: str = "balanced",
    n_iter_search: int = 12,
) -> dict:
    """Pipeline Admin: split → train logística → eval test → persistir artefactos."""
    if train_pct + val_pct + test_pct != 100:
        raise ValueError("train + val + test deben sumar 100%")

    train_size, val_size, test_size = train_pct / 100, val_pct / 100, test_pct / 100
    wall_t0 = time.perf_counter()

    t_split = time.perf_counter()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, train_size=train_size, val_size=val_size, test_size=test_size
    )
    split_seconds = time.perf_counter() - t_split

    quality = {
        "rows_final": int(len(df)),
        "positive_rate": float(df[TARGET_COLUMN].mean()),
        "split": {"train": train_size, "val": val_size, "test": test_size},
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
        "features": FEATURE_COLUMNS,
    }
    save_processed(X_train, X_val, X_test, y_train, y_val, y_test, quality)

    model, fit_info, fit_seconds = _fit_logistic(
        X_train,
        y_train,
        X_val,
        y_val,
        tune_C=tune_C,
        C=C,
        class_weight=class_weight,
        n_iter_search=n_iter_search,
    )

    t_eval = time.perf_counter()
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    auc_train = float(roc_auc_score(y_train, proba_train))
    auc_val = float(roc_auc_score(y_val, proba_val))
    auc_test = float(roc_auc_score(y_test, proba_test))
    auc_pr = float(average_precision_score(y_test, proba_test))
    brier = float(brier_score_loss(y_test, proba_test))
    boot_mean, boot_lo, boot_hi = bootstrap_auc(y_test, proba_test)

    t_screen, sens_s, spec_s, f1_s = best_threshold_for_sensitivity(y_test, proba_test)
    t_high, sens_h, spec_h = high_risk_threshold(y_test, proba_test)
    at_screen = metrics_at_threshold(y_test, proba_test, t_screen)
    at_05 = metrics_at_threshold(y_test, proba_test, 0.5)
    eval_seconds = time.perf_counter() - t_eval

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "logistic.joblib")

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    experiment = {
        "random_state": RANDOM_STATE,
        "split": {"train": train_size, "val": val_size, "test": test_size},
        "n_iter_search": int(n_iter_search if tune_C else 0),
        "class_balance_train": {"neg": neg, "pos": pos},
        "logistic": {
            **fit_info,
            "train_auc": auc_train,
            "val_auc": auc_val,
            "test_auc": auc_test,
            "class_weight": class_weight,
        },
        "source": "admin_retrain_logistic",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    # Conserva bloque xgboost previo si existía (no se reentrena aquí).
    prev_exp_path = MODELS_DIR / "experiment.json"
    if prev_exp_path.exists():
        try:
            prev = json.loads(prev_exp_path.read_text(encoding="utf-8"))
            if "xgboost" in prev:
                experiment["xgboost"] = prev["xgboost"]
        except Exception:  # noqa: BLE001
            pass
    prev_exp_path.write_text(json.dumps(_jsonable(experiment), indent=2), encoding="utf-8")

    logistic_metrics = {
        "auc_roc": auc_test,
        "auc_roc_bootstrap_mean": boot_mean,
        "auc_roc_ci95": [boot_lo, boot_hi],
        "auc_pr": auc_pr,
        "brier": brier,
        "at_0_5": at_05,
        "at_screening_threshold": {
            **at_screen,
            "threshold": float(t_screen),
        },
        "high_risk_threshold": {
            "threshold": float(t_high),
            "sensitivity": float(sens_h),
            "specificity": float(spec_h),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_doc: dict[str, Any] = {
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
        "models": {"Regresión logística": logistic_metrics},
        "selection": {
            "model": "Regresión logística",
            "reason": "Reentrenamiento Admin (solo logística).",
        },
    }
    if metrics_path.exists():
        try:
            old = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "XGBoost" in old.get("models", {}):
                metrics_doc["models"]["XGBoost"] = old["models"]["XGBoost"]
        except Exception:  # noqa: BLE001
            pass
    metrics_path.write_text(json.dumps(_jsonable(metrics_doc), indent=2), encoding="utf-8")

    wall_seconds = time.perf_counter() - wall_t0
    search_curve = list(fit_info.get("search_curve") or [])
    n_features = len(FEATURE_COLUMNS)
    # Meta de discriminación (AUC ≥ 0.80 ⇒ error 1−AUC ≤ 0.20). No inventa umbrales NN.
    performance_goal = 0.20
    min_err = fit_info.get("min_observed_classification_error")
    if min_err is None and search_curve:
        min_err = float(min(r["classification_error"] for r in search_curve))
    if min_err is None:
        min_err = float(1.0 - auc_val)
    initial_err = (
        float(search_curve[0]["classification_error"])
        if search_curve
        else float(1.0 - auc_val)
    )
    n_iter_done = int(len(search_curve) if search_curve else (1 if not tune_C else 0))
    n_iter_target = int(n_iter_search if tune_C else 1)
    met_goal = float(min_err) <= performance_goal
    status_detail = (
        "Entrenamiento terminado: se cumplió el criterio de desempeño"
        if met_goal
        else "Entrenamiento terminado: búsqueda finalizada (meta no alcanzada)"
    )

    # Muestra de prueba para gráfico Target vs Output (estilo plotregression).
    rng = np.random.default_rng(RANDOM_STATE)
    n_cal = min(2000, len(y_test))
    idx = rng.choice(len(y_test), size=n_cal, replace=False)
    y_cal = np.asarray(y_test, dtype=float)[idx]
    p_cal = np.asarray(proba_test, dtype=float)[idx]
    if np.std(y_cal) > 0 and np.std(p_cal) > 0:
        r_corr = float(np.corrcoef(y_cal, p_cal)[0, 1])
        slope, intercept = np.polyfit(y_cal, p_cal, 1)
    else:
        r_corr, slope, intercept = float("nan"), float("nan"), float("nan")

    run_report = {
        "model": "Regresión logística",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_pct": {"train": train_pct, "val": val_pct, "test": test_pct},
        "architecture": {
            "n_inputs": n_features,
            "steps": [
                {"name": "Input", "units": n_features},
                {"name": "StandardScaler", "units": n_features},
                {"name": "LogisticRegression (sigmoid)", "units": 1},
                {"name": "Output (probabilidad)", "units": 1},
            ],
        },
        "algorithms": {
            "data_division": f"estratificada {train_pct}/{val_pct}/{test_pct}",
            "training": "lbfgs (regresión logística)",
            "performance": "AUC-ROC (error reportado = 1 − AUC val)",
            "calculations": "scikit-learn",
        },
        "progress": {
            "iteration": {
                "initial": 0,
                "stopped": n_iter_done,
                "target": n_iter_target,
            },
            "elapsed_time": _format_duration(wall_seconds),
            "performance_error": {
                "initial": initial_err,
                "stopped": float(min_err),
                "target": performance_goal,
                "met": met_goal,
            },
            "best_C": fit_info.get("best_params", {}).get("clf__C", C),
            "best_val_auc": float(1.0 - float(min_err)),
        },
        "hyperparameters": {
            "tune_C": tune_C,
            "C": fit_info.get("best_params", {}).get("clf__C", C),
            "class_weight": class_weight,
            "n_iter_search": int(n_iter_search if tune_C else 0),
            "solver": "lbfgs",
            "max_iter": 2000,
        },
        "timing": {
            "split": _format_duration(split_seconds),
            "fit": _format_duration(fit_seconds),
            "evaluate": _format_duration(eval_seconds),
            "total": _format_duration(wall_seconds),
        },
        "data": {
            "n_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "n_features": n_features,
            "class_balance_train": {"neg": neg, "pos": pos},
        },
        "metrics": {
            "auc_train": auc_train,
            "auc_val": auc_val,
            "auc_test": auc_test,
            "auc_pr_test": auc_pr,
            "brier_test": brier,
            "sensitivity_screening": float(sens_s),
            "specificity_screening": float(spec_s),
            "f1_screening": float(f1_s),
            "accuracy_0_5": float(accuracy_score(y_test, (proba_test >= 0.5).astype(int))),
            "recall_0_5": float(recall_score(y_test, (proba_test >= 0.5).astype(int))),
            "r_target_output_test": r_corr,
        },
        "calibration_sample": {
            "target": y_cal.tolist(),
            "output": p_cal.tolist(),
            "r": r_corr,
            "fit_slope": float(slope) if slope == slope else None,
            "fit_intercept": float(intercept) if intercept == intercept else None,
            "n": int(n_cal),
        },
        "search_curve": search_curve,
        "min_observed_classification_error": float(min_err),
        "performance_goal": performance_goal,
        "status": "ok",
        "status_detail": status_detail,
    }
    TRAINING_RUN_JSON.write_text(
        json.dumps(_jsonable(run_report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    settings = {
        "train_pct": train_pct,
        "val_pct": val_pct,
        "test_pct": test_pct,
        "tune_C": tune_C,
        "C": float(fit_info.get("best_params", {}).get("clf__C", C)),
        "class_weight": class_weight,
        "n_iter_search": int(n_iter_search),
    }
    save_admin_settings(settings)
    return run_report


def load_training_run() -> dict | None:
    if not TRAINING_RUN_JSON.exists():
        return None
    return json.loads(TRAINING_RUN_JSON.read_text(encoding="utf-8"))
