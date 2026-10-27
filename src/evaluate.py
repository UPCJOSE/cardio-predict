"""Evaluación cuantitativa y comparación de modelos."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    DATA_PROCESSED,
    FIGURES_DIR,
    MIN_HIGH_SPECIFICITY,
    MIN_SENSITIVITY,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
)


def load_test():
    test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
    y = test[TARGET_COLUMN]
    X = test.drop(columns=[TARGET_COLUMN])
    return X, y


def bootstrap_auc(y_true, y_prob, n_boot: int = 200, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def best_threshold_for_sensitivity(y_true, y_prob, min_sens: float = MIN_SENSITIVITY):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve thresholds has len = len(recalls)-1
    best = None
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, pred, zero_division=0)
        spec = recall_score(y_true, pred, pos_label=0, zero_division=0)
        if sens >= min_sens:
            f1 = f1_score(y_true, pred, zero_division=0)
            candidate = (t, sens, spec, f1)
            if best is None or candidate[3] > best[3]:
                best = candidate
    if best is None:
        # fallback: maximize F1
        best_f1, best_t = -1, 0.5
        for t in np.linspace(0.01, 0.99, 99):
            pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        pred = (y_prob >= best_t).astype(int)
        return best_t, recall_score(y_true, pred), recall_score(
            y_true, pred, pos_label=0
        ), best_f1
    return best


def high_risk_threshold(y_true, y_prob, min_spec: float = MIN_HIGH_SPECIFICITY):
    best = None
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, pred, zero_division=0)
        spec = recall_score(y_true, pred, pos_label=0, zero_division=0)
        if spec >= min_spec:
            candidate = (t, sens, spec)
            if best is None or candidate[1] > best[1]:
                best = candidate
    return best if best else (0.8, 0.0, 0.0)


def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_roc(curves: dict, path: Path):
    plt.figure(figsize=(7, 5))
    for name, (fpr, tpr, auc) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("1 - Especificidad (FPR)")
    plt.ylabel("Sensibilidad (TPR)")
    plt.title("Curvas ROC — conjunto de prueba")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr(curves: dict, path: Path):
    plt.figure(figsize=(7, 5))
    for name, (rec, prec, ap) in curves.items():
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Sensibilidad (Recall)")
    plt.ylabel("Precisión")
    plt.title("Curvas Precisión-Sensibilidad — conjunto de prueba")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison_bars(results: dict, path: Path):
    models = list(results.keys())
    metrics = ["auc_roc", "auc_pr", "f1", "recall_sensitivity", "specificity"]
    labels = ["AUC-ROC", "AUC-PR", "F1", "Sensibilidad", "Especificidad"]
    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(9, 5))
    for i, m in enumerate(models):
        vals = [
            results[m]["auc_roc"]
            if k == "auc_roc"
            else results[m]["auc_pr"]
            if k == "auc_pr"
            else results[m]["at_screening_threshold"][k]
            for k in metrics
        ]
        plt.bar(x + i * width, vals, width, label=m)
    plt.xticks(x + width / 2, labels, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Valor")
    plt.title("Comparación de modelos (umbral de tamizaje)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run() -> dict:
    X, y = load_test()
    log_model = joblib.load(MODELS_DIR / "logistic.joblib")
    xgb_model = joblib.load(MODELS_DIR / "xgboost.joblib")

    probs = {
        "Regresión logística": log_model.predict_proba(X)[:, 1],
        "XGBoost": xgb_model.predict_proba(X)[:, 1],
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    roc_curves = {}
    pr_curves = {}
    results = {}

    for name, prob in probs.items():
        auc = float(roc_auc_score(y, prob))
        ap = float(average_precision_score(y, prob))
        brier = float(brier_score_loss(y, prob))
        auc_mean, auc_lo, auc_hi = bootstrap_auc(y, prob)

        t_screen, sens, spec, f1 = best_threshold_for_sensitivity(y, prob)
        t_high, sens_h, spec_h = high_risk_threshold(y, prob)
        screen_metrics = metrics_at_threshold(y, prob, t_screen)
        # default 0.5 for reference
        default_metrics = metrics_at_threshold(y, prob, 0.5)

        fpr, tpr, _ = roc_curve(y, prob)
        prec, rec, _ = precision_recall_curve(y, prob)
        roc_curves[name] = (fpr, tpr, auc)
        pr_curves[name] = (rec, prec, ap)

        results[name] = {
            "auc_roc": auc,
            "auc_roc_bootstrap_mean": auc_mean,
            "auc_roc_ci95": [auc_lo, auc_hi],
            "auc_pr": ap,
            "brier": brier,
            "at_0_5": default_metrics,
            "at_screening_threshold": screen_metrics,
            "high_risk_threshold": {
                "threshold": float(t_high),
                "sensitivity": float(sens_h),
                "specificity": float(spec_h),
            },
        }

    plot_roc(roc_curves, FIGURES_DIR / "roc_curves.png")
    plot_pr(pr_curves, FIGURES_DIR / "pr_curves.png")
    plot_comparison_bars(results, FIGURES_DIR / "model_comparison.png")

    # Criterio de selección de la metodología.
    auc_log = results["Regresión logística"]["auc_roc"]
    auc_xgb = results["XGBoost"]["auc_roc"]
    if auc_xgb - auc_log < 0.01:
        selected = "Regresión logística"
        reason = (
            "Diferencia de AUC-ROC < 0.01; se prefiere la logística por interpretabilidad."
        )
    else:
        selected = "XGBoost" if auc_xgb >= auc_log else "Regresión logística"
        reason = "Mayor AUC-ROC en el conjunto de prueba."

    report = {
        "n_test": int(len(y)),
        "positive_rate_test": float(y.mean()),
        "models": results,
        "selection": {"model": selected, "reason": reason},
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Tabla markdown breve
    lines = [
        "| Modelo | AUC-ROC | IC95% | AUC-PR | F1 (tamizaje) | Sens. | Espec. | Brier |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        s = r["at_screening_threshold"]
        lo, hi = r["auc_roc_ci95"]
        lines.append(
            f"| {name} | {r['auc_roc']:.4f} | [{lo:.4f}, {hi:.4f}] | "
            f"{r['auc_pr']:.4f} | {s['f1']:.4f} | {s['recall_sensitivity']:.4f} | "
            f"{s['specificity']:.4f} | {r['brier']:.4f} |"
        )
    lines.append("")
    lines.append(f"**Modelo seleccionado:** {selected}. {reason}")
    (REPORTS_DIR / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    run()
