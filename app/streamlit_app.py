"""Prototipo Streamlit: 14 preguntas → probabilidad y nivel de riesgo.

Carga models/*.joblib (no reentrena). Por defecto usa la logística, que fue
el modelo operativo según reports/metrics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FEATURE_COLUMNS, MODELS_DIR, REPORTS_DIR  # noqa: E402

AGE_LABELS = [
    "18-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80 o más",
]
HEALTH_LABELS = ["Excelente", "Muy buena", "Buena", "Regular", "Mala"]
SMOKE_LABELS = ["Nunca", "Exfumador", "Actual"]

FEATURE_LABELS = {
    "AgeCategory": "Edad",
    "Sex": "Sexo (hombre)",
    "BMI": "IMC",
    "GeneralHealth": "Salud percibida",
    "PhysicalHealthDays": "Días de mala salud física",
    "MentalHealthDays": "Días de mala salud mental",
    "PhysicalActivities": "Actividad física",
    "SleepHours": "Horas de sueño",
    "SmokerStatus": "Tabaquismo",
    "AlcoholDrinkers": "Consumo de alcohol",
    "HadDiabetes": "Diabetes",
    "HadStroke": "Antecedente de ACV",
    "HadKidneyDisease": "Enfermedad renal",
    "DifficultyWalking": "Dificultad para caminar",
}

MODEL_FILES = {
    "Regresión logística": MODELS_DIR / "logistic.joblib",
    "XGBoost": MODELS_DIR / "xgboost.joblib",
}


@st.cache_resource
def load_artifacts():
    """Carga .joblib + metrics.json una sola vez (caché de Streamlit)."""
    metrics = {}
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    models = {}
    load_errors = {}
    for name, path in MODEL_FILES.items():
        if not path.exists():
            load_errors[name] = f"No se encontró {path.name}"
            continue
        try:
            models[name] = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 — un .joblib ajeno no debe tumbar la demo
            load_errors[name] = str(exc)

    selected = metrics.get("selection", {}).get("model", "Regresión logística")
    if selected not in models:
        selected = next(iter(models), None)
    return models, metrics, selected, load_errors


def thresholds_for(metrics: dict, model_name: str) -> tuple[float, float]:
    """Devuelve (umbral tamizaje, umbral alto) del modelo elegido."""
    block = metrics.get("models", {}).get(model_name, {})
    screen_t = block.get("at_screening_threshold", {}).get("threshold", 0.47)
    high_t = block.get("high_risk_threshold", {}).get("threshold", 0.71)
    return float(screen_t), float(high_t)


def local_contributions(model, X: pd.DataFrame) -> pd.DataFrame | None:
    """Aporte de cada variable en esta persona (coeficiente × valor estandarizado).

    Solo aplica a la logística (tiene scaler + clf). XGBoost no usa este cálculo.
    """
    if not hasattr(model, "named_steps"):
        return None
    if "scaler" not in model.named_steps or "clf" not in model.named_steps:
        return None
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    scaled = scaler.transform(X)
    contrib = clf.coef_[0] * scaled[0]
    table = pd.DataFrame(
        {
            "Factor": [FEATURE_LABELS[c] for c in FEATURE_COLUMNS],
            "Aporte": contrib,
        }
    )
    table["Dirección"] = np.where(table["Aporte"] >= 0, "↑ riesgo", "↓ riesgo")
    return table.sort_values("Aporte", key=np.abs, ascending=False)


def form_to_row(values: dict) -> dict:
    """Pasa las respuestas del formulario a los 14 números que espera el modelo."""
    return {
        "AgeCategory": AGE_LABELS.index(values["age"]),
        "Sex": 1 if values["sex"] == "Hombre" else 0,
        "BMI": float(values["bmi"]),
        "GeneralHealth": HEALTH_LABELS.index(values["health"]),
        "PhysicalHealthDays": int(values["phys_days"]),
        "MentalHealthDays": int(values["ment_days"]),
        "PhysicalActivities": 1 if values["activity"] == "Sí" else 0,
        "SleepHours": int(values["sleep"]),
        "SmokerStatus": SMOKE_LABELS.index(values["smoke"]),
        "AlcoholDrinkers": 1 if values["alcohol"] == "Sí" else 0,
        "HadDiabetes": 1 if values["diabetes"] == "Sí" else 0,
        "HadStroke": 1 if values["stroke"] == "Sí" else 0,
        "HadKidneyDisease": 1 if values["kidney"] == "Sí" else 0,
        "DifficultyWalking": 1 if values["walking"] == "Sí" else 0,
    }


def main():
    """Pantalla: formulario → predict_proba → Bajo / Moderado / Alto."""
    st.set_page_config(page_title="Riesgo cardiovascular", layout="centered")
    st.title("Estimación orientativa de riesgo cardiovascular")
    st.caption(
        "Herramienta educativa de apoyo. **No sustituye** valoración médica ni diagnóstico."
    )

    models, metrics, selected, load_errors = load_artifacts()
    if not models:
        st.error("No hay modelos entrenados. Ejecute: `python scripts/run_pipeline.py`")
        for name, err in load_errors.items():
            st.caption(f"{name}: {err}")
        return
    if load_errors:
        missing = ", ".join(load_errors)
        st.sidebar.warning(
            f"No se pudo cargar: {missing}. La demo sigue con el modelo disponible."
        )

    reason = metrics.get("selection", {}).get("reason", "")
    st.info(f"Modelo operativo del proyecto: **{selected}**. {reason}")

    names = list(models.keys())
    default_idx = names.index(selected) if selected in names else 0
    model_name = st.sidebar.radio(
        "Modelo para esta estimación",
        names,
        index=default_idx,
        help="Por defecto se usa el modelo seleccionado en la evaluación (regresión logística).",
    )
    model = models[model_name]
    t_screen, t_high = thresholds_for(metrics, model_name)

    model_metrics = metrics.get("models", {}).get(model_name, {})
    if model_metrics:
        st.sidebar.metric("AUC-ROC (prueba)", f"{model_metrics['auc_roc']:.3f}")
        st.sidebar.caption(
            f"Umbral de tamizaje {t_screen:.2f} · umbral alto {t_high:.2f}"
        )

    with st.form("risk_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.selectbox("Edad", AGE_LABELS, index=7)
            sex = st.selectbox("Sexo", ["Mujer", "Hombre"])
            bmi = st.number_input("IMC", min_value=12.0, max_value=60.0, value=27.0, step=0.1)
            health = st.selectbox("Salud general percibida", HEALTH_LABELS, index=2)
            phys_days = st.slider("Días de mala salud física (último mes)", 0, 30, 0)
            ment_days = st.slider("Días de mala salud mental (último mes)", 0, 30, 2)
            activity = st.selectbox("Actividad física el último mes", ["Sí", "No"])
        with c2:
            sleep = st.slider("Horas de sueño promedio", 1, 24, 7)
            smoke = st.selectbox("Tabaquismo", SMOKE_LABELS)
            alcohol = st.selectbox("Consumió alcohol el último mes", ["No", "Sí"])
            diabetes = st.selectbox("Diagnóstico de diabetes", ["No", "Sí"])
            stroke = st.selectbox("Antecedente de ACV", ["No", "Sí"])
            kidney = st.selectbox("Enfermedad renal", ["No", "Sí"])
            walking = st.selectbox("Dificultad para caminar/escaleras", ["No", "Sí"])
        submitted = st.form_submit_button("Estimar riesgo")

    if submitted:
        st.session_state["last_row"] = form_to_row(
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "health": health,
                "phys_days": phys_days,
                "ment_days": ment_days,
                "activity": activity,
                "sleep": sleep,
                "smoke": smoke,
                "alcohol": alcohol,
                "diabetes": diabetes,
                "stroke": stroke,
                "kidney": kidney,
                "walking": walking,
            }
        )

    row = st.session_state.get("last_row")
    if row is None:
        st.info("Complete el formulario y pulse Estimar riesgo.")
        return

    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    prob = float(model.predict_proba(X)[0, 1])

    if prob < t_screen:
        level, color, advice = (
            "Bajo",
            "green",
            "Mantener hábitos saludables y control anual.",
        )
    elif prob < t_high:
        level, color, advice = (
            "Moderado",
            "orange",
            "Tomar la presión arterial y consultar en el puesto de salud.",
        )
    else:
        level, color, advice = (
            "Alto",
            "red",
            "Remisión prioritaria a consulta médica.",
        )

    st.subheader(f"Nivel de riesgo: :{color}[{level}]")
    st.caption(f"Estimación con **{model_name}**")
    st.metric("Probabilidad estimada", f"{prob:.1%}")
    st.write(advice)
    st.progress(min(max(prob, 0.0), 1.0))

    contrib = local_contributions(model, X)
    if contrib is not None:
        st.markdown("**Factores que más pesan en esta persona** (regresión logística)")
        st.dataframe(contrib.head(6), hide_index=True, width="stretch")
    else:
        importance_path = REPORTS_DIR / "xgboost_importance.json"
        if importance_path.exists():
            imp = pd.read_json(importance_path)
            imp["Factor"] = imp["feature"].map(FEATURE_LABELS)
            st.markdown("**Importancia global de XGBoost** (no es explicación caso a caso)")
            st.bar_chart(imp.set_index("Factor")["importance"].head(8))

    st.warning(
        "Resultado orientativo con base en encuesta BRFSS (EE. UU.). "
        "No es un diagnóstico clínico."
    )


if __name__ == "__main__":
    main()
