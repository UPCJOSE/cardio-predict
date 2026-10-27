"""Prototipo de orientación preventiva (no diagnóstico médico)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
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
    "80 or older",
]
HEALTH_LABELS = ["Excellent", "Very good", "Good", "Fair", "Poor"]
SMOKE_LABELS = ["Never", "Former", "Current"]


@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / "xgboost.joblib"
    fallback = MODELS_DIR / "logistic.joblib"
    path = model_path if model_path.exists() else fallback
    if not path.exists():
        return None, None, None
    model = joblib.load(path)
    metrics = {}
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    # umbral de tamizaje del modelo seleccionado / xgboost
    screen_t = 0.35
    high_t = 0.70
    key = "XGBoost" if "xgboost" in path.name else "Regresión logística"
    if metrics.get("models", {}).get(key):
        screen_t = metrics["models"][key]["at_screening_threshold"]["threshold"]
        high_t = metrics["models"][key]["high_risk_threshold"]["threshold"]
    return model, screen_t, high_t


def main():
    st.set_page_config(page_title="Riesgo cardiovascular", page_icon="❤️", layout="centered")
    st.title("Estimación orientativa de riesgo cardiovascular")
    st.caption(
        "Herramienta educativa de apoyo. **No sustituye** valoración médica ni diagnóstico."
    )

    model, t_screen, t_high = load_artifacts()
    if model is None:
        st.error("No hay modelos entrenados. Ejecute: `python scripts/run_pipeline.py`")
        return

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

    if not submitted:
        st.info("Complete el formulario y pulse Estimar riesgo.")
        return

    row = {
        "AgeCategory": AGE_LABELS.index(age),
        "Sex": 1 if sex == "Hombre" else 0,
        "BMI": float(bmi),
        "GeneralHealth": HEALTH_LABELS.index(health),
        "PhysicalHealthDays": int(phys_days),
        "MentalHealthDays": int(ment_days),
        "PhysicalActivities": 1 if activity == "Sí" else 0,
        "SleepHours": int(sleep),
        "SmokerStatus": SMOKE_LABELS.index(smoke),
        "AlcoholDrinkers": 1 if alcohol == "Sí" else 0,
        "HadDiabetes": 1 if diabetes == "Sí" else 0,
        "HadStroke": 1 if stroke == "Sí" else 0,
        "HadKidneyDisease": 1 if kidney == "Sí" else 0,
        "DifficultyWalking": 1 if walking == "Sí" else 0,
    }
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
    st.metric("Probabilidad estimada", f"{prob:.1%}")
    st.write(advice)
    st.progress(min(max(prob, 0.0), 1.0))
    st.warning(
        "Resultado orientativo con base en encuesta BRFSS (EE. UU.). "
        "No es un diagnóstico clínico."
    )


if __name__ == "__main__":
    main()
