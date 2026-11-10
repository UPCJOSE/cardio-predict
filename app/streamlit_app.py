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

# Casos de demo para proyección en vivo (README).
PRESET_LOW = {
    "age": "18-24",
    "sex": "Mujer",
    "bmi": 22.0,
    "health": "Excelente",
    "phys_days": 0,
    "ment_days": 0,
    "activity": "Sí",
    "sleep": 8,
    "smoke": "Nunca",
    "alcohol": "No",
    "diabetes": "No",
    "stroke": "No",
    "kidney": "No",
    "walking": "No",
}
PRESET_HIGH = {
    "age": "80 o más",
    "sex": "Hombre",
    "bmi": 35.0,
    "health": "Mala",
    "phys_days": 20,
    "ment_days": 10,
    "activity": "No",
    "sleep": 5,
    "smoke": "Actual",
    "alcohol": "Sí",
    "diabetes": "Sí",
    "stroke": "Sí",
    "kidney": "No",
    "walking": "Sí",
}
PRESET_DEFAULT = {
    "age": "55-59",
    "sex": "Mujer",
    "bmi": 27.0,
    "health": "Buena",
    "phys_days": 0,
    "ment_days": 2,
    "activity": "Sí",
    "sleep": 7,
    "smoke": "Nunca",
    "alcohol": "No",
    "diabetes": "No",
    "stroke": "No",
    "kidney": "No",
    "walking": "No",
}

THEME_OPTIONS = ("Claro", "Oscuro", "Sistema")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

html, body, [class*="css"] {
  font-family: "DM Sans", sans-serif;
}

.stApp,
.stApp:has(.theme-light),
.stApp:has(.theme-system) {
  --cp-bg-1: rgba(19, 111, 99, 0.14);
  --cp-bg-2: rgba(70, 120, 170, 0.13);
  --cp-bg-3: #d7f0ea;
  --cp-bg-4: #d9e8f7;
  --cp-bg-base-a: #f4f8fb;
  --cp-bg-base-b: #eef3f7;
  --cp-panel: rgba(255,255,255,0.86);
  --cp-panel-border: rgba(15, 55, 70, 0.08);
  --cp-form: rgba(255,255,255,0.72);
  --cp-text: #0b3d4a;
  --cp-muted: #5b6b75;
  --cp-risk-text: #08323c;
  --cp-bajo-a: #e7f8ef;
  --cp-bajo-b: #d7f1e4;
  --cp-bajo-border: #9fd8b8;
  --cp-mod-a: #fff6e8;
  --cp-mod-b: #ffe8c7;
  --cp-mod-border: #f0c27a;
  --cp-alto-a: #fdecec;
  --cp-alto-b: #f8d0d0;
  --cp-alto-border: #e39a9a;
  --cp-menu-btn: #0b3d4a;
  --cp-menu-btn-hover: #136f63;
  --cp-menu-btn-text: #f7fcfb;
  --cp-secondary-bg: rgba(255,255,255,0.78);
  --cp-secondary-text: #0b3d4a;
  --cp-secondary-border: rgba(15, 55, 70, 0.12);
}

.stApp:has(.theme-dark) {
  --cp-bg-1: rgba(45, 180, 160, 0.12);
  --cp-bg-2: rgba(90, 140, 200, 0.12);
  --cp-bg-3: #16353c;
  --cp-bg-4: #1a2f45;
  --cp-bg-base-a: #0d171c;
  --cp-bg-base-b: #101c24;
  --cp-panel: rgba(18, 32, 40, 0.92);
  --cp-panel-border: rgba(180, 210, 220, 0.12);
  --cp-form: rgba(16, 28, 35, 0.88);
  --cp-text: #d7e8ea;
  --cp-muted: #9bb0b8;
  --cp-risk-text: #e8f3f2;
  --cp-bajo-a: #14352c;
  --cp-bajo-b: #1a4638;
  --cp-bajo-border: #3f8f6e;
  --cp-mod-a: #3a2e16;
  --cp-mod-b: #4a3918;
  --cp-mod-border: #c49a4a;
  --cp-alto-a: #3a1c1c;
  --cp-alto-b: #4a2424;
  --cp-alto-border: #c77777;
  --cp-menu-btn: #1a8a7a;
  --cp-menu-btn-hover: #23a090;
  --cp-menu-btn-text: #f7fcfb;
  --cp-secondary-bg: rgba(255,255,255,0.06);
  --cp-secondary-text: #d7e8ea;
  --cp-secondary-border: rgba(180, 210, 220, 0.18);
}

@media (prefers-color-scheme: dark) {
  .stApp:has(.theme-system) {
    --cp-bg-1: rgba(45, 180, 160, 0.12);
    --cp-bg-2: rgba(90, 140, 200, 0.12);
    --cp-bg-3: #16353c;
    --cp-bg-4: #1a2f45;
    --cp-bg-base-a: #0d171c;
    --cp-bg-base-b: #101c24;
    --cp-panel: rgba(18, 32, 40, 0.92);
    --cp-panel-border: rgba(180, 210, 220, 0.12);
    --cp-form: rgba(16, 28, 35, 0.88);
    --cp-text: #d7e8ea;
    --cp-muted: #9bb0b8;
    --cp-risk-text: #e8f3f2;
    --cp-bajo-a: #14352c;
    --cp-bajo-b: #1a4638;
    --cp-bajo-border: #3f8f6e;
    --cp-mod-a: #3a2e16;
    --cp-mod-b: #4a3918;
    --cp-mod-border: #c49a4a;
    --cp-alto-a: #3a1c1c;
    --cp-alto-b: #4a2424;
    --cp-alto-border: #c77777;
    --cp-menu-btn: #1a8a7a;
    --cp-menu-btn-hover: #23a090;
    --cp-menu-btn-text: #f7fcfb;
    --cp-secondary-bg: rgba(255,255,255,0.06);
    --cp-secondary-text: #d7e8ea;
    --cp-secondary-border: rgba(180, 210, 220, 0.18);
  }
}

.stApp {
  background:
    radial-gradient(900px 900px at -8% 18%, var(--cp-bg-1) 0%, transparent 58%),
    radial-gradient(850px 820px at 108% 22%, var(--cp-bg-2) 0%, transparent 56%),
    radial-gradient(1100px 480px at 12% -12%, var(--cp-bg-3) 0%, transparent 58%),
    radial-gradient(980px 460px at 96% -8%, var(--cp-bg-4) 0%, transparent 52%),
    linear-gradient(180deg, var(--cp-bg-base-a) 0%, var(--cp-bg-base-b) 100%);
  color: var(--cp-text);
}

.block-container {
  padding-top: 1.2rem;
  padding-bottom: 3rem;
  max-width: 1340px;
}

.hero {
  background: linear-gradient(135deg, #0b3d4a 0%, #136f63 55%, #1a8a7a 100%);
  color: #f7fcfb;
  border-radius: 22px;
  padding: 1.6rem 1.8rem 1.4rem;
  margin-bottom: 0.35rem;
  box-shadow: 0 18px 40px rgba(11, 61, 74, 0.22);
}
.hero h1 {
  font-family: "Fraunces", serif;
  font-size: 2.05rem;
  line-height: 1.15;
  margin: 0 0 0.45rem 0;
  font-weight: 700;
}
.hero p {
  margin: 0;
  font-size: 1.02rem;
  opacity: 0.92;
  max-width: 46rem;
}
.hero-note {
  margin-top: 0.85rem;
  font-size: 0.88rem;
  opacity: 0.88;
}

.cp-side-title {
  font-family: "Fraunces", serif;
  color: var(--cp-text);
  font-size: 1.2rem;
  margin: 0.45rem 0 0.15rem 0;
}
.cp-side-caption {
  color: var(--cp-muted);
  font-size: 0.88rem;
  margin-bottom: 0.85rem;
}

.panel {
  background: var(--cp-panel);
  border: 1px solid var(--cp-panel-border);
  border-radius: 18px;
  padding: 1.1rem 1.2rem 0.9rem;
  box-shadow: 0 10px 28px rgba(20, 40, 60, 0.06);
  margin-bottom: 1rem;
}
.panel h3 {
  font-family: "Fraunces", serif;
  margin: 0 0 0.35rem 0;
  color: var(--cp-text);
  font-size: 1.25rem;
}
.panel .muted {
  color: var(--cp-muted);
  font-size: 0.92rem;
  margin-bottom: 0.8rem;
}

.risk-card {
  border-radius: 20px;
  padding: 1.35rem 1.4rem 1.2rem;
  color: var(--cp-risk-text);
  margin: 0.4rem 0 1rem 0;
  border: 1px solid transparent;
}
.risk-card.bajo {
  background: linear-gradient(135deg, var(--cp-bajo-a), var(--cp-bajo-b));
  border-color: var(--cp-bajo-border);
}
.risk-card.moderado {
  background: linear-gradient(135deg, var(--cp-mod-a), var(--cp-mod-b));
  border-color: var(--cp-mod-border);
}
.risk-card.alto {
  background: linear-gradient(135deg, var(--cp-alto-a), var(--cp-alto-b));
  border-color: var(--cp-alto-border);
}
.risk-label {
  font-size: 0.8rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.75;
  margin-bottom: 0.25rem;
}
.risk-level {
  font-family: "Fraunces", serif;
  font-size: 2.4rem;
  line-height: 1;
  margin: 0 0 0.35rem 0;
}
.risk-prob {
  font-size: 1.55rem;
  font-weight: 700;
  margin: 0.15rem 0 0.55rem 0;
}
.risk-advice {
  font-size: 1.02rem;
  margin: 0;
}

.section-title {
  font-family: "Fraunces", serif;
  color: var(--cp-text);
  font-size: 1.05rem;
  margin: 0.2rem 0 0.55rem 0;
}

div[data-testid="stForm"] {
  background: var(--cp-form);
  border: 1px solid var(--cp-panel-border);
  border-radius: 18px;
  padding: 0.85rem 1rem 0.4rem;
}

.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] button {
  background: #136f63 !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
  border-radius: 12px !important;
  padding: 0.55rem 1rem !important;
}
.stButton > button {
  border-radius: 12px !important;
}
.stButton > button[kind="secondary"] {
  background: var(--cp-secondary-bg) !important;
  color: var(--cp-secondary-text) !important;
  border: 1px solid var(--cp-secondary-border) !important;
}

.stApp:has(.theme-dark) [data-testid="stMarkdownContainer"],
.stApp:has(.theme-dark) label,
.stApp:has(.theme-dark) p,
.stApp:has(.theme-dark) span {
  color: var(--cp-text);
}
@media (prefers-color-scheme: dark) {
  .stApp:has(.theme-system) [data-testid="stMarkdownContainer"],
  .stApp:has(.theme-system) label,
  .stApp:has(.theme-system) p,
  .stApp:has(.theme-system) span {
    color: var(--cp-text);
  }
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {display: none !important;}
section[data-testid="stSidebar"] {display: none !important;}
[data-testid="collapsedControl"] {display: none !important;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}
.stAppDeployButton, .stDeployButton {display: none !important;}
</style>
"""


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


def init_form_state(default_model: str | None = None) -> None:
    """Inicializa keys del formulario y de la barra lateral una sola vez."""
    for key, value in PRESET_DEFAULT.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "Claro"
    if "sidebar_open" not in st.session_state:
        st.session_state["sidebar_open"] = True
    if default_model and "model_choice" not in st.session_state:
        st.session_state["model_choice"] = default_model


def apply_preset(preset: dict, preset_name: str) -> None:
    """Carga un caso de demo y calcula el resultado al instante."""
    for key, value in preset.items():
        st.session_state[key] = value
    st.session_state["last_row"] = form_to_row(preset)
    st.session_state["active_preset"] = preset_name


def theme_class(theme_mode: str) -> str:
    """Clase CSS según el modo elegido (Claro / Oscuro / Sistema)."""
    mapping = {"Claro": "theme-light", "Oscuro": "theme-dark", "Sistema": "theme-system"}
    return mapping.get(theme_mode, "theme-light")


def apply_theme(theme_mode: str) -> None:
    """Inyecta CSS base + marcador de tema para variables y modo sistema."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="{theme_class(theme_mode)}" style="display:none"></div>',
        unsafe_allow_html=True,
    )


def toggle_sidebar() -> None:
    """Abre o cierra la barra lateral de presentación."""
    st.session_state["sidebar_open"] = not st.session_state.get("sidebar_open", True)


def render_hero() -> None:
    st.markdown(
        """
<div class="hero">
  <h1>Cardio Predict</h1>
  <p>
    Estimación orientativa de riesgo cardiovascular a partir de 14 variables de encuesta,
    comparando regresión logística y XGBoost.
  </p>
  <div class="hero-note">
    Herramienta educativa de apoyo. No sustituye valoración médica ni diagnóstico.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls(
    names: list[str],
    metrics: dict,
    selected: str,
    reason: str,
) -> tuple[str, float, float]:
    """Controles de presentación dentro de la barra lateral."""
    st.markdown(
        '<div class="cp-side-title">Presentación</div>'
        '<div class="cp-side-caption">Tema, modelo y datos de la demo en vivo.</div>',
        unsafe_allow_html=True,
    )
    st.radio(
        "Apariencia",
        THEME_OPTIONS,
        key="theme_mode",
        horizontal=True,
        help="Claro, Oscuro o según el sistema (como en Deploy).",
    )
    st.markdown("**Modelo para esta estimación**")
    model_name = st.radio(
        "Modelo",
        names,
        key="model_choice",
        label_visibility="collapsed",
        help="Por defecto se usa el modelo seleccionado en la evaluación.",
    )
    t_screen, t_high = thresholds_for(metrics, model_name)
    model_metrics = metrics.get("models", {}).get(model_name, {})
    if model_metrics:
        st.metric("AUC-ROC (prueba)", f"{model_metrics['auc_roc']:.3f}")
        st.caption(f"Umbral de tamizaje {t_screen:.2f} · umbral alto {t_high:.2f}")
    with st.expander("¿Por qué este modelo?", expanded=False):
        st.write(f"**Operativo del proyecto:** {selected}")
        if reason:
            st.write(reason)
        st.caption(
            "Partición actual del experimento: 70 % train / 15 % val / 15 % test."
        )
    return model_name, t_screen, t_high


def render_estimator(
    model,
    model_name: str,
    t_screen: float,
    t_high: float,
) -> None:
    """Formulario de perfil + panel de resultado."""
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            """
<div class="panel">
  <h3>Perfil de la persona</h3>
  <div class="muted">Complete las 14 variables o use un caso de demostración.</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        active_preset = st.session_state.get("active_preset")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button(
                "Caso bajo",
                use_container_width=True,
                type="primary" if active_preset == "bajo" else "secondary",
            ):
                apply_preset(PRESET_LOW, "bajo")
                st.rerun()
        with b2:
            if st.button(
                "Caso típico",
                use_container_width=True,
                type="primary" if active_preset == "tipico" else "secondary",
            ):
                apply_preset(PRESET_DEFAULT, "tipico")
                st.rerun()
        with b3:
            if st.button(
                "Caso alto",
                use_container_width=True,
                type="primary" if active_preset == "alto" else "secondary",
            ):
                apply_preset(PRESET_HIGH, "alto")
                st.rerun()

        with st.form("risk_form"):
            st.markdown(
                '<div class="section-title">Demografía y salud percibida</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.selectbox("Edad", AGE_LABELS, key="age")
                st.selectbox("Sexo", ["Mujer", "Hombre"], key="sex")
                st.number_input(
                    "IMC",
                    min_value=12.0,
                    max_value=60.0,
                    step=0.1,
                    key="bmi",
                )
            with c2:
                st.selectbox("Salud general percibida", HEALTH_LABELS, key="health")
                st.selectbox(
                    "Actividad física el último mes", ["Sí", "No"], key="activity"
                )
                st.slider("Horas de sueño promedio", 1, 24, key="sleep")

            st.markdown(
                '<div class="section-title">Hábitos y bienestar reciente</div>',
                unsafe_allow_html=True,
            )
            c3, c4 = st.columns(2)
            with c3:
                st.slider(
                    "Días de mala salud física (último mes)", 0, 30, key="phys_days"
                )
                st.slider(
                    "Días de mala salud mental (último mes)", 0, 30, key="ment_days"
                )
                st.selectbox("Tabaquismo", SMOKE_LABELS, key="smoke")
            with c4:
                st.selectbox(
                    "Consumió alcohol el último mes", ["No", "Sí"], key="alcohol"
                )
                st.selectbox("Diagnóstico de diabetes", ["No", "Sí"], key="diabetes")
                st.selectbox("Antecedente de ACV", ["No", "Sí"], key="stroke")

            st.markdown(
                '<div class="section-title">Antecedentes</div>',
                unsafe_allow_html=True,
            )
            c5, c6 = st.columns(2)
            with c5:
                st.selectbox("Enfermedad renal", ["No", "Sí"], key="kidney")
            with c6:
                st.selectbox(
                    "Dificultad para caminar/escaleras", ["No", "Sí"], key="walking"
                )

            submitted = st.form_submit_button(
                "Estimar riesgo",
                use_container_width=True,
                type="primary",
            )

        if submitted:
            st.session_state["last_row"] = form_to_row(
                {key: st.session_state[key] for key in PRESET_DEFAULT}
            )
            st.session_state["active_preset"] = None

    with right:
        st.markdown(
            """
<div class="panel">
  <h3>Resultado</h3>
  <div class="muted">Se actualiza al estimar o al cargar un caso de demo.</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        row = st.session_state.get("last_row")
        if row is None:
            st.info(
                "Pulse **Estimar riesgo** o cargue un **caso de demostración** "
                "para ver el nivel Bajo / Moderado / Alto."
            )
            return

        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        prob = float(model.predict_proba(X)[0, 1])

        if prob < t_screen:
            level, advice = (
                "Bajo",
                "Mantener hábitos saludables y control anual.",
            )
        elif prob < t_high:
            level, advice = (
                "Moderado",
                "Tomar la presión arterial y consultar en el puesto de salud.",
            )
        else:
            level, advice = (
                "Alto",
                "Remisión prioritaria a consulta médica.",
            )

        render_result(level, prob, advice, model_name)

        contrib = local_contributions(model, X)
        if contrib is not None:
            st.markdown("**Factores que más pesan en esta persona**")
            st.dataframe(contrib.head(6), hide_index=True, width="stretch")
        else:
            importance_path = REPORTS_DIR / "xgboost_importance.json"
            if importance_path.exists():
                imp = pd.read_json(importance_path)
                imp["Factor"] = imp["feature"].map(FEATURE_LABELS)
                st.markdown("**Importancia global de XGBoost**")
                st.caption("No es explicación caso a caso.")
                st.bar_chart(imp.set_index("Factor")["importance"].head(8))

        st.warning(
            "Resultado orientativo con base en encuesta BRFSS (EE. UU.). "
            "No es un diagnóstico clínico."
        )


def render_result(level: str, prob: float, advice: str, model_name: str) -> None:
    css_level = level.lower()
    st.markdown(
        f"""
<div class="risk-card {css_level}">
  <div class="risk-label">Nivel de riesgo estimado</div>
  <div class="risk-level">{level}</div>
  <div class="risk-prob">{prob:.1%} de probabilidad</div>
  <p class="risk-advice">{advice}</p>
  <p class="risk-advice" style="opacity:0.75;margin-top:0.55rem;font-size:0.92rem;">
    Modelo usado: {model_name}
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(prob, 0.0), 1.0))


def main():
    """Pantalla: formulario → predict_proba → Bajo / Moderado / Alto."""
    st.set_page_config(
        page_title="Cardio Predict · UPC",
        page_icon="❤",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_form_state()
    apply_theme(st.session_state["theme_mode"])

    models, metrics, selected, load_errors = load_artifacts()
    if not models:
        st.error("No hay modelos entrenados. Ejecute: `python scripts/run_pipeline.py`")
        for name, err in load_errors.items():
            st.caption(f"{name}: {err}")
        return
    if load_errors:
        missing = ", ".join(load_errors)
        st.warning(
            f"No se pudo cargar: {missing}. La demo sigue con el modelo disponible."
        )

    reason = metrics.get("selection", {}).get("reason", "")
    names = list(models.keys())
    if selected not in names:
        selected = names[0]
    if "model_choice" not in st.session_state or st.session_state["model_choice"] not in names:
        st.session_state["model_choice"] = selected

    sidebar_open = st.session_state.get("sidebar_open", True)

    if sidebar_open:
        side, main_col = st.columns([0.24, 0.76], gap="large")
        with side:
            st.button(
                "☰",
                key="sidebar_toggle_open",
                help="Ocultar barra lateral",
                on_click=toggle_sidebar,
            )
            model_name, t_screen, t_high = render_sidebar_controls(
                names, metrics, selected, reason
            )
        with main_col:
            render_hero()
            render_estimator(models[model_name], model_name, t_screen, t_high)
    else:
        toggle_col, _ = st.columns([0.06, 0.94])
        with toggle_col:
            st.button(
                "☰",
                key="sidebar_toggle_closed",
                help="Mostrar barra lateral",
                on_click=toggle_sidebar,
            )
        model_name = st.session_state["model_choice"]
        t_screen, t_high = thresholds_for(metrics, model_name)
        render_hero()
        render_estimator(models[model_name], model_name, t_screen, t_high)


if __name__ == "__main__":
    main()
