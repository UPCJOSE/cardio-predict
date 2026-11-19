"""Panel Administrador: datos externos, splits, reentrenamiento logística y dashboard."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.admin_auth import (
    change_password,
    current_username,
    verify_login,
)
from src.config import (
    ACTIVE_DATASET_JSON,
    DATA_PROCESSED,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.dataset_validation import (
    active_dataset_bytes_json,
    load_active_dataset,
    save_active_dataset,
    validate_upload,
)
from src.logistic_retrain import (
    load_admin_settings,
    load_training_run,
    retrain_logistic_from_frame,
    save_admin_settings,
)


def ensure_seed_active_dataset() -> None:
    """Si no hay dataset activo, intenta armarlo desde train/val/test existentes."""
    if load_active_dataset() is not None:
        return
    parts = []
    for name in ("train", "val", "test"):
        path = DATA_PROCESSED / f"{name}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    if not set(cols).issubset(df.columns):
        return
    save_active_dataset(df[cols])


def render_login() -> bool:
    """Formulario de acceso Admin. True si la sesión ya está autenticada."""
    if st.session_state.get("admin_authenticated"):
        return True

    st.markdown("### Acceso administrador")
    st.caption("Gestión de datos y reentrenamiento de la regresión logística.")
    with st.form("admin_login"):
        user = st.text_input("Usuario", value="")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Entrar", type="primary")
    if submitted:
        if verify_login(user, password):
            st.session_state["admin_authenticated"] = True
            st.rerun()
        st.error("Usuario o contraseña incorrectos.")
    st.info(
        f"Usuario actual: `{current_username()}`. "
        "La primera vez la clave por defecto es `upc-cardio` "
        "(cámbiela en el panel tras entrar)."
    )
    return False


def _render_password_section() -> None:
    st.markdown("#### Seguridad")
    st.caption(f"Usuario administrador: **{current_username()}**")
    with st.form("change_admin_password"):
        current = st.text_input("Contraseña actual", type="password")
        new_user = st.text_input(
            "Nuevo usuario (opcional)",
            value=current_username(),
            help="Deje el mismo nombre si solo quiere cambiar la clave.",
        )
        new_pass = st.text_input("Nueva contraseña", type="password")
        new_pass2 = st.text_input("Confirmar nueva contraseña", type="password")
        submitted = st.form_submit_button("Guardar credenciales", type="primary")
    if submitted:
        if new_pass != new_pass2:
            st.error("La confirmación no coincide con la nueva contraseña.")
        else:
            ok, message = change_password(current, new_pass, new_username=new_user)
            if ok:
                st.success(message)
            else:
                st.error(message)


def _render_dataset_section() -> pd.DataFrame | None:
    st.markdown("#### 1. Base de datos externa")
    st.caption(
        "Suba CSV/JSON en formato BRFSS cleaned o canónico "
        f"({len(FEATURE_COLUMNS)} variables + `{TARGET_COLUMN}`)."
    )
    uploaded = st.file_uploader(
        "Archivo nuevo",
        type=["csv", "json"],
        help="Se valida antes de activarlo para reentrenar.",
    )
    if uploaded is not None:
        result = validate_upload(uploaded.name, uploaded.getvalue())
        if result.issues:
            st.warning(f"Se detectaron {len(result.issues)} aviso(s)/error(es).")
            st.dataframe(result.issues_frame(), hide_index=True, width="stretch")
        if result.ok and result.df is not None:
            st.success(
                f"Validación OK · esquema `{result.schema}` · "
                f"{result.summary.get('rows_valid')} filas útiles."
            )
            if st.button("Activar este dataset", type="primary"):
                info = save_active_dataset(result.df)
                st.session_state["last_validation_summary"] = result.summary
                st.success(
                    f"Dataset activo guardado ({info['n_rows']} filas). "
                    f"JSON: `{Path(info['json']).name}`"
                )
                st.rerun()
        elif not result.ok:
            st.error("El archivo no se puede usar para reentrenar hasta corregir los errores.")

    active = load_active_dataset()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Filas activas", "—" if active is None else f"{len(active):,}")
    with c2:
        rate = None if active is None else float(active[TARGET_COLUMN].mean())
        st.metric("Tasa positiva", "—" if rate is None else f"{rate:.1%}")
    with c3:
        st.metric("Archivo JSON", "sí" if ACTIVE_DATASET_JSON.exists() else "no")

    payload = active_dataset_bytes_json()
    st.download_button(
        "Descargar dataset activo (JSON)",
        data=payload if payload else b"{}",
        file_name="active_dataset.json",
        mime="application/json",
        disabled=payload is None,
        use_container_width=True,
    )
    return active


def _render_split_and_params() -> dict:
    st.markdown("#### 2. Partición e hiperparámetros (logística)")
    settings = load_admin_settings()
    train_pct = st.number_input(
        "% entrenamiento",
        min_value=1,
        max_value=98,
        value=int(settings.get("train_pct", 70)),
        step=1,
    )
    val_pct = st.number_input(
        "% validación",
        min_value=1,
        max_value=98,
        value=int(settings.get("val_pct", 15)),
        step=1,
    )
    test_pct = st.number_input(
        "% prueba",
        min_value=1,
        max_value=98,
        value=int(settings.get("test_pct", 15)),
        step=1,
    )
    total = int(train_pct + val_pct + test_pct)
    if total != 100:
        st.error(f"La suma actual es {total}%. Debe ser exactamente 100%.")
    else:
        st.caption(f"Partición válida: {train_pct}-{val_pct}-{test_pct}.")

    tune_C = st.checkbox(
        "Buscar C automáticamente",
        value=bool(settings.get("tune_C", True)),
        help=(
            "C controla la regularización de la logística. "
            "Si está activo, prueba varios C en validación y elige el de mejor AUC. "
            "Si está apagado, usa el C manual de abajo."
        ),
    )
    C = st.number_input(
        "C (si no se busca automáticamente)",
        min_value=0.001,
        max_value=100.0,
        value=float(settings.get("C", 1.0)),
        format="%.4f",
        disabled=tune_C,
    )
    class_weight = st.selectbox(
        "Pesos de clase",
        options=["balanced", "None"],
        index=0 if settings.get("class_weight", "balanced") == "balanced" else 1,
        help="'balanced' compensa el desbalance; None trata ambas clases igual.",
    )
    n_iter = st.slider(
        "Intentos de búsqueda de C",
        min_value=3,
        max_value=20,
        value=int(settings.get("n_iter_search", 12)),
        disabled=not tune_C,
    )

    params = {
        "train_pct": int(train_pct),
        "val_pct": int(val_pct),
        "test_pct": int(test_pct),
        "tune_C": bool(tune_C),
        "C": float(C),
        "class_weight": "balanced" if class_weight == "balanced" else None,
        "n_iter_search": int(n_iter),
        "split_ok": total == 100,
    }
    # Guarda preferencias aunque aún no entrene.
    if st.button("Guardar configuración", use_container_width=True):
        save_admin_settings(
            {k: v for k, v in params.items() if k != "split_ok"}
        )
        st.success("Configuración guardada.")
    return params


def _render_train(active: pd.DataFrame | None, params: dict) -> None:
    st.markdown("#### 3. Reentrenar regresión logística")
    disabled = active is None or not params["split_ok"]
    if active is None:
        st.warning("Active un dataset válido antes de entrenar.")
    if st.button(
        "Entrenar ahora",
        type="primary",
        disabled=disabled,
        use_container_width=True,
    ):
        with st.spinner("Entrenando regresión logística…"):
            try:
                report = retrain_logistic_from_frame(
                    active,
                    train_pct=params["train_pct"],
                    val_pct=params["val_pct"],
                    test_pct=params["test_pct"],
                    tune_C=params["tune_C"],
                    C=params["C"],
                    class_weight=params["class_weight"],
                    n_iter_search=params["n_iter_search"],
                )
                st.session_state["last_training_run"] = report
                # Fuerza recarga de modelos cacheados en la vista cliente.
                st.cache_resource.clear()
                st.success(
                    "Entrenamiento terminado. "
                    f"Tiempo total: {report['timing']['total']['human']}."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falló el entrenamiento: {exc}")


def _status_es(detail: str | None, met: bool) -> str:
    """Traduce mensajes de estado (incluye corridas guardadas en inglés)."""
    text = (detail or "").strip()
    mapping = {
        "Training finished: Met performance criterion": (
            "Entrenamiento terminado: se cumplió el criterio de desempeño"
        ),
        "Training finished: Search completed (goal not met)": (
            "Entrenamiento terminado: búsqueda finalizada (meta no alcanzada)"
        ),
        "Training finished: Search completed": (
            "Entrenamiento terminado: búsqueda finalizada"
        ),
        "Training finished": "Entrenamiento terminado",
    }
    if text in mapping:
        return mapping[text]
    if text:
        return text
    return (
        "Entrenamiento terminado: se cumplió el criterio de desempeño"
        if met
        else "Entrenamiento terminado: búsqueda finalizada"
    )


def _architecture_html(report: dict) -> str:
    arch = report.get("architecture") or {}
    n_in = int(arch.get("n_inputs") or len(FEATURE_COLUMNS))
    return f"""
<div style="border:1px solid #9aa5b1;border-radius:8px;padding:12px 14px;
background:linear-gradient(#f7fafc,#eef2f6);font-family:Segoe UI,sans-serif;">
  <div style="font-weight:700;margin-bottom:10px;color:#1f2933;">
    Regresión logística (vista)
  </div>
  <div style="display:flex;flex-direction:column;align-items:center;gap:8px;">
    <div style="background:#2f9e44;color:#fff;padding:8px 18px;border-radius:6px;
    min-width:160px;text-align:center;font-weight:600;">
      Entrada<br/><span style="font-size:1.1rem;">{n_in}</span>
    </div>
    <div style="color:#52606d;">↓</div>
    <div style="background:#fff;border:1px solid #7b8794;padding:10px 14px;border-radius:6px;
    min-width:220px;text-align:center;">
      <div style="font-size:0.85rem;color:#52606d;">W · b → estandarización</div>
      <strong>StandardScaler</strong>
      <div style="font-size:0.85rem;margin-top:4px;">{n_in} variables</div>
    </div>
    <div style="color:#52606d;">↓</div>
    <div style="background:#fff;border:1px solid #7b8794;padding:10px 14px;border-radius:6px;
    min-width:220px;text-align:center;">
      <div style="font-size:0.85rem;color:#52606d;">W · b → + → σ</div>
      <strong>Regresión logística</strong>
      <div style="font-size:0.85rem;margin-top:4px;">salida 1 (sigmoide)</div>
    </div>
    <div style="color:#52606d;">↓</div>
    <div style="background:#0b6e4f;color:#fff;padding:8px 18px;border-radius:6px;
    min-width:160px;text-align:center;font-weight:600;">
      Salida<br/><span style="font-size:0.95rem;">P(riesgo)</span>
    </div>
  </div>
  <div style="margin-top:10px;font-size:0.8rem;color:#52606d;text-align:center;">
    Arquitectura real del modelo operativo (no es red neuronal multicapa).
  </div>
</div>
"""


def _progress_frame(report: dict) -> pd.DataFrame:
    progress = report.get("progress") or {}
    timing = report.get("timing") or {}
    it = progress.get("iteration") or {}
    perf = progress.get("performance_error") or {}
    elapsed = progress.get("elapsed_time") or timing.get("total") or {}

    def _fmt(v, digits=6):
        if v is None:
            return "—"
        try:
            return f"{float(v):.{digits}g}"
        except (TypeError, ValueError):
            return str(v)

    rows = [
        {
            "Unidad": "Iteración (búsqueda de C)",
            "Valor inicial": it.get("initial", 0),
            "Valor final": it.get("stopped", "—"),
            "Valor meta": it.get("target", "—"),
        },
        {
            "Unidad": "Tiempo transcurrido",
            "Valor inicial": "—",
            "Valor final": elapsed.get("human", "—"),
            "Valor meta": "—",
        },
        {
            "Unidad": "Desempeño (1 − AUC val)",
            "Valor inicial": _fmt(perf.get("initial")),
            "Valor final": _fmt(perf.get("stopped")),
            "Valor meta": _fmt(perf.get("target"), 3),
        },
        {
            "Unidad": "Mejor C",
            "Valor inicial": "—",
            "Valor final": _fmt(progress.get("best_C"), 4),
            "Valor meta": "búsqueda / fijo",
        },
        {
            "Unidad": "Mejor AUC val",
            "Valor inicial": "—",
            "Valor final": _fmt(progress.get("best_val_auc"), 4),
            "Valor meta": "≥ 0.80",
        },
    ]
    return pd.DataFrame(rows)


def _plot_performance(report: dict):
    curve = report.get("search_curve") or []
    goal = float(report.get("performance_goal", 0.20))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if curve:
        xs = [r["iteration"] for r in curve]
        ys = [r["classification_error"] for r in curve]
        best_ys = [r["min_classification_error"] for r in curve]
        ax.semilogy(xs, ys, "b-", label="Error de validación (1 − AUC)")
        ax.axhline(min(best_ys), color="#1d4ed8", ls="--", label="Mejor")
        ax.axhline(goal, color="black", ls="--", label="Meta")
        best_i = int(np.argmin(best_ys)) + 1
        ax.plot(best_i, min(best_ys), "bo")
        ax.set_title(
            f"Mejor desempeño en validación: {min(best_ys):.5g} en la iteración {best_i}"
        )
        ax.set_xlabel(f"{len(xs)} iteraciones (búsqueda de C)")
    else:
        stopped = float(
            (report.get("progress") or {})
            .get("performance_error", {})
            .get("stopped", 1.0 - float(report.get("metrics", {}).get("auc_val", 0)))
        )
        ax.semilogy(
            [0, 1], [stopped, stopped], "b-", label="Error de validación (1 − AUC)"
        )
        ax.axhline(goal, color="black", ls="--", label="Meta")
        ax.set_title(f"Desempeño en validación: {stopped:.5g} (C fijo, sin búsqueda)")
        ax.set_xlabel("Iteración")
    ax.set_ylabel("Error (1 − AUC val)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_search_state(report: dict):
    """Estado de búsqueda real: C y mejor error. Sin Mu/gradiente (no aplican)."""
    curve = report.get("search_curve") or []
    if not curve:
        return None
    xs = [r["iteration"] for r in curve]
    cs = [r["C"] for r in curve]
    best_err = [r["min_classification_error"] for r in curve]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].semilogy(xs, cs, "b-o", ms=4)
    axes[0].set_ylabel("C")
    axes[0].set_title(
        f"C = {cs[int(np.argmin(best_err))]:.5g} en la mejor iteración"
    )
    axes[0].grid(True, which="both", ls=":", alpha=0.5)
    axes[1].semilogy(xs, best_err, "b-")
    axes[1].set_ylabel("mejor error")
    axes[1].set_xlabel(f"{len(xs)} iteraciones")
    axes[1].set_title(f"Mejor error = {min(best_err):.5g}")
    axes[1].grid(True, which="both", ls=":", alpha=0.5)
    fig.suptitle("Estado de búsqueda (hiperparámetro C)", y=1.02)
    fig.tight_layout()
    return fig


def _plot_calibration(report: dict):
    sample = report.get("calibration_sample") or {}
    target = np.asarray(sample.get("target") or [], dtype=float)
    output = np.asarray(sample.get("output") or [], dtype=float)
    if len(target) == 0 or len(output) == 0:
        return None
    r = sample.get("r")
    slope = sample.get("fit_slope")
    intercept = sample.get("fit_intercept")
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(
        target, output, "o", ms=3, mfc="none", mec="black", alpha=0.35, label="Datos"
    )
    ax.plot([0, 1], [0, 1], "k:", label="Y = T")
    if slope is not None and intercept is not None:
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, slope * xs + intercept, "b-", label="Ajuste")
        ax.set_ylabel(f"Salida ≈ {slope:.3g}·Objetivo + {intercept:.3g}")
    else:
        ax.set_ylabel("Salida (probabilidad)")
    r_txt = f"{r:.5f}" if isinstance(r, (int, float)) and r == r else "—"
    ax.set_title(f"Conjunto de prueba: R={r_txt}")
    ax.set_xlabel("Objetivo")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    return fig


def _render_dashboard() -> None:
    st.markdown("#### 4. Consola de entrenamiento (estilo MATLAB, logística)")
    report = st.session_state.get("last_training_run") or load_training_run()
    if not report:
        st.info("Aún no hay corridas registradas en `reports/training_run.json`.")
        return

    # Compatibilidad con corridas antiguas sin bloque progress.
    if "progress" not in report:
        curve = report.get("search_curve") or []
        metrics = report.get("metrics") or {}
        min_err = report.get("min_observed_classification_error")
        if min_err is None:
            min_err = 1.0 - float(metrics.get("auc_val", 0.0) or 0.0)
        report = {
            **report,
            "architecture": {
                "n_inputs": len(FEATURE_COLUMNS),
            },
            "algorithms": {
                "data_division": str(report.get("split_pct", {})),
                "training": "lbfgs (regresión logística)",
                "performance": "AUC-ROC (error = 1 − AUC val)",
                "calculations": "scikit-learn",
            },
            "progress": {
                "iteration": {
                    "initial": 0,
                    "stopped": len(curve) or 1,
                    "target": (report.get("hyperparameters") or {}).get(
                        "n_iter_search", len(curve) or 1
                    ),
                },
                "elapsed_time": (report.get("timing") or {}).get("total", {}),
                "performance_error": {
                    "initial": (
                        curve[0]["classification_error"] if curve else min_err
                    ),
                    "stopped": min_err,
                    "target": 0.20,
                    "met": float(min_err) <= 0.20,
                },
                "best_C": (report.get("hyperparameters") or {}).get("C"),
                "best_val_auc": 1.0 - float(min_err),
            },
            "performance_goal": 0.20,
            "status_detail": (
                "Entrenamiento terminado: se cumplió el criterio de desempeño"
                if float(min_err) <= 0.20
                else "Entrenamiento terminado: búsqueda finalizada"
            ),
        }

    left, right = st.columns([0.95, 1.25], gap="large")
    with left:
        st.markdown(_architecture_html(report), unsafe_allow_html=True)
    with right:
        met = bool(
            (report.get("progress") or {})
            .get("performance_error", {})
            .get("met", False)
        )
        detail = _status_es(report.get("status_detail"), met)
        if met:
            st.success(detail)
        else:
            st.warning(detail)

        st.markdown("**Progreso**")
        st.dataframe(_progress_frame(report), hide_index=True, width="stretch")

        alg = report.get("algorithms") or {}
        st.markdown("**Algoritmos**")
        st.markdown(
            f"- **División de datos:** {alg.get('data_division', '—')}\n"
            f"- **Entrenamiento:** {alg.get('training', '—')}\n"
            f"- **Desempeño:** {alg.get('performance', '—')}\n"
            f"- **Cálculos:** {alg.get('calculations', '—')}"
        )

        st.markdown("**Gráficas**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Desempeño", use_container_width=True):
                st.session_state["admin_plot"] = "performance"
        with c2:
            if st.button("Estado de búsqueda", use_container_width=True):
                st.session_state["admin_plot"] = "search"
        with c3:
            if st.button("Calibración", use_container_width=True):
                st.session_state["admin_plot"] = "calibration"

    plot_key = st.session_state.get("admin_plot", "performance")
    st.markdown("---")
    if plot_key == "performance":
        st.caption(
            "Desempeño: error de validación (1 − AUC) vs iteración de búsqueda de C. "
            "Meta = 0.20 (AUC ≥ 0.80)."
        )
        fig = _plot_performance(report)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    elif plot_key == "search":
        st.caption(
            "Estado de búsqueda: evolución real de C y del mejor error. "
            "No se muestran Mu/gradiente (propios de redes con trainlm)."
        )
        fig = _plot_search_state(report)
        if fig is None:
            st.info("Sin curva de búsqueda (entrenamiento con C fijo).")
        else:
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
    else:
        st.caption(
            "Calibración: objetivo (0/1) vs salida (probabilidad) en prueba, "
            "con ajuste lineal y R. Equivalente útil a plotregression."
        )
        fig = _plot_calibration(report)
        if fig is None:
            st.info(
                "Esta corrida no incluye muestra de calibración. "
                "Vuelva a pulsar **Entrenar ahora** para generarla."
            )
        else:
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    metrics = report.get("metrics") or {}
    data = report.get("data") or {}
    with st.expander("Métricas y datos (resumen)", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Concepto": "AUC train", "Valor": f"{metrics.get('auc_train', float('nan')):.4f}"},
                    {"Concepto": "AUC val", "Valor": f"{metrics.get('auc_val', float('nan')):.4f}"},
                    {"Concepto": "AUC test", "Valor": f"{metrics.get('auc_test', float('nan')):.4f}"},
                    {
                        "Concepto": "Sensibilidad (tamizaje)",
                        "Valor": f"{metrics.get('sensitivity_screening', float('nan')):.4f}",
                    },
                    {
                        "Concepto": "Especificidad (tamizaje)",
                        "Valor": f"{metrics.get('specificity_screening', float('nan')):.4f}",
                    },
                    {
                        "Concepto": "Filas train/val/test",
                        "Valor": (
                            f"{data.get('train_rows', '—')} / "
                            f"{data.get('val_rows', '—')} / "
                            f"{data.get('test_rows', '—')}"
                        ),
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Bitácora JSON completa", expanded=False):
        st.json(report)


def render_admin_panel() -> None:
    """UI completa del administrador (solo logística)."""
    ensure_seed_active_dataset()

    st.markdown("## Panel administrador")
    st.caption(
        "Datos externos · partición · reentrenamiento logístico · dashboard. "
        "Use ☰ para apariencia, volver a cliente o cerrar sesión."
    )

    with st.expander("Cambiar usuario / contraseña", expanded=False):
        _render_password_section()
    st.divider()
    active = _render_dataset_section()
    st.divider()
    params = _render_split_and_params()
    st.divider()
    _render_train(active, params)
    st.divider()
    _render_dashboard()
