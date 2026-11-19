"""Validación e ingestión de datasets externos para el panel Admin.

Acepta:
1) Esquema canónico numérico (14 features + HeartDisease).
2) CSV BRFSS cleaned (HeartDisease Yes/No, GenHealth, etc.) → se limpia
   con clean_and_engineer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ACTIVE_DATASET_JSON,
    ACTIVE_DATASET_PARQUET,
    DATA_PROCESSED,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.data_prep import clean_and_engineer

BRFSS_REQUIRED = [
    "HeartDisease",
    "BMI",
    "PhysicalHealth",
    "MentalHealth",
    "SleepTime",
    "Sex",
    "PhysicalActivity",
    "AlcoholDrinking",
    "Stroke",
    "KidneyDisease",
    "DiffWalking",
    "Diabetic",
    "AgeCategory",
    "GenHealth",
    "Smoking",
]

CANONICAL_REQUIRED = FEATURE_COLUMNS + [TARGET_COLUMN]

# Rangos esperados en el esquema canónico (tras limpieza).
CANONICAL_RANGES = {
    "AgeCategory": (0, 12),
    "Sex": (0, 1),
    "BMI": (10.0, 60.0),
    "GeneralHealth": (0, 4),
    "PhysicalHealthDays": (0, 30),
    "MentalHealthDays": (0, 30),
    "PhysicalActivities": (0, 1),
    "SleepHours": (1, 24),
    "SmokerStatus": (0, 2),
    "AlcoholDrinkers": (0, 1),
    "HadDiabetes": (0, 1),
    "HadStroke": (0, 1),
    "HadKidneyDisease": (0, 1),
    "DifficultyWalking": (0, 1),
    "HeartDisease": (0, 1),
}


@dataclass
class ValidationIssue:
    """Un error localizado para mostrar al administrador."""

    where: str
    reason: str
    row: int | None = None
    column: str | None = None


@dataclass
class ValidationResult:
    ok: bool
    schema: str | None = None
    df: pd.DataFrame | None = None
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def issues_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["fila", "columna", "ubicacion", "motivo"])
        return pd.DataFrame(
            [
                {
                    "fila": i.row,
                    "columna": i.column,
                    "ubicacion": i.where,
                    "motivo": i.reason,
                }
                for i in self.issues
            ]
        )


def _read_upload(name: str, data: bytes) -> pd.DataFrame:
    """Lee CSV o JSON desde bytes subidos en Streamlit."""
    lower = name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(BytesIO(data))
    if lower.endswith(".json"):
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            return pd.DataFrame(payload["records"])
        return pd.DataFrame(payload)
    raise ValueError("Formato no soportado. Use .csv o .json")


def detect_schema(columns: list[str]) -> str | None:
    cols = set(columns)
    if set(CANONICAL_REQUIRED).issubset(cols):
        return "canonical"
    if set(BRFSS_REQUIRED).issubset(cols):
        return "brfss"
    return None


def _validate_canonical(df: pd.DataFrame, max_row_issues: int = 80) -> ValidationResult:
    issues: list[ValidationIssue] = []
    missing = [c for c in CANONICAL_REQUIRED if c not in df.columns]
    for col in missing:
        issues.append(
            ValidationIssue(
                where="columnas",
                column=col,
                reason=f"Falta la columna obligatoria '{col}'.",
            )
        )
    if missing:
        return ValidationResult(ok=False, schema="canonical", issues=issues)

    work = df[CANONICAL_REQUIRED].copy()
    for col in CANONICAL_REQUIRED:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    null_counts = work.isna().sum()
    for col, n_null in null_counts.items():
        if n_null:
            issues.append(
                ValidationIssue(
                    where="valores nulos",
                    column=col,
                    reason=f"{int(n_null)} fila(s) con valor no numérico o vacío.",
                )
            )

    row_issues = 0
    for col, (lo, hi) in CANONICAL_RANGES.items():
        series = work[col]
        bad = series.notna() & ((series < lo) | (series > hi))
        bad_idx = list(work.index[bad][: max(0, max_row_issues - row_issues)])
        for idx in bad_idx:
            issues.append(
                ValidationIssue(
                    where="rango",
                    row=int(idx) + 2,  # +2: encabezado + 1-index humano aproximado
                    column=col,
                    reason=f"Valor {series.loc[idx]} fuera de [{lo}, {hi}].",
                )
            )
            row_issues += 1
        n_bad = int(bad.sum())
        if n_bad > len(bad_idx):
            issues.append(
                ValidationIssue(
                    where="rango",
                    column=col,
                    reason=f"{n_bad - len(bad_idx)} fila(s) adicionales fuera de rango (no listadas).",
                )
            )

    if work[TARGET_COLUMN].nunique(dropna=True) < 2:
        issues.append(
            ValidationIssue(
                where="objetivo",
                column=TARGET_COLUMN,
                reason="HeartDisease debe tener al menos dos clases (0 y 1).",
            )
        )

    blocking = [i for i in issues if i.where in {"columnas", "objetivo"}]
    # Nulos/rangos: se pueden eliminar filas; si quedan muy pocas, falla.
    clean = work.dropna().copy()
    for col, (lo, hi) in CANONICAL_RANGES.items():
        clean = clean[(clean[col] >= lo) & (clean[col] <= hi)]
    clean = clean.reset_index(drop=True)

    if len(clean) < 50:
        issues.append(
            ValidationIssue(
                where="tamaño",
                reason=f"Tras limpiar quedan {len(clean)} filas; se necesitan al menos 50.",
            )
        )

    ok = not blocking and len(clean) >= 50 and clean[TARGET_COLUMN].nunique() >= 2
    summary = {
        "rows_input": int(len(df)),
        "rows_valid": int(len(clean)),
        "positive_rate": float(clean[TARGET_COLUMN].mean()) if len(clean) else None,
        "schema": "canonical",
    }
    return ValidationResult(
        ok=ok,
        schema="canonical",
        df=clean if ok else None,
        issues=issues,
        summary=summary,
    )


def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Valida un DataFrame ya cargado (CSV/JSON)."""
    schema = detect_schema(list(df.columns))
    if schema is None:
        missing_c = [c for c in CANONICAL_REQUIRED if c not in df.columns]
        missing_b = [c for c in BRFSS_REQUIRED if c not in df.columns]
        return ValidationResult(
            ok=False,
            issues=[
                ValidationIssue(
                    where="esquema",
                    reason=(
                        "No se reconoció el formato. "
                        f"Al esquema canónico le faltan: {missing_c[:8]}{'…' if len(missing_c) > 8 else ''}. "
                        f"Al BRFSS le faltan: {missing_b[:8]}{'…' if len(missing_b) > 8 else ''}."
                    ),
                )
            ],
        )

    if schema == "brfss":
        try:
            cleaned, quality = clean_and_engineer(df)
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(
                ok=False,
                schema="brfss",
                issues=[
                    ValidationIssue(
                        where="limpieza BRFSS",
                        reason=f"No se pudo transformar el CSV: {exc}",
                    )
                ],
            )
        # Revalidar numérico por si la limpieza dejó outliers extremos.
        result = _validate_canonical(cleaned)
        result.schema = "brfss→canonical"
        result.summary = {**quality, **result.summary, "schema": "brfss→canonical"}
        return result

    return _validate_canonical(df)


def validate_upload(name: str, data: bytes) -> ValidationResult:
    """Punto de entrada desde Streamlit (archivo subido)."""
    try:
        df = _read_upload(name, data)
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(
            ok=False,
            issues=[ValidationIssue(where="archivo", reason=str(exc))],
        )
    if df.empty:
        return ValidationResult(
            ok=False,
            issues=[ValidationIssue(where="archivo", reason="El archivo está vacío.")],
        )
    return validate_dataframe(df)


def save_active_dataset(df: pd.DataFrame) -> dict:
    """Persiste el dataset activo en parquet + JSON para exportar/reentrenar."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ordered = df[CANONICAL_REQUIRED].copy()
    ordered.to_parquet(ACTIVE_DATASET_PARQUET, index=False)
    payload = {
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "n_rows": int(len(ordered)),
        "records": ordered.to_dict(orient="records"),
    }
    ACTIVE_DATASET_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "parquet": str(ACTIVE_DATASET_PARQUET),
        "json": str(ACTIVE_DATASET_JSON),
        "n_rows": int(len(ordered)),
        "positive_rate": float(ordered[TARGET_COLUMN].mean()),
    }


def load_active_dataset() -> pd.DataFrame | None:
    """Carga el dataset activo si existe."""
    if ACTIVE_DATASET_PARQUET.exists():
        return pd.read_parquet(ACTIVE_DATASET_PARQUET)
    if ACTIVE_DATASET_JSON.exists():
        payload = json.loads(ACTIVE_DATASET_JSON.read_text(encoding="utf-8"))
        return pd.DataFrame(payload.get("records", []))
    return None


def active_dataset_bytes_json() -> bytes | None:
    """Contenido JSON del dataset activo para el botón de descarga."""
    if ACTIVE_DATASET_JSON.exists():
        return ACTIVE_DATASET_JSON.read_bytes()
    df = load_active_dataset()
    if df is None:
        return None
    info = save_active_dataset(df)
    return Path(info["json"]).read_bytes()
