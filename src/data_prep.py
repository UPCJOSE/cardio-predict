"""Limpieza, transformación, ingeniería de características y partición.

Convierte el CSV BRFSS (texto Yes/No, rangos de edad, etc.) al esquema numérico
de 14 predictores + HeartDisease, y escribe train.parquet / test.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED,
    FEATURE_COLUMNS,
    ORDINAL_MAPS,
    RANDOM_STATE,
    RAW_CSV,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Lee el CSV crudo. Si no está, hay que correr scripts/download_data.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecute: python scripts/download_data.py"
        )
    return pd.read_csv(path)


def _map_smoking(value: str) -> str:
    """El CSV cleaned 2020 solo trae Yes/No; no existe categoría 'exfumador'."""
    if value == "Yes":
        return "Current"
    return "Never"


def clean_and_engineer(df: pd.DataFrame):
    """Pasa el CSV público a columnas canónicas, quita duplicados y recorta IMC.

    Returns:
        (tabla limpia, dict de calidad para quality_report.json)
    """
    out = pd.DataFrame(index=df.index)

    out["HeartDisease"] = (df["HeartDisease"] == "Yes").astype(int)
    out["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    out["PhysicalHealthDays"] = pd.to_numeric(df["PhysicalHealth"], errors="coerce")
    out["MentalHealthDays"] = pd.to_numeric(df["MentalHealth"], errors="coerce")
    out["SleepHours"] = pd.to_numeric(df["SleepTime"], errors="coerce")

    out["Sex"] = (df["Sex"] == "Male").astype(int)
    out["PhysicalActivities"] = (df["PhysicalActivity"] == "Yes").astype(int)
    out["AlcoholDrinkers"] = (df["AlcoholDrinking"] == "Yes").astype(int)
    out["HadStroke"] = (df["Stroke"] == "Yes").astype(int)
    out["HadKidneyDisease"] = (df["KidneyDisease"] == "Yes").astype(int)
    out["DifficultyWalking"] = (df["DiffWalking"] == "Yes").astype(int)

    # Diabetes: sí clínico (excluye gestacional / borderline como no-sí).
    out["HadDiabetes"] = df["Diabetic"].isin(["Yes"]).astype(int)

    out["AgeCategory"] = df["AgeCategory"].map(ORDINAL_MAPS["AgeCategory"])
    out["GeneralHealth"] = df["GenHealth"].map(ORDINAL_MAPS["GeneralHealth"])

    # Smoking en cleaned 2020 es Yes/No; se mapea a Never/Current (sin Former).
    smoke = df["Smoking"].map(_map_smoking)
    out["SmokerStatus"] = smoke.map(ORDINAL_MAPS["SmokerStatus"])

    before = len(out)
    out = out.drop_duplicates()
    n_dupes = before - len(out)

    # Recorte de IMC en percentiles 1–99 (metodología).
    lo, hi = out["BMI"].quantile([0.01, 0.99])
    out["BMI"] = out["BMI"].clip(lo, hi)

    out["SleepHours"] = out["SleepHours"].clip(1, 24)
    out["PhysicalHealthDays"] = out["PhysicalHealthDays"].clip(0, 30)
    out["MentalHealthDays"] = out["MentalHealthDays"].clip(0, 30)

    out = out.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    quality = {
        "rows_raw": int(len(df)),
        "duplicates_removed": int(n_dupes),
        "rows_final": int(len(out)),
        "positive_rate": float(out[TARGET_COLUMN].mean()),
        "bmi_clip_p01": float(lo),
        "bmi_clip_p99": float(hi),
        "features": FEATURE_COLUMNS,
    }
    return out, quality


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 estratificado por HeartDisease (misma proporción de positivos en ambos)."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    quality: dict,
) -> None:
    """Guarda train/test en parquet y el informe de calidad en JSON."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train = X_train.copy()
    train[TARGET_COLUMN] = y_train.values
    test = X_test.copy()
    test[TARGET_COLUMN] = y_test.values
    train.to_parquet(DATA_PROCESSED / "train.parquet", index=False)
    test.to_parquet(DATA_PROCESSED / "test.parquet", index=False)
    (DATA_PROCESSED / "quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )


def run() -> dict:
    """Pipeline completo de datos. Devuelve el dict de calidad (también lo imprime)."""
    raw = load_raw()
    cleaned, quality = clean_and_engineer(raw)
    X_train, X_test, y_train, y_test = split_data(cleaned)
    quality["train_rows"] = int(len(X_train))
    quality["test_rows"] = int(len(X_test))
    quality["train_positive_rate"] = float(y_train.mean())
    quality["test_positive_rate"] = float(y_test.mean())
    save_processed(X_train, X_test, y_train, y_test, quality)
    print(json.dumps(quality, indent=2))
    return quality


if __name__ == "__main__":
    run()
