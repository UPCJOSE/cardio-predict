"""Constantes del experimento: rutas, semilla, variables y umbrales.

Cualquier script debe importar de aquí para no duplicar nombres de columnas
ni cambiar a medias la partición train/test.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# CSV público BRFSS cleaned (espejo sin login de Kaggle).
RAW_CSV = DATA_RAW / "heart_2020_cleaned.csv"
RAW_URL = (
    "https://gist.githubusercontent.com/colemilne54/"
    "4701ba22d76b0457bcc8852d7d315772/raw/heart_2020_cleaned.csv"
)

RANDOM_STATE = 42  # reproduce la misma partición y el mismo RandomizedSearchCV
TEST_SIZE = 0.20  # 20 % prueba; el 80 % restante es entrenamiento
N_CV_FOLDS = 5  # pliegues estratificados para tunear hiperparámetros
N_ITER_SEARCH = 15  # intentos de RandomizedSearchCV

# 14 predictores de la metodología (nombres canónicos tras el mapeo del CSV).
FEATURE_COLUMNS = [
    "AgeCategory",
    "Sex",
    "BMI",
    "GeneralHealth",
    "PhysicalHealthDays",
    "MentalHealthDays",
    "PhysicalActivities",
    "SleepHours",
    "SmokerStatus",
    "AlcoholDrinkers",
    "HadDiabetes",
    "HadStroke",
    "HadKidneyDisease",
    "DifficultyWalking",
]

TARGET_COLUMN = "HeartDisease"  # 1 = autoreporte de enfermedad cardíaca

# Texto del CSV → número que sí aceptan los modelos.
ORDINAL_MAPS = {
    "AgeCategory": {
        "18-24": 0,
        "25-29": 1,
        "30-34": 2,
        "35-39": 3,
        "40-44": 4,
        "45-49": 5,
        "50-54": 6,
        "55-59": 7,
        "60-64": 8,
        "65-69": 9,
        "70-74": 10,
        "75-79": 11,
        "80 or older": 12,
    },
    "GeneralHealth": {
        "Excellent": 0,
        "Very good": 1,
        "Good": 2,
        "Fair": 3,
        "Poor": 4,
    },
    "SmokerStatus": {
        "Never": 0,
        "Former": 1,
        "Current": 2,
    },
}

# Metas para traducir probabilidad → nivel de riesgo (bajo / moderado / alto).
MIN_SENSITIVITY = 0.80  # umbral de tamizaje: no dejar pasar casos de riesgo
MIN_HIGH_SPECIFICITY = 0.90  # umbral alto: pocos falsos positivos
