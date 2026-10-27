# Modelo predictivo de riesgo cardiovascular (Fase 2)

Proyecto de aula — Universidad Popular del Cesar.  
Estimación de riesgo cardiovascular a partir de variables de encuesta (BRFSS), comparando **regresión logística** (baseline) y **XGBoost** (modelo propuesto).

> El prototipo es **orientativo** y **no sustituye** el diagnóstico médico.

## Integrantes

- Gustavo José Mutto  
- Yafeth López Moscote  
- José López Pinzón  

## Qué incluye esta fase

1. Preprocesamiento: limpieza, transformación, features y partición train/test estratificada.  
2. Dos modelos con justificación y ajuste de hiperparámetros (RandomizedSearchCV + CV 5-fold).  
3. Evaluación cuantitativa (AUC-ROC/PR, sensibilidad, especificidad, F1, Brier) y comparación.  
4. Documento técnico de avance en `docs/`.  
5. Prototipo Streamlit para ingresar las 14 variables y obtener nivel de riesgo.

## Estructura

```
app/                # Prototipo Streamlit
data/raw/           # CSV original (descargable; no versionado)
data/processed/     # train/test parquet (generados)
docs/               # Documento de avance Fase 2
models/             # .joblib + experiment.json
notebooks/          # Espacio para EDA adicional
reports/            # metrics.json, figuras, tabla comparativa
scripts/            # download_data.py, run_pipeline.py
src/                # data_prep, train, evaluate, config
```

## Datos

Fuente pública: *Indicators of Heart Disease* (BRFSS cleaned, CDC), espejo disponible sin autenticación de Kaggle.

```bash
python scripts/download_data.py
```

El archivo queda en `data/raw/heart_2020_cleaned.csv` (~24 MB).  
La metodología del curso se redactó sobre BRFSS 2022; este repositorio usa el CSV cleaned público con **variables equivalentes** (ver documento de avance).

## Instalación y ejecución

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Pipeline completo
python scripts/run_pipeline.py

# Prototipo
streamlit run app/streamlit_app.py --server.port 8505
```

## Modelos y criterio de selección

| Modelo | Rol | Manejo del desbalance |
|--------|-----|------------------------|
| Regresión logística | Baseline interpretable | `class_weight='balanced'` |
| XGBoost | Modelo propuesto | `scale_pos_weight ≈ n_neg/n_pos` |

Selección (metodología): mayor AUC-ROC en prueba; si la diferencia es &lt; 0.01, se prefiere la logística.

## Métricas

Tras `run_pipeline.py` revise:

- `reports/metrics.json`  
- `reports/comparison_table.md`  
- `reports/figures/roc_curves.png`  
- `reports/figures/pr_curves.png`  
- `reports/figures/model_comparison.png`  
- `models/experiment.json` (hiperparámetros)

## Licencia de datos

Datos BRFSS del CDC (dominio público / uso abierto). Este código del proyecto de aula se distribuye para fines académicos.
