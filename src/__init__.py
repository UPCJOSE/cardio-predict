"""Pipeline de riesgo cardiovascular (Fase 2).

Orden de ejecución (scripts/run_pipeline.py):
1. scripts/download_data.py  → CSV BRFSS en data/raw/
2. src.data_prep             → limpia, codifica y parte train/test
3. src.train                 → entrena logística y XGBoost; guarda models/*.joblib
4. src.evaluate              → métricas y figuras en reports/
5. app/streamlit_app.py      → prototipo que carga esos .joblib
"""
