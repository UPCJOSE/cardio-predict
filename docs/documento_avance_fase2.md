# Fase 2 – Modelo predictivo de riesgo cardiovascular

**Título:** Modelo predictivo de riesgo cardiovascular en población general y rural basado en regresión logística y XGBoost  

**Proyecto de aula – Segunda fase (segundo corte)**  
**Asignatura:** Inteligencia Artificial  
**Integrantes:** Gustavo José Mutto · Yafeth López Moscote · José López Pinzón  
**Universidad Popular del Cesar — Valledupar, 2026**

---

## 1. Introducción

Las enfermedades cardiovasculares (ECV) son la principal causa de muerte en Colombia. Las escalas clásicas como Framingham se construyeron con poblaciones norteamericanas/europeas y, en una cohorte colombiana, alcanzaron un AUC de apenas 0,53 (Agudelo et al., 2025), además de exigir laboratorio poco disponible en zonas rurales.

Este proyecto estima el riesgo cardiovascular con variables de encuesta (edad, sexo, IMC, salud percibida, actividad física, tabaquismo y antecedentes). Se comparan **regresión logística** (baseline interpretable) y **XGBoost** (modelo propuesto). El resultado se traduce a niveles bajo / moderado / alto como apoyo preventivo, **sin sustituir** el diagnóstico médico.

En la fase 1 se formuló la propuesta y el EDA. **Esta fase 2 actualiza esa propuesta con el preprocesamiento ejecutado, el entrenamiento, el ajuste de hiperparámetros y la evaluación cuantitativa sobre el conjunto de prueba.**

## 2. Estado del arte (síntesis)

Se mantiene la revisión de la propuesta: a nivel internacional, Xi et al. (2022), Ahiduzzaman y Hasan (2025) e Imani et al. (2026) muestran que XGBoost/ensambles superan levemente a la logística en BRFSS/NHANES, con AUC altos cuando se maneja el desbalance. A nivel nacional, Agudelo et al. (2025) evidencian el fallo de Framingham en Colombia. En Cesar/Caribe solo hay estudios descriptivos (MinSalud, Mosquera Heredia, Rebolledo-Cobos).

**Brecha:** falta un modelo restringido a variables de encuesta, de bajo costo y evaluado pensando en contextos rurales colombianos.

## 3. Planteamiento y objetivos

**Pregunta:** ¿es posible estimar de forma confiable el riesgo cardiovascular con variables de fácil recolección mediante regresión logística y XGBoost?

**Objetivo general:** desarrollar un modelo predictivo de riesgo cardiovascular (LR y XGBoost) como herramienta de orientación preventiva.

**Objetivos específicos de esta fase (cumplidos en código):**

1. Usar 14 predictores de fácil recolección.  
2. Depurar la base BRFSS cleaned y particionar train/test.  
3. Implementar LR y XGBoost con manejo del desbalance.  
4. Evaluar y comparar con métricas adecuadas (AUC-ROC/PR, sensibilidad, F1, etc.).  
5. Seleccionar el modelo según el criterio metodológico e interpretar coeficientes/importancias.

El prototipo Streamlit (objetivo 6) queda operativo como avance hacia la fase 3.

## 4. Justificación de algoritmos

| Modelo | Rol | Por qué |
|--------|-----|---------|
| **Regresión logística** | Baseline | Interpretable (odds ratios), ligera, estándar clínico-epidemiológico; en Xi et al. (2022) sigue siendo competitiva. |
| **XGBoost** | Propuesto | Capacidad no lineal y de interacciones; mejor desempeño reportado en BRFSS 2022 (Imani et al., 2026) y NHANES (Ahiduzzaman y Hasan, 2025). |

No se usó deep learning: el tamaño tabular y la necesidad de interpretabilidad en campo favorecen modelos clásicos/boosting.

## 5. Metodología experimental

### 5.1 Datos

- **Fuente:** Indicators of Heart Disease (BRFSS cleaned, CDC), espejo público del dataset de Kaggle.  
- **Nota:** la propuesta citaba BRFSS 2022 (209.958 filas depuradas). En este repositorio se usa el CSV cleaned público con **variables equivalentes** (HeartDisease = Yes/No como proxy de infarto/angina reportados), por disponibilidad sin autenticación Kaggle.  
- Filas crudas: **319.795** → tras quitar duplicados y nulos: **287.662** (positivos ≈ **9,4 %**).

### 5.2 Preprocesamiento

- Eliminación de duplicados (32.133).  
- Codificación: sexo, actividad, alcohol, diabetes, ACV, renal, dificultad para caminar → binarias; edad y salud general → ordinales; tabaquismo → Never/Current (el cleaned 2020 no distingue Former).  
- Recorte de IMC en p1–p99 (17,74–49,23); sueño 1–24; días de salud 0–30.  
- **Partición:** 80 % train / 20 % test, estratificada, `random_state=42` → 230.129 / 57.533.  
- El test **no se usó** hasta la evaluación final.  
- Validación: CV estratificada de 5 pliegues dentro de train.

### 5.3 Entrenamiento e hiperparámetros

- **LR:** `StandardScaler` + `LogisticRegression(class_weight='balanced')`. Búsqueda aleatoria sobre `C` (12 candidatos × 5 folds).  
  - Mejor CV AUC-ROC: **0,8307** con `C=0,001`.  
- **XGBoost:** `scale_pos_weight ≈ 9,59`, `RandomizedSearchCV` (15 iteraciones × 5 folds) sobre profundidad, learning rate, subsample, etc.  
  - Mejor CV AUC-ROC: **0,8336** con `max_depth=3`, `learning_rate=0,05`, `n_estimators=300`, `subsample=1,0`, `colsample_bytree=0,85`, `reg_lambda=5`, `gamma=0,5`.

### 5.4 Métricas y umbrales

Prioridad clínica: **sensibilidad** (no dejar pasar casos de riesgo). Se reportan AUC-ROC (con IC 95 % bootstrap), AUC-PR, F1, especificidad, Brier. Exactitud se informa pero no decide (clase mayoritaria ~91 %).

Umbral de **tamizaje:** el que logra sensibilidad ≥ 0,80 maximizando F1.  
Umbral **alto:** especificidad ≥ 0,90.

**Criterio de selección:** mayor AUC-ROC en test; si |Δ| < 0,01 → preferir logística.

## 6. Resultados

### 6.1 Comparación en conjunto de prueba (n = 57.533)

| Modelo | AUC-ROC | IC 95 % | AUC-PR | F1 (tamizaje) | Sens. | Espec. | Brier |
|--------|--------:|---------|-------:|--------------:|------:|-------:|------:|
| Regresión logística | **0,8323** | [0,8271; 0,8369] | 0,3502 | 0,3489 | 0,8031 | 0,7082 | 0,1716 |
| XGBoost | **0,8355** | [0,8306; 0,8402] | 0,3557 | 0,3523 | 0,8098 | 0,7095 | 0,1724 |

Figuras: `reports/figures/roc_curves.png`, `pr_curves.png`, `model_comparison.png`.

### 6.2 Selección

Δ AUC-ROC (XGBoost − LR) ≈ **0,0032 < 0,01**.  
**Modelo seleccionado: regresión logística**, por desempeño casi idéntico y mayor interpretabilidad para uso en campo.

Umbrales (logística): tamizaje ≈ **0,47** (sens. 0,80); alto ≈ **0,71** (espec. 0,90).

### 6.3 Interpretabilidad

**Odds ratios (logística, top):** AgeCategory 2,69 · GeneralHealth 1,63 · Sex (hombre) 1,42 · HadStroke 1,26 · SmokerStatus 1,20 · HadDiabetes 1,17 · HadKidneyDisease 1,12 · DifficultyWalking 1,09.

**Importancia XGBoost (top):** AgeCategory, DifficultyWalking, GeneralHealth, HadStroke, HadDiabetes, Sex.

Coincide con la literatura (edad, salud percibida, sexo, diabetes/ACV).

### 6.4 Limitaciones

1. Datos de EE. UU. (BRFSS); no se recalibró aún a prevalencia colombiana.  
2. Sin `_URBSTAT` en el CSV cleaned → evaluación rural pendiente (fase 3 / archivo CDC).  
3. Variable objetivo autoreportada (sesgo de recuerdo).  
4. Tabaquismo sin categoría “exfumador” en esta versión cleaned.  
5. AUC-PR ~0,35 refleja el desbalance; la sensibilidad prioritaria se logró vía umbral.

## 7. Repositorio y reproducibilidad

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
streamlit run app/streamlit_app.py --server.port 8505
```

Artefactos: `models/*.joblib`, `models/experiment.json`, `reports/metrics.json`, figuras y este documento. Control de versiones con commits sucesivos (scaffold → entrenamiento → informe).

## 8. Conclusiones y próximos pasos

Se cumplió el núcleo de la Fase 2: preprocesamiento, dos modelos justificados, tuning documentado y evaluación comparativa. XGBoost gana por un margen mínimo; la **logística** queda como modelo operativo por el criterio de parsimonia.

**Fase 3:** recalibración a Colombia (RIPS/CAC), análisis por subgrupo rural con BRFSS original, SHAP detallado, lineamientos de uso del prototipo y artículo/póster.

## Referencias (principales)

Agudelo et al. (2025); Ahiduzzaman y Hasan (2025); Imani et al. (2026); Xi et al. (2022); Chen y Guestrin (2016); Chawla et al. (2002); CDC BRFSS; MinSalud; Zurique-Sánchez et al. (2019). Lista completa en la propuesta de fase 1 y en el `.docx` original del equipo.
