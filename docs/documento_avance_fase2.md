Fase 2 – Modelo predictivo de riesgo cardiovascular
Oct 27, 2026 · @GUSTAVO
Portada
Título: Modelo predictivo de riesgo cardiovascular en población general y rural basado en regresión logística y XGBoost
Proyecto de aula – Segunda fase (segundo corte)
Asignatura: Inteligencia Artificial
Integrantes:
Gustavo José Mutto
Yafeth López Moscote
José López Pinzón
Universidad Popular del Cesar
Valledupar, 2026
1. Introducción
Las enfermedades cardiovasculares (ECV) son la principal causa de muerte en Colombia y una de las mayores cargas de salud pública en el mundo. Su aparición se asocia con factores de riesgo modificables (hipertensión arterial, diabetes, sobrepeso, sedentarismo, tabaquismo, consumo nocivo de alcohol y estrés) y con determinantes sociales que limitan el acceso oportuno a los servicios de salud.
Las escalas de riesgo más usadas, como la de Framingham, se construyeron con poblaciones norteamericanas y europeas. Al evaluarse en una cohorte colombiana, Framingham obtuvo un AUC de 0,53, una capacidad de discriminación cercana al azar (Agudelo et al., 2025). Además, estas escalas exigen pruebas de laboratorio (colesterol, HDL) que no siempre están disponibles en zonas rurales.
Este proyecto propone un modelo de inteligencia artificial que estima el riesgo cardiovascular a partir de variables de encuesta, fáciles de recoger sin laboratorio: edad, sexo, índice de masa corporal, salud percibida, actividad física, tabaquismo y antecedentes como diabetes, accidente cerebrovascular o enfermedad renal. Se comparan dos modelos de clasificación supervisada, la regresión logística y XGBoost, y el resultado se presenta como un nivel de riesgo (bajo, moderado, alto) que orienta la prevención. El modelo es un apoyo y no sustituye el diagnóstico médico.
En la primera fase se formuló la propuesta y se realizó el análisis exploratorio de datos (EDA) sobre la encuesta BRFSS 2022 (209.958 registros depurados). En esta segunda fase el documento se completa con la introducción, el estado del arte en los niveles internacional, nacional y local, la justificación y la metodología, que describe las actividades que se realizarán para cumplir cada objetivo específico.
2. Estado del arte
La revisión se organiza en tres niveles. En el internacional y el nacional se priorizaron trabajos publicados desde 2022. En el nivel local (Cesar y región Caribe) la producción es escasa, por lo que se amplió la ventana de búsqueda e incluyó informes oficiales.
2.1 Nivel internacional
Xi, Wang y Sun (2022) compararon la regresión logística con Elastic-Net, SVM, Random Forest, XGBoost, aprendizaje profundo y un ensamble en 143.043 pacientes hipertensos de 600 hospitales chinos, con 18 variables clínicas y validación cruzada de 5 pliegues. El ensamble obtuvo el mayor AUC (0,760), seguido de Random Forest (0,754) y XGBoost (0,751), frente a 0,737 de la regresión logística. La logística sigue siendo una línea base competitiva e interpretable; este diseño comparativo es el que adopta el proyecto.
Ahiduzzaman y Hasan (2025) entrenaron regresión logística, Random Forest, SVM, XGBoost y LightGBM con 12.382 adultos de la encuesta NHANES 2017–2023 y 30 predictores. XGBoost logró la mayor exactitud (0,82) y sensibilidad (0,86), y la edad fue el predictor más importante. Interpretaron el modelo con SHAP y LIME, enfoque que se aplicará en este proyecto.
Imani et al. (2026) analizaron 221.643 participantes de BRFSS 2022 —la misma fuente de este proyecto— con solo 5,8 % de casos positivos. Aplicaron SMOTE únicamente en entrenamiento y validación cruzada estratificada de 10 pliegues. XGBoost fue el mejor (AUC 0,94; sensibilidad 87,4 %), y la edad ≥ 65 años, el sexo masculino y la diabetes fueron los predictores más fuertes. Demuestra que variables de encuesta, sin laboratorio, permiten predecir el riesgo con buen desempeño.
2.2 Nivel nacional
Agudelo et al. (2025) estimaron el riesgo en 847 personas seguidas durante diez años en un hospital de Bogotá. Las redes neuronales alcanzaron un AUC de 0,69 y otras técnicas entre 0,56 y 0,65, todas por encima de Framingham (AUC 0,53). Es el antecedente nacional más directo y muestra que las escalas extranjeras discriminan mal en Colombia.
Hernández et al. (2025) caracterizaron, con registros clínicos, el perfil de riesgo de pacientes atendidos en zonas rurales del país. La hipertensión arterial fue la comorbilidad más frecuente, y concluyen que su control es prioritario en contextos rurales.
Palacios et al. (2022) estudiaron 540 participantes de un programa de riesgo cardiovascular en un municipio rural del norte del Valle del Cauca. La hipertensión arterial estuvo presente en el 95,5 % y el sobrepeso u obesidad en el 65,7 %; con Framingham calibrada, el 75,7 % quedó en riesgo bajo. Los autores señalan que la información sobre riesgo cardiovascular en población rural colombiana es escasa.
Zurique-Sánchez et al. (2019), con participación de la Universidad Popular del Cesar, realizaron una revisión sistemática y metaanálisis de 23 estudios: la prevalencia de hipertensión en adultos colombianos fue del 24 % (IC 95 %: 19–29 %), con tendencia al aumento.
2.3 Nivel local (Cesar y región Caribe)
Ministerio de Salud y Protección Social (2017) reportó que la prevalencia de hipertensión en personas de 18 a 69 años afiliadas al sistema de salud en el Cesar pasó de 3,23 % en 2011 a 4,89 % en 2015, un aumento sostenido en cinco años. Por su parte, la Cuenta de Alto Costo ubica a la región Caribe como la segunda con más casos de hipertensión del país (20,98 %) (MinSalud, 2026).
Mosquera Heredia et al. (2011) evaluaron 225 niños de 8 a 11 años de cinco escuelas públicas de Valledupar. El sedentarismo (41,7 %), los triglicéridos elevados (18,5 %) y el HDL bajo (26,9 %) fueron los factores de riesgo más frecuentes, y el 6,6 % de los niños presentó obesidad. Indica que la exposición al riesgo en la ciudad comienza desde la infancia.
Rebolledo-Cobos et al. (2024) estudiaron 395 trabajadores de la salud de Barranquilla y Riohacha. Más de la mitad tenía sobrepeso u obesidad, el 59,94 % no hacía 30 minutos diarios de actividad física y cerca de una cuarta parte tenía riesgo moderado o mayor de diabetes a diez años (FINDRISC). Destaca que variables sencillas de cuestionario y antropometría permiten identificar riesgo en la región.
2.4 Síntesis y brecha
Nivel
Trabajo
Datos
Método
Aporte al proyecto
Internacional
Xi et al. (2022)
143.043 hipertensos, China
LR, RF, XGBoost, DL, ensamble
Diseño comparativo LR vs. XGBoost
Internacional
Ahiduzzaman y Hasan (2025)
NHANES, EE. UU. (n = 12.382)
LR, RF, SVM, XGBoost, LightGBM + SHAP
Interpretabilidad con SHAP
Internacional
Imani et al. (2026)
BRFSS 2022 (n = 221.643)
XGBoost, RF, LR, NB + SMOTE
Misma fuente; manejo del desbalance
Nacional
Agudelo et al. (2025)
Cohorte de Bogotá (n = 847)
RN, árboles, SVM, RF, redes bayesianas
Framingham falla en Colombia (AUC 0,53)
Nacional
Hernández et al. (2025)
Registros clínicos rurales
Estudio transversal
Hipertensión prioritaria en lo rural
Nacional
Palacios et al. (2022)
Programa rural, Valle (n = 540)
Estudio transversal + Framingham
Escasez de datos rurales
Nacional
Zurique-Sánchez et al. (2019)
23 estudios colombianos
Metaanálisis
Prevalencia de hipertensión 24 %
Local
MinSalud (2017)
Afiliados al SGSSS, Cesar
Informe epidemiológico
Hipertensión en aumento en el Cesar
Local
Mosquera Heredia et al. (2011)
Escolares de Valledupar (n = 225)
Estudio descriptivo
Sedentarismo desde la infancia
Local
Rebolledo-Cobos et al. (2024)
Barranquilla y Riohacha (n = 395)
Estudio transversal
Variables de cuestionario útiles
Brecha identificada: los modelos de mejor desempeño se entrenaron con poblaciones extranjeras, y el único antecedente colombiano con IA usa una cohorte urbana y variables de laboratorio. En el Cesar y el Caribe solo hay estudios descriptivos de prevalencia, sin modelos predictivos. No se encontró un modelo restringido a variables de encuesta, de bajo costo y evaluado pensando en contextos rurales colombianos; ese es el aporte diferencial del proyecto.
3. Planteamiento del problema
Las enfermedades cardiovasculares representan cerca del 27 % de las muertes en Colombia, lo que equivale a unas 8 personas fallecidas cada hora por causas relacionadas con el corazón (Revista Momentos, 2025).
La falta de control de la presión arterial es el común denominador de estas muertes. Entre sus causas están los factores hereditarios, el diagnóstico tardío, la falta de ejercicio, el consumo nocivo de alcohol, la diabetes, el sobrepeso, el estrés, la falta de seguimiento al tratamiento y el desconocimiento de la enfermedad. Muchas personas no saben que se recomienda consultar al médico cuando la presión supera 140/90 mmHg y que lo ideal es mantenerla por debajo de 120/80 mmHg.
El problema se agrava en las zonas rurales. Allí el centro de salud más cercano suele quedar lejos, hay menos personal médico y menos acceso a pruebas de laboratorio, lo que retrasa la detección temprana (Hernández et al., 2025; Palacios et al., 2022). En el Cesar, la prevalencia de hipertensión registrada en afiliados al sistema de salud creció de 3,23 % a 4,89 % entre 2011 y 2015 (MinSalud, 2017), mientras que en la población general del país se estima en 24 % (Zurique-Sánchez et al., 2019). Esa diferencia sugiere un subregistro importante: muchas personas tienen la enfermedad y no lo saben.
A esto se suma que las herramientas de estimación disponibles no se ajustan al país: Framingham, la escala adoptada en Colombia, discrimina casi al azar en población colombiana (AUC 0,53) y requiere exámenes de laboratorio (Agudelo et al., 2025). Falta, por tanto, una herramienta que identifique a las personas con mayor riesgo usando solo preguntas y medidas sencillas, aplicable por un promotor de salud en una vereda.
Pregunta de investigación: ¿es posible estimar de forma confiable el riesgo cardiovascular de una persona a partir de variables clínicas y demográficas de fácil recolección, mediante técnicas de aprendizaje automático (regresión logística y XGBoost), para apoyar la orientación preventiva en zonas con acceso limitado a servicios de salud?
4. Objetivos
4.1 Objetivo general
Desarrollar un modelo predictivo de riesgo cardiovascular basado en técnicas de inteligencia artificial (regresión logística y XGBoost) que, a partir de variables clínicas y demográficas de fácil recolección, permita estimar el riesgo cardiovascular en población general y rural de Colombia, como herramienta de apoyo a la orientación preventiva y complementaria al diagnóstico médico.
4.2 Objetivos específicos
Identificar las variables clínicas y demográficas de fácil recolección con mayor relevancia para la predicción del riesgo cardiovascular en población general y rural.
Construir y depurar una base de datos a partir de la encuesta BRFSS 2022, contextualizada con fuentes colombianas (ECV 2023 del DANE, RIPS y Cuenta de Alto Costo), garantizando su calidad.
Implementar modelos de regresión logística y XGBoost para la clasificación del riesgo cardiovascular, con manejo del desbalance de clases.
Evaluar y comparar el desempeño de ambos modelos mediante métricas de clasificación (exactitud, sensibilidad, especificidad, F1 y AUC-ROC), incluyendo su desempeño en el subgrupo rural.
Seleccionar el modelo con mejor desempeño y documentar su interpretabilidad (SHAP), limitaciones y condiciones de uso en contextos rurales.
Diseñar un prototipo de interfaz que permita ingresar las variables de una persona y obtener su nivel de riesgo, acompañado de lineamientos de uso como herramienta de orientación preventiva, no sustitutiva del diagnóstico profesional.
Cambios respecto a la fase 1: el objetivo 2 aclara que BRFSS es la base de entrenamiento y que las fuentes colombianas sirven para contextualizar (no se pueden unir registro a registro con datos de EE. UU.); el objetivo 4 agrega la evaluación en el subgrupo rural, y el objetivo 6 incorpora el prototipo de interfaz que exige la fase 3.
5. Justificación
Relevancia social. Según la Cuenta de Alto Costo, al 28 de febrero de 2023 había 5.116.795 personas diagnosticadas con hipertensión en Colombia (prevalencia del 9,55 %), el 60,86 % mujeres, con mayor concentración en las regiones Central (28,35 %) y Caribe (20,98 %) (MinSalud, 2026). La enfermedad cardiovascular es la primera causa de mortalidad del país y la segunda de mortalidad prematura entre los 30 y 70 años. En el Cesar la hipertensión registrada viene en aumento (MinSalud, 2017), y desde la infancia aparecen factores como el sedentarismo (Mosquera Heredia et al., 2011).
Necesidad técnica. La escala de Framingham obtuvo un AUC de 0,53 en una cohorte colombiana (Agudelo et al., 2025) y requiere colesterol y HDL, pruebas poco disponibles en zonas rurales. En cambio, estudios recientes con BRFSS 2022 muestran que variables de encuesta permiten predecir enfermedad cardiovascular con AUC de 0,94 usando XGBoost (Imani et al., 2026). Comparar la regresión logística (simple e interpretable) con XGBoost (mayor capacidad) permite elegir con evidencia el modelo más adecuado.
Beneficio práctico. Un modelo que solo pide preguntas, una báscula y un metro puede aplicarlo un promotor de salud o un auxiliar de enfermería en una vereda, priorizar a quién remitir y orientar hábitos preventivos. La salida en niveles de riesgo (bajo, moderado, alto) es fácil de comunicar a la comunidad.
Pertinencia institucional y académica. El proyecto se alinea con el Plan Nacional de Desarrollo 2022-2026, que plantea avanzar hacia un modelo de salud preventivo y predictivo (MinSalud, 2026). Para la Universidad Popular del Cesar aporta un antecedente local de aplicación de inteligencia artificial en salud pública, una región donde la literatura es solo descriptiva.
Viabilidad. Todas las fuentes son abiertas y gratuitas (BRFSS, ECV, RIPS, CAC), no contienen datos personales identificables y el desarrollo se hace con herramientas libres (Python, scikit-learn, XGBoost, Google Colab), lo que hace viable el proyecto en 16 semanas.
6. Metodología
6.1 Tipo de investigación
Enfoque: cuantitativo, porque el problema se resuelve midiendo variables y estimando probabilidades con modelos estadísticos y de aprendizaje automático.
Tipo: aplicada, porque busca producir una herramienta concreta para la orientación preventiva.
Alcance: correlacional-predictivo: identifica qué variables se asocian con la enfermedad cardiovascular y las usa para predecir su presencia.
Diseño: no experimental y transversal, con datos secundarios (encuesta BRFSS 2022). No se interviene sobre las personas ni se hace seguimiento en el tiempo (Hernández Sampieri et al., 2014).
6.2 Método de investigación: CRISP-DM
El proyecto sigue CRISP-DM (Cross-Industry Standard Process for Data Mining), uno de los métodos más usados en proyectos de ciencia de datos (Chapman et al., 2000). Sus seis fases se recorren en orden, con regresos cuando la evaluación muestra que hay que mejorar los datos o el modelo. Cada fase CRISP-DM se corresponde con las fases que exige el proyecto de aula: análisis (comprensión del problema y de los datos), diseño (preparación y modelado), prueba (evaluación) e implementación (despliegue del prototipo).

Figura 1. Fases del método CRISP-DM aplicadas al proyecto
Las fases 3 y 4 (preparación y modelado) son las que se ejecutan en este corte; si la evaluación no alcanza las metas de sensibilidad y AUC, se regresa a ajustar los datos o el modelo.
6.3 Población, muestra y variables
Población: adultos de 18 años o más encuestados por el BRFSS 2022 del CDC.
Muestra: 209.958 registros completos tras eliminar 126 duplicados (fase 1). De ellos, 18.511 (8,8 %) presentan enfermedad cardiovascular.
Variable objetivo: enfermedad = 1 si la persona reporta infarto previo (HadHeartAttack) o angina/enfermedad coronaria (HadAngina); 0 en caso contrario.
Se usarán 14 predictores, todos obtenibles con una encuesta corta, una báscula y un metro:
Variable (BRFSS)
Pregunta en campo
Tipo
Codificación
AgeCategory
¿Cuántos años tiene?
Ordinal
0 (18–24) a 12 (80+)
Sex
Sexo
Binaria
1 = hombre
BMI
Peso (kg) / talla² (m)
Numérica
Recorte p1–p99
GeneralHealth
¿Cómo califica su salud?
Ordinal
0 excelente a 4 mala
PhysicalHealthDays
Días del último mes con mala salud física
Numérica
0–30
MentalHealthDays
Días del último mes con mala salud mental
Numérica
0–30
PhysicalActivities
¿Hizo actividad física el último mes?
Binaria
1 = sí
SleepHours
Horas de sueño promedio
Numérica
1–24
SmokerStatus
¿Fuma o fumó?
Ordinal
0 nunca, 1 exfumador, 2 actual
AlcoholDrinkers
¿Consumió alcohol el último mes?
Binaria
1 = sí
HadDiabetes
¿Le han diagnosticado diabetes?
Binaria
1 = sí (excluye gestacional)
HadStroke
¿Ha tenido un ACV?
Binaria
1 = sí
HadKidneyDisease
¿Le han diagnosticado enfermedad renal?
Binaria
1 = sí
DifficultyWalking
¿Tiene dificultad para caminar o subir escaleras?
Binaria
1 = sí
Se excluyen a propósito: ChestScan (hacerse una tomografía de tórax es consecuencia de sospecha cardíaca y filtraría la respuesta al modelo), State y las vacunas (no aplican a Colombia) y RaceEthnicityCategory (las categorías raciales de EE. UU. no se trasladan a la población colombiana y podrían introducir sesgos).
6.4 Actividades por objetivo específico
Objetivo específico
Actividades
Técnicas y herramientas
Entregable
1. Identificar variables relevantes
Revisar la literatura; analizar prevalencia y correlación por factor (EDA de fase 1); aplicar chi-cuadrado e información mutua; elegir las 14 variables de fácil recolección; confirmar su peso con la importancia de variables del modelo
pandas, SciPy, seaborn
Lista justificada de variables
2. Construir y depurar la base de datos
Descargar BRFSS 2022 (versión Kaggle y archivo original del CDC); eliminar duplicados; codificar variables; recortar IMC en p1–p99; comparar eliminación frente a imputación de nulos; dividir 80 % entrenamiento y 20 % prueba de forma estratificada; consultar ECV 2023, RIPS y CAC para prevalencias colombianas
pandas, scikit-learn
Dataset listo para modelar e informe de calidad
3. Implementar los modelos
Construir el pipeline de preprocesamiento; entrenar regresión logística con pesos de clase; entrenar XGBoost con scale_pos_weight; probar una variante con SMOTE solo en entrenamiento; ajustar hiperparámetros con búsqueda aleatoria y validación cruzada de 5 pliegues
scikit-learn, XGBoost, imbalanced-learn
Modelos entrenados (.joblib)
4. Evaluar y comparar
Calcular métricas sobre el conjunto de prueba; trazar curvas ROC y precisión-sensibilidad; fijar el umbral para sensibilidad ≥ 0,80; estimar intervalos de confianza del AUC por bootstrap; comparar por subgrupos (sexo, edad, zona rural)
scikit-learn, matplotlib
Tabla comparativa de métricas
5. Seleccionar e interpretar
Aplicar el criterio de selección; calcular odds ratios (logística) y valores SHAP (XGBoost); verificar la calibración; recalibrar a la prevalencia colombiana; redactar limitaciones
SHAP, scikit-learn
Informe de interpretabilidad y limitaciones
6. Prototipo y lineamientos
Diseñar un formulario con las 14 preguntas; mostrar probabilidad, nivel de riesgo y factores que más pesan; incluir el aviso de que no es un diagnóstico; probar con casos ficticios; redactar lineamientos de uso
Streamlit o Gradio
Prototipo funcional y guía de uso
6.5 Diseño de la experimentación
Partición: 80 % entrenamiento y 20 % prueba, estratificada por la variable objetivo (semilla fija = 42). El conjunto de prueba no se toca hasta la evaluación final.
Validación: validación cruzada estratificada de 5 pliegues dentro del entrenamiento para ajustar hiperparámetros.
Desbalance (91,2 % / 8,8 %): pesos de clase (class_weight = balanced en la logística; scale_pos_weight ≈ 10,3 en XGBoost). Como alternativa, SMOTE dentro del pipeline, aplicado solo a los pliegues de entrenamiento para no contaminar la validación (Chawla et al., 2002; Imani et al., 2026).
Métricas: sensibilidad (prioritaria: no dejar pasar casos de riesgo), especificidad, F1, AUC-ROC, AUC-PR (más informativa con clases desbalanceadas) y puntaje de Brier (calibración). La exactitud se reporta, pero no decide, porque un modelo que siempre dice "sano" tendría 91 %.
Criterio de selección: el modelo con mayor AUC-ROC en prueba. Si la diferencia con la logística es menor de 0,01, se prefiere la logística por ser más simple de explicar e implementar en campo.
6.6 Niveles de riesgo
El modelo entrega una probabilidad entre 0 y 1, que se traduce en tres niveles con dos umbrales fijados sobre los datos de validación:
Bajo: probabilidad menor que el umbral de tamizaje (el que logra sensibilidad ≥ 0,80). Recomendación: hábitos saludables y control anual.
Moderado: entre el umbral de tamizaje y el umbral alto. Recomendación: tomar la presión arterial y consultar en el puesto de salud.
Alto: probabilidad mayor que el umbral alto (el que logra especificidad ≥ 0,90). Recomendación: remisión prioritaria a consulta médica.
6.7 Evaluación en contexto rural y colombiano
BRFSS describe a la población de EE. UU., por lo que el componente rural se aborda en dos pasos:
Subgrupo rural: el archivo original del BRFSS incluye la variable calculada _URBSTAT (1 = condado urbano, 2 = condado rural) (CDC, 2022). Se reportarán las métricas por separado para el subgrupo rural; si el desempeño cae de forma importante, se documentará como limitación.
Recalibración a Colombia: el modelo aprende la prevalencia de BRFSS (8,8 %). Para usarlo en Colombia se ajusta el intercepto a la prevalencia local obtenida de RIPS/SISPRO o la CAC, sin reentrenar:
\operatorname{logit}(p_{COL}) = \operatorname{logit}(p_{modelo}) + \ln\frac{\pi_{COL}}{1-\pi_{COL}} - \ln\frac{\pi_{BRFSS}}{1-\pi_{BRFSS}}
donde π es la prevalencia de enfermedad cardiovascular en cada población. Además, la ECV 2023 del DANE permite comparar la distribución de edad, diabetes y tabaquismo entre zona urbana y rural colombiana y la muestra de BRFSS, para anticipar en qué perfiles el modelo podría equivocarse más.
6.8 Consideraciones éticas
La investigación se clasifica "sin riesgo" según el artículo 11 de la Resolución 8430 de 1993, porque usa solo datos secundarios anonimizados y de acceso público. Se respetan la Ley 1581 de 2012 y los términos de uso de cada fuente. El prototipo mostrará siempre que el resultado es orientativo y no reemplaza la valoración médica.
6.9 Cronograma
Fase CRISP-DM
Semanas
Actividad
Entregable
Corte
Comprensión del problema
S1–S2
Propuesta y estado del arte
Documento de propuesta
1
Comprensión de los datos
S2–S5
Recolección, integración y EDA
Notebook de EDA
1
Preparación de los datos
S5–S7
Limpieza y feature engineering
Dataset listo para modelar
2
Modelado
S7–S10
Regresión logística y XGBoost
Modelos entrenados
2
Evaluación
S10–S12
Comparación, rural, recalibración
Tabla comparativa y SHAP
3
Despliegue
S12–S14
Prototipo de interfaz y lineamientos
Prototipo funcional
3
Cierre
S14–S16
Artículo, póster y sustentación
Informe final y demostración
3

## 7. Resultados experimentales (actualización Fase 2)

Esta sección actualiza la propuesta inicial con los resultados obtenidos al ejecutar el pipeline del repositorio (`scripts/run_pipeline.py`).

### 7.1 Datos procesados

Se utilizó el CSV público *Indicators of Heart Disease* (BRFSS cleaned). Tras eliminar 32.133 duplicados y aplicar limpieza:

- Filas finales: **287.662** (positivos ≈ 9,4 %).
- Train / test estratificado (80/20, semilla 42): **230.129 / 57.533**.
- 14 predictores de fácil recolección, alineados a la metodología (§6.3).
- IMC recortado en p1–p99 (17,74–49,23).

Nota sobre la fuente: la propuesta citaba BRFSS 2022 (209.958 filas). Por disponibilidad sin autenticación Kaggle se usó el cleaned público con variables equivalentes; el objetivo `HeartDisease` actúa como proxy del autoreporte de enfermedad coronaria/infarto.

### 7.2 Proceso experimental de hiperparámetros

Validación cruzada estratificada de 5 pliegues + `RandomizedSearchCV` (semilla 42).

| Modelo | Espacio explorado | Mejor CV AUC-ROC | Parámetros ganadores |
|--------|-------------------|-----------------:|----------------------|
| Regresión logística | `C` ∈ logspace(-3, 2) | 0,8307 | `C=0,001`, `class_weight=balanced` |
| XGBoost | n_estimators, max_depth, lr, subsample, colsample, min_child_weight, gamma, reg_lambda (15 iter.) | 0,8336 | depth=3, lr=0,05, n_estimators=300, subsample=1,0, colsample=0,85, λ=5, γ=0,5, scale_pos_weight≈9,59 |

Artefacto: `models/experiment.json`.

### 7.3 Evaluación cuantitativa en prueba

| Modelo | AUC-ROC | IC 95 % (bootstrap) | AUC-PR | F1 (tamizaje) | Sensibilidad | Especificidad | Brier |
|--------|--------:|---------------------|-------:|--------------:|-------------:|--------------:|------:|
| Regresión logística | 0,8323 | [0,8271; 0,8369] | 0,3502 | 0,3489 | 0,8031 | 0,7082 | 0,1716 |
| XGBoost | 0,8355 | [0,8306; 0,8402] | 0,3557 | 0,3523 | 0,8098 | 0,7095 | 0,1724 |

Umbral de tamizaje (sensibilidad ≥ 0,80): logística ≈ 0,47; XGBoost ≈ 0,49. Umbral alto (especificidad ≥ 0,90): ≈ 0,71 / 0,73.

### 7.4 Selección del modelo

Δ AUC-ROC (XGBoost − logística) ≈ **0,0032 < 0,01**. Según el criterio de §6.5 se selecciona la **regresión logística** por interpretabilidad e implementación en campo, sin pérdida práctica de discriminación.

### 7.5 Interpretabilidad

Odds ratios (logística): AgeCategory 2,69; GeneralHealth 1,63; Sex 1,42; HadStroke 1,26; SmokerStatus 1,20; HadDiabetes 1,17; HadKidneyDisease 1,12; DifficultyWalking 1,09.

Importancias XGBoost (ganancia relativa): AgeCategory, DifficultyWalking, GeneralHealth, HadStroke, HadDiabetes, Sex.

### 7.6 Limitaciones observadas

1. Datos de EE. UU.; recalibración a prevalencia colombiana pendiente.
2. Sin `_URBSTAT` en el CSV cleaned → subgrupo rural pendiente.
3. Objetivo autoreportado; tabaquismo sin categoría «exfumador» en esta versión.
4. AUC-PR ~0,35 refleja desbalance; la sensibilidad se priorizó vía umbral.

### 7.7 Prototipo

Interfaz Streamlit (`app/streamlit_app.py`) con las 14 preguntas, probabilidad, nivel de riesgo y aviso de no diagnóstico.

## 8. Conclusiones de la Fase 2

Se ejecutaron limpieza, partición, entrenamiento, tuning y evaluación comparativa. Ambos modelos discriminan bien (AUC-ROC ≈ 0,83). XGBoost mejora de forma marginal; la logística queda como modelo operativo. La Fase 3 abordará recalibración colombiana, análisis rural, SHAP extendido y lineamientos finales.

## Referencias bibliográficas
Agudelo, J., Bedoya, O. F., Muñoz, Ó., Rodríguez Belalcázar, K. D., y Ruiz, Á. J. (2025). Cardiovascular risk estimation in Colombia using artificial intelligence techniques. Cardiology Research and Practice, 2025(1), 2566839. https://doi.org/10.1155/crp/2566839
Ahiduzzaman, M., y Hasan, M. N. (2025). Interpretable machine learning for cardiovascular risk prediction: Insights from NHANES dietary and health data. PLOS One, 20(11), e0335915. https://doi.org/10.1371/journal.pone.0335915
Centers for Disease Control and Prevention. (2022). Calculated variables in the 2021 BRFSS data file. https://www.cdc.gov/brfss/annual_data/2021/pdf/2021-calculated-variables-version4-508.pdf
Centers for Disease Control and Prevention. (2023). Behavioral Risk Factor Surveillance System: 2022 BRFSS survey data and documentation. https://www.cdc.gov/brfss/annual_data/annual_2022.html
Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., y Wirth, R. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.
Chawla, N. V., Bowyer, K. W., Hall, L. O., y Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953
Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). https://doi.org/10.1145/2939672.2939785
Cuenta de Alto Costo. (2023). Situación de la enfermedad renal crónica, la hipertensión arterial y la diabetes mellitus en Colombia 2023. Fondo Colombiano de Enfermedades de Alto Costo.
Departamento Administrativo Nacional de Estadística. (2024). Encuesta Nacional de Calidad de Vida (ECV) 2023. https://www.dane.gov.co
Hernández, J. A., Therán, J. S., Dulcey, L. A., Gómez, J. A., Ochoa, V., y García, C. F. (2025). Perfil de riesgo cardiovascular y análisis epidemiológico en población rural de Colombia. Revista Hispanoamericana de Ciencias de la Salud, 11(1). https://doi.org/10.56239/rhcs.2025.111.911
Hernández Sampieri, R., Fernández Collado, C., y Baptista Lucio, P. (2014). Metodología de la investigación (6.ª ed.). McGraw-Hill.
Imani, M., Maroosi, A., Shojaei, S., Heidari, K., Hoseinzadeh, S. M., Daneshi, N., Saber, Z., Sajadi, N., y Mohammadzadeh, M. (2026). Predicting cardiovascular diseases using imbalanced data: An XGBoost-based analysis of the 2022 BRFSS dataset. American Heart Journal Plus: Cardiology Research and Practice, 63, 100719. https://doi.org/10.1016/j.ahjo.2026.100719
Lundberg, S. M., y Lee, S.-I. (2017). A unified approach to interpreting model predictions. En Advances in Neural Information Processing Systems 30 (pp. 4765–4774).
Ministerio de Salud y Protección Social. (1993). Resolución 8430 de 1993: normas científicas, técnicas y administrativas para la investigación en salud.
Ministerio de Salud y Protección Social. (2017). Día Mundial de la Hipertensión 2017. https://www.minsalud.gov.co/sites/rid/Lists/BibliotecaDigital/RIDE/VS/PP/ENT/dia-mundial-hipertension-2017.pdf
Ministerio de Salud y Protección Social. (2026). Lineamientos para la gestión territorial integral de la salud pública orientados a la prevención y control de la enfermedad cardiovascular, la diabetes mellitus y la enfermedad renal crónica. https://www.minsalud.gov.co/sites/rid/Lists/BibliotecaDigital/RIDE/VS/PP/ENT/lineamientos-gestion-enfermedades-cardiovasculares-metabolicas.pdf
Mosquera Heredia, M. I., Ospino Fernández, L. F., Mosquera Heredia, M. J., y Ramírez Romero, M. (2011). Factores de riesgo cardiovascular en niños de 8-11 años de cinco escuelas públicas de Valledupar, Cesar, Colombia. Revista Médica de Risaralda, 17(1), 13–21. https://dialnet.unirioja.es/servlet/articulo?codigo=3946750
Palacios, C., Morales, J., García, L., y Badiel, M. (2022). Prevalencia y caracterización del riesgo cardiovascular en una población rural. Revista Colombiana de Cardiología, 29(2), 255–262. https://doi.org/10.24875/rccar.m22000144
Pytlak, K. (s. f.). Indicators of Heart Disease (2022 UPDATE) [Conjunto de datos basado en BRFSS 2022]. Kaggle. https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
Rebolledo-Cobos, R., Bracho-Velásquez, L., Amador-Rodero, E., Sarmiento-Rubiano, L., Florez-Pacheco, P., y Pulido-Iriarte, T. (2024). Factores de riesgo cardiovascular en trabajadores del sector salud del caribe colombiano. Revista Sanmartiniana de Ciencias de la Salud, 1, 27–38. https://revista.sanmartin.edu.co/index.php/ciencias_de_la_salud/article/download/7/3/7
Revista Momentos. (2025, 23 de mayo). Cada hora mueren 8 personas en Colombia por enfermedades cardiovasculares. https://revistamomentos.co/cada-hora-mueren-8-personas-en-colombia-por-enfermedades-cardiovasculares/
Xi, Y., Wang, H., y Sun, N. (2022). Machine learning outperforms traditional logistic regression and offers new possibilities for cardiovascular risk prediction: A study involving 143,043 Chinese patients with hypertension. Frontiers in Cardiovascular Medicine, 9, 1025705. https://doi.org/10.3389/fcvm.2022.1025705
Zurique-Sánchez, M. S., Zurique-Sánchez, C. P., Camacho-López, P. A., Sánchez-Sanabria, M., y Hernández-Hernández, S. C. (2019). Prevalencia de hipertensión arterial en Colombia: revisión sistemática y metaanálisis. Acta Médica Colombiana, 44. https://doi.org/10.36104/amc.2019.1293