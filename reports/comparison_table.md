| Modelo | AUC-ROC | IC95% | AUC-PR | F1 (tamizaje) | Sens. | Espec. | Brier |
|---|---:|---|---:|---:|---:|---:|---:|
| Regresión logística | 0.8323 | [0.8261, 0.8385] | 0.3515 | 0.3471 | 0.8082 | 0.7031 | 0.1730 |
| XGBoost | 0.8357 | [0.8300, 0.8418] | 0.3540 | 0.3532 | 0.8033 | 0.7138 | 0.1738 |

**Modelo seleccionado:** Regresión logística. Diferencia de AUC-ROC < 0.01; se prefiere la logística por interpretabilidad.