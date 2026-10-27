| Modelo | AUC-ROC | IC95% | AUC-PR | F1 (tamizaje) | Sens. | Espec. | Brier |
|---|---:|---|---:|---:|---:|---:|---:|
| Regresión logística | 0.8323 | [0.8271, 0.8369] | 0.3502 | 0.3489 | 0.8031 | 0.7082 | 0.1716 |
| XGBoost | 0.8355 | [0.8306, 0.8402] | 0.3557 | 0.3523 | 0.8098 | 0.7095 | 0.1724 |

**Modelo seleccionado:** Regresión logística. Diferencia de AUC-ROC < 0.01; se prefiere la logística por interpretabilidad.