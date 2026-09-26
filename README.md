# Predicción de Incumplimiento de Crédito

Proyecto de Machine Learning para predecir si un cliente cumplirá o no con el pago de su crédito, usando el dataset "Default of Credit Card Clients" del UCI Machine Learning Repository.

## Descripción del proyecto

El objetivo es construir un modelo de clasificación binaria que, a partir de información demográfica, historial de pagos y montos de facturación de clientes de un banco en Taiwán, prediga la probabilidad de que un cliente incumpla su pago al mes siguiente.

## Dataset

- **Fuente:** [UCI Machine Learning Repository - Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Tamaño:** 30,000 registros, 23 variables explicativas más la variable objetivo
- **Variable objetivo:** `default payment next month` (1 = incumplió, 0 = pagó)

El archivo de datos no se incluye en este repositorio. Para obtenerlo, descarga el `.zip` desde el enlace anterior y colócalo en la carpeta `data/` (o donde prefieras organizarlo).

## Estructura del proyecto

```
credito-ml/
├── notebook.ipynb        # Notebook principal con todo el análisis
├── README.md              # Este archivo
├── requirements.txt        # Dependencias del proyecto
└── data/                  # Carpeta local para el dataset (no versionada)
```

## Metodología

1. Análisis exploratorio de datos (EDA): distribución del target, detección de categorías no documentadas en `EDUCATION` y `MARRIAGE`, identificación de registros duplicados.
2. Limpieza y preprocesamiento: eliminación de duplicados, consolidación de categorías no documentadas, codificación one-hot de variables categóricas nominales.
3. División de los datos en entrenamiento y prueba, con estratificación por la variable objetivo dado el desbalance de clases (~78% / 22%).
4. Entrenamiento y comparación de tres modelos: regresión logística, Random Forest y Gradient Boosting.
5. Ajuste de hiperparámetros mediante validación cruzada (`RandomizedSearchCV`) optimizando ROC AUC.
6. Evaluación con métricas apropiadas para clases desbalanceadas: precisión, recall, F1 y ROC AUC, priorizando el recall de la clase de incumplimiento por su relevancia en el contexto de riesgo crediticio.
7. Interpretación del modelo final mediante permutation importance.

## Resultados

| Modelo | Precision (clase 1) | Recall (clase 1) | F1 (clase 1) | ROC AUC |
|---|---|---|---|---|
| Regresión logística | 0.70 | 0.23 | 0.35 | 0.727 |
| Regresión logística (balanced) | 0.38 | 0.64 | 0.48 | 0.728 |
| Random Forest (balanced) | 0.51 | 0.59 | 0.55 | 0.779 |
| Random Forest (ajustado) | 0.53 | 0.59 | 0.56 | 0.781 |
| **Gradient Boosting (modelo final)** | 0.48 | 0.62 | 0.54 | **0.784** |

Las variables más relevantes según el análisis de importancia fueron el historial de pago más reciente (`PAY_0`), el monto de la factura más reciente (`BILL_AMT1`) y el límite de crédito (`LIMIT_BAL`).

## Limitaciones y próximos pasos

- El modelo aún no identifica un porcentaje importante de los clientes que incurren en incumplimiento (recall de 0.62), por lo que requeriría un análisis adicional del costo de negocio antes de implementarse en producción.
- El desempeño alcanzó una meseta (~0.78-0.784 de ROC AUC) que no mejoró significativamente con ajuste de hiperparámetros ni cambio de algoritmo, lo que sugiere que las mejoras futuras deberían enfocarse en la ingeniería de variables.
- Próximos pasos sugeridos: creación de nuevas variables a partir del historial de pagos, ajuste del umbral de decisión según el costo de negocio, evaluación de técnicas de sobremuestreo (SMOTE), y validación con datos más recientes.

## Cómo reproducir el proyecto

```bash
conda create --name credito-ml python=3.11
conda activate credito-ml
pip install -r requirements.txt
python -m ipykernel install --user --name=credito-ml --display-name "Python (credito-ml)"
jupyter notebook
```

## Autor

Proyecto desarrollado por Ale como ejercicio de aprendizaje de Machine Learning aplicado a riesgo crediticio.
