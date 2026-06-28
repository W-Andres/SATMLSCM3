# 📦 SAT-ML SCM v2.0

## Sistema Inteligente de Analítica Predictiva Logística basado en Machine Learning para la Optimización de Inventarios

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

---

## Descripción del proyecto

SAT-ML SCM (Satisfaction Machine Learning for Supply Chain Management) es un sistema inteligente de apoyo a la toma de decisiones diseñado para anticipar el riesgo de quiebres de inventario mediante técnicas de Machine Learning.

El proyecto integra variables operativas de la cadena de suministro con indicadores de satisfacción del cliente para estimar la demanda futura y generar alertas tempranas que faciliten la planificación logística.

Este desarrollo fue realizado como trabajo de grado para optar al título de **Ingeniero de Sistemas**.

---

# Objetivos

## Objetivo General

Desarrollar un sistema inteligente basado en Machine Learning que permita predecir la demanda logística y apoyar la toma de decisiones para optimizar la gestión de inventarios.

## Objetivos Específicos

* Generar un histórico de datos logísticos.
* Entrenar modelos de Machine Learning.
* Comparar diferentes algoritmos predictivos.
* Evaluar el desempeño mediante métricas estadísticas.
* Proyectar la demanda para las siguientes 72 horas.
* Identificar riesgos de quiebre de inventario.
* Visualizar resultados mediante un Dashboard interactivo.

---

# Arquitectura del proyecto

```text
SATMLSCM2/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── assets/
├── auth/
├── data/
├── database/
├── exports/
├── models/
├── services/
├── utils/
└── tests/
```

---

# Arquitectura lógica

```text
Usuario

↓

Interfaz Streamlit

↓

Servicios

↓

Modelo Machine Learning

↓

Motor Predictivo

↓

Resultados

↓

Dashboard Ejecutivo
```

---

# Tecnologías utilizadas

| Tecnología   | Uso                       |
| ------------ | ------------------------- |
| Python       | Desarrollo                |
| Streamlit    | Dashboard                 |
| Pandas       | Procesamiento de datos    |
| NumPy        | Cálculo científico        |
| Scikit-Learn | Machine Learning          |
| Plotly       | Visualización             |
| SQLite       | Persistencia              |
| Joblib       | Almacenamiento del modelo |

---

# Flujo del sistema

1. Generación del histórico.
2. Limpieza y preparación de datos.
3. Ingeniería de características.
4. Entrenamiento del modelo.
5. Validación.
6. Predicción.
7. Generación de alertas.
8. Visualización.

---

# Modelo de Machine Learning

El sistema compara varios algoritmos de regresión y selecciona automáticamente el de mejor desempeño.

Modelos evaluados:

* Linear Regression
* Decision Tree
* Random Forest
* Gradient Boosting

La selección se realiza utilizando métricas como:

* MAE
* RMSE
* MAPE
* R²

---

# Variables utilizadas

| Variable             | Tipo       |
| -------------------- | ---------- |
| CEDI                 | Categórica |
| Día de la semana     | Entero     |
| Temperatura promedio | Numérica   |
| Tiempo de tráfico    | Numérica   |
| Stock disponible     | Numérica   |
| Índice SAT           | Numérica   |

Variable objetivo:

**Demanda_Real**

---

# Dashboard

El sistema incluye:

* KPIs ejecutivos
* Pronóstico de demanda
* Comparación entre CEDI
* Importancia de variables
* Alertas semafóricas
* Exportación de resultados

---

# Instalación

Clonar el repositorio:

```bash
git clone https://github.com/W-Andres/SATMLSCM2.git
```

Entrar al proyecto:

```bash
cd SATMLSCM2
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
streamlit run app.py
```

---

# Estructura del modelo

```text
Dataset

↓

Preprocesamiento

↓

Entrenamiento

↓

Validación

↓

Predicción

↓

Dashboard
```

---

# Resultados esperados

El sistema permite:

* Reducir el riesgo de quiebre de inventario.
* Apoyar la toma de decisiones.
* Anticipar la demanda.
* Mejorar la satisfacción del cliente.
* Optimizar el abastecimiento.

---

# Futuras mejoras

* Integración con SAP.
* Integración con Oracle.
* Consumo de API REST.
* Entrenamiento automático.
* Predicción en tiempo real.
* XGBoost.
* SHAP Explainability.
* Docker.
* Kubernetes.

---

# Autor

**Wilson Andrés Carvajal Barreto**

Trabajo de Grado

Ingeniería de Sistemas

Fundación Universitaria Compensar

---

# Licencia

Proyecto desarrollado con fines académicos.

Uso libre para investigación y formación.
