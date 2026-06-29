"""
==============================================================
forecasting.py
--------------------------------------------------------------
Servicio de proyección de demanda para SAT-ML SCM

Responsabilidades:
- Cargar un modelo entrenado.
- Generar predicciones.
- Calcular riesgo de quiebre de inventario.
- Calcular stock recomendado.
- Generar alertas.
- Preparar información para el Dashboard.

Autor:
Wilson Andrés Carvajal Barreto
Versión: 2.0
==============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    MODEL_FILE,
    HORAS_PROYECCION,
)

from utils.logger import LoggerManager

logger = LoggerManager.get_logger(__name__)


# ==========================================================
# CLASES
# ==========================================================

@dataclass
class ForecastResult:
    """
    Resultado de una proyección.
    """

    demanda_estimada: float
    stock_actual: float
    stock_recomendado: float
    diferencia: float
    riesgo: str
    alerta: str


# ==========================================================
# SERVICIO
# ==========================================================

class ForecastService:

    def __init__(self):

        self.model = None

        self.load_model()

    # ------------------------------------------------------

    def load_model(self):

        """
        Carga el modelo entrenado.
        """

        if not Path(MODEL_FILE).exists():

            raise FileNotFoundError(

                f"No existe el modelo:\n{MODEL_FILE}"

            )

        self.model = joblib.load(MODEL_FILE)

        logger.info("Modelo cargado correctamente.")

    # ------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:

        """
        Genera predicciones.
        """

        logger.info(

            "Realizando predicción (%s registros)",

            len(X)

        )

        return self.model.predict(X)

    # ------------------------------------------------------

    @staticmethod
    def calcular_stock_recomendado(
        demanda: float,
        seguridad: float = 0.15
    ) -> float:

        """
        Calcula el stock recomendado.

        Se agrega un 15% de stock de seguridad.
        """

        return demanda * (1 + seguridad)

    # ------------------------------------------------------

    @staticmethod
    def evaluar_riesgo(
        stock_actual: float,
        demanda: float
    ) -> tuple[str, str]:

        """
        Evalúa el riesgo de quiebre.
        """

        relacion = stock_actual / demanda

        if relacion >= 1.20:

            return (

                "Bajo",

                "Inventario suficiente."

            )

        elif relacion >= 1:

            return (

                "Medio",

                "Monitorear inventario."

            )

        else:

            return (

                "Alto",

                "Reposición inmediata requerida."

            )

    # ------------------------------------------------------

    def generar_resultado(
        self,
        X: pd.DataFrame,
        stock_actual: float
    ) -> ForecastResult:

        """
        Ejecuta una proyección completa.
        """

        demanda = float(

            self.predict(X)[0]

        )

        recomendado = self.calcular_stock_recomendado(

            demanda

        )

        riesgo, alerta = self.evaluar_riesgo(

            stock_actual,

            demanda

        )

        diferencia = stock_actual - recomendado

        logger.info(

            "Proyección completada."

        )

        return ForecastResult(

            demanda_estimada=round(demanda, 2),

            stock_actual=round(stock_actual, 2),

            stock_recomendado=round(recomendado, 2),

            diferencia=round(diferencia, 2),

            riesgo=riesgo,

            alerta=alerta

        )

    # ------------------------------------------------------

    def proyectar_dataframe(
        self,
        df: pd.DataFrame,
        columna_stock: str = "Stock_Disponible"
    ) -> pd.DataFrame:

        """
        Agrega las proyecciones a un DataFrame.
        """

        datos = df.copy()

        X = datos.drop(

            columns=[

                columna_stock

            ],

            errors="ignore"

        )

        predicciones = self.predict(X)

        datos["Demanda_Predicha"] = predicciones

        datos["Stock_Recomendado"] = (

            predicciones * 1.15

        ).round()

        datos["Riesgo"] = np.where(

            datos[columna_stock]

            >= datos["Stock_Recomendado"],

            "Bajo",

            "Alto"

        )

        return datos

    # ------------------------------------------------------

    @staticmethod
    def resumen(
        df: pd.DataFrame
    ) -> dict:

        """
        Devuelve KPIs para el Dashboard.
        """

        return {

            "Demanda Promedio":

                round(df["Demanda_Predicha"].mean(), 2),

            "Stock Promedio":

                round(df["Stock_Disponible"].mean(), 2),

            "Stock Recomendado":

                round(df["Stock_Recomendado"].mean(), 2),

            "Alertas":

                int(

                    (df["Riesgo"] == "Alto").sum()

                ),

            "Horizonte":

                f"{HORAS_PROYECCION} horas"

        }
