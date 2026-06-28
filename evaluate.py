"""
==============================================================
evaluate.py
--------------------------------------------------------------
Evaluación de modelos de Machine Learning

SAT-ML SCM

Autor:
Wilson Andrés Carbajal Barreto

Versión:
2.0
==============================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import cross_val_score

from utils.logger import LoggerManager

logger = LoggerManager.get_logger(__name__)


# ==========================================================
# DATACLASS
# ==========================================================

@dataclass
class EvaluationResult:

    mae: float
    mse: float
    rmse: float
    mape: float
    r2: float
    cv_mean: float
    cv_std: float


# ==========================================================
# EVALUADOR
# ==========================================================

class ModelEvaluator:

    """
    Clase para evaluar modelos de regresión.
    """

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> EvaluationResult:

        """
        Calcula métricas principales.
        """

        mae = mean_absolute_error(y_true, y_pred)

        mse = mean_squared_error(y_true, y_pred)

        rmse = np.sqrt(mse)

        mape = np.mean(

            np.abs((y_true - y_pred) / y_true)

        ) * 100

        r2 = r2_score(y_true, y_pred)

        logger.info("Métricas calculadas.")

        return EvaluationResult(

            mae=mae,

            mse=mse,

            rmse=rmse,

            mape=mape,

            r2=r2,

            cv_mean=np.nan,

            cv_std=np.nan

        )

    # ------------------------------------------------------

    @staticmethod
    def cross_validation(
        model,
        X,
        y,
        cv: int = 5
    ) -> tuple:

        """
        Ejecuta Validación Cruzada.
        """

        scores = cross_val_score(

            model,

            X,

            y,

            cv=cv,

            scoring="r2"

        )

        logger.info(

            "Cross Validation completado."

        )

        return (

            scores.mean(),

            scores.std()

        )

    # ------------------------------------------------------

    @staticmethod
    def evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    ) -> EvaluationResult:

        """
        Entrena y evalúa un modelo.
        """

        model.fit(

            X_train,

            y_train

        )

        predictions = model.predict(

            X_test

        )

        metrics = ModelEvaluator.calculate_metrics(

            y_test,

            predictions

        )

        cv_mean, cv_std = (

            ModelEvaluator.cross_validation(

                model,

                X_train,

                y_train

            )

        )

        metrics.cv_mean = cv_mean

        metrics.cv_std = cv_std

        logger.info(

            "Modelo evaluado correctamente."

        )

        return metrics

    # ------------------------------------------------------

    @staticmethod
    def compare_models(

        results: dict

    ) -> pd.DataFrame:

        """
        Convierte un diccionario de resultados
        en DataFrame ordenado.
        """

        filas = []

        for nombre, resultado in results.items():

            filas.append(

                {

                    "Modelo": nombre,

                    "MAE": resultado.mae,

                    "RMSE": resultado.rmse,

                    "MAPE (%)": resultado.mape,

                    "R²": resultado.r2,

                    "CV Mean": resultado.cv_mean,

                    "CV Std": resultado.cv_std

                }

            )

        df = pd.DataFrame(filas)

        return df.sort_values(

            "R²",

            ascending=False

        )

    # ------------------------------------------------------

    @staticmethod
    def best_model(

        comparison: pd.DataFrame

    ) -> str:

        """
        Devuelve el mejor modelo.
        """

        return comparison.iloc[0]["Modelo"]