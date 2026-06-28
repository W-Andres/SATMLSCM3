"""
==============================================================
train.py
--------------------------------------------------------------
Entrenamiento de modelos de Machine Learning

SAT-ML SCM

Autor:
Wilson Andrés Carbajal Barreto
Versión: 2.0
==============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from config import (
    MODEL_FILE,
    RANDOM_STATE,
    RF_PARAMS,
    TARGET_COLUMN,
    TEST_SIZE,
)

from models.evaluate import ModelEvaluator

from utils.logger import LoggerManager

logger = LoggerManager.get_logger(__name__)


# ==========================================================
# DATACLASS
# ==========================================================

@dataclass
class TrainingResult:

    model_name: str
    pipeline: Pipeline
    metrics: object
    comparison: pd.DataFrame


# ==========================================================
# TRAINER
# ==========================================================

class ModelTrainer:

    """
    Entrena varios modelos y selecciona automáticamente
    el de mejor desempeño.
    """

    def __init__(self):

        self.best_model = None

        self.best_pipeline = None

    # ------------------------------------------------------

    @staticmethod
    def split_dataset(
        df: pd.DataFrame
    ):

        X = df.drop(columns=[TARGET_COLUMN])

        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE

        )

        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------

    @staticmethod
    def preprocessing(X):

        numeric = X.select_dtypes(

            include=["number"]

        ).columns

        categorical = X.select_dtypes(

            exclude=["number"]

        ).columns

        numeric_pipeline = Pipeline(

            steps=[

                (

                    "imputer",

                    SimpleImputer(

                        strategy="median"

                    )

                )

            ]

        )

        categorical_pipeline = Pipeline(

            steps=[

                (

                    "imputer",

                    SimpleImputer(

                        strategy="most_frequent"

                    )

                ),

                (

                    "encoder",

                    OneHotEncoder(

                        handle_unknown="ignore"

                    )

                )

            ]

        )

        return ColumnTransformer(

            [

                (

                    "numeric",

                    numeric_pipeline,

                    numeric

                ),

                (

                    "categorical",

                    categorical_pipeline,

                    categorical

                )

            ]

        )

    # ------------------------------------------------------

    def candidate_models(self):

        return {

            "Linear Regression":

                LinearRegression(),

            "Decision Tree":

                DecisionTreeRegressor(

                    random_state=RANDOM_STATE

                ),

            "Gradient Boosting":

                GradientBoostingRegressor(

                    random_state=RANDOM_STATE

                ),

            "Random Forest":

                RandomForestRegressor(

                    **RF_PARAMS

                )

        }

    # ------------------------------------------------------

    @staticmethod
    def optimize_random_forest(

        pipeline,

        X_train,

        y_train

    ):

        parameters = {

            "model__n_estimators": [200,300,500],

            "model__max_depth": [8,10,12,None],

            "model__min_samples_leaf":[1,2,4]

        }

        search = GridSearchCV(

            estimator=pipeline,

            param_grid=parameters,

            scoring="r2",

            cv=5,

            n_jobs=-1

        )

        search.fit(

            X_train,

            y_train

        )

        logger.info(

            "GridSearch finalizado."

        )

        return search.best_estimator_

    # ------------------------------------------------------

    def train(

        self,

        df: pd.DataFrame

    ) -> TrainingResult:

        logger.info(

            "Entrenamiento iniciado."

        )

        X_train, X_test, y_train, y_test = (

            self.split_dataset(df)

        )

        preprocessor = self.preprocessing(

            X_train

        )

        resultados = {}

        pipelines = {}

        for nombre, modelo in (

            self.candidate_models().items()

        ):

            pipeline = Pipeline(

                [

                    (

                        "preprocessor",

                        preprocessor

                    ),

                    (

                        "model",

                        modelo

                    )

                ]

            )

            if nombre == "Random Forest":

                pipeline = self.optimize_random_forest(

                    pipeline,

                    X_train,

                    y_train

                )

            metrics = (

                ModelEvaluator.evaluate_model(

                    pipeline,

                    X_train,

                    X_test,

                    y_train,

                    y_test

                )

            )

            resultados[nombre] = metrics

            pipelines[nombre] = pipeline

            logger.info(

                "%s entrenado.",

                nombre

            )

        comparison = (

            ModelEvaluator.compare_models(

                resultados

            )

        )

        best_name = (

            ModelEvaluator.best_model(

                comparison

            )

        )

        self.best_pipeline = (

            pipelines[best_name]

        )

        self.best_model = best_name

        Path(MODEL_FILE).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        joblib.dump(

            self.best_pipeline,

            MODEL_FILE

        )

        logger.info(

            "Mejor modelo: %s",

            best_name

        )

        logger.info(

            "Modelo almacenado en %s",

            MODEL_FILE

        )

        return TrainingResult(

            model_name=best_name,

            pipeline=self.best_pipeline,

            metrics=resultados[best_name],

            comparison=comparison

        )