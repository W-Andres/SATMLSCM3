"""
==============================================================
generator.py
--------------------------------------------------------------
Generador de datos sintéticos para SAT-ML SCM

Este módulo genera el histórico de operaciones logísticas que
alimenta el modelo de Machine Learning.

En producción esta capa podrá reemplazarse por una conexión
directa al ERP o WMS corporativo.

Autor:
Wilson Andrés Carbajal Barreto

Versión:
2.0
==============================================================
"""

from datetime import datetime, timedelta
from pathlib import Path

import logging
import numpy as np
import pandas as pd

from config import (
    RANDOM_STATE,
    DIAS_HISTORICOS,
    CEDIS,
    TEMPERATURA_MEDIA,
    TEMPERATURA_STD,
)

# ==========================================================
# LOGGER
# ==========================================================

logger = logging.getLogger(__name__)


# ==========================================================
# GENERADOR DE DATOS
# ==========================================================

class DataGenerator:
    """
    Generador del histórico logístico para SAT-ML.
    """

    def __init__(
        self,
        dias: int = DIAS_HISTORICOS,
        random_state: int = RANDOM_STATE
    ) -> None:

        self.dias = dias
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------

    def _base_demanda(self, cedi: str) -> int:

        demanda = {

            "Soacha": 280,

            "Tenjo": 190

        }

        return demanda.get(cedi, 200)

    # ------------------------------------------------------

    def generar(self) -> pd.DataFrame:
        """
        Genera un DataFrame con el histórico logístico.

        Returns
        -------
        DataFrame
        """

        registros = []

        fecha_inicio = (
            datetime.today()
            - timedelta(days=self.dias)
        )

        for cedi in CEDIS:

            base = self._base_demanda(cedi)

            for i in range(self.dias):

                fecha = fecha_inicio + timedelta(days=i)

                dia_semana = fecha.weekday()

                # Mayor demanda viernes y sábado

                estacionalidad = (
                    1.18
                    if dia_semana in [4, 5]
                    else 1
                )

                tendencia = 1 + (i / self.dias) * 0.12

                ruido = self.rng.normal(0, 18)

                demanda = max(
                    30,
                    base * estacionalidad * tendencia + ruido
                )

                temperatura = self.rng.normal(
                    TEMPERATURA_MEDIA,
                    TEMPERATURA_STD
                )

                trafico = self.rng.normal(
                    35 if cedi == "Soacha" else 22,
                    8
                )

                satisfaccion = np.clip(

                    self.rng.normal(4.2, 0.45),

                    1,

                    5

                )

                stock = demanda * self.rng.uniform(

                    0.85,

                    1.25

                )

                registros.append(

                    {

                        "Fecha": fecha,

                        "CEDI": cedi,

                        "Dia_Semana": dia_semana,

                        "Temperatura_Promedio_C":

                            round(temperatura, 1),

                        "Tiempo_Trafico_Min":

                            round(max(5, trafico), 1),

                        "Stock_Disponible":

                            round(stock),

                        "Indice_Satisfaccion_SAT":

                            round(satisfaccion, 2),

                        "Demanda_Real":

                            round(demanda)

                    }

                )

        df = pd.DataFrame(registros)

        logger.info(

            "Histórico generado correctamente (%s registros)",

            len(df)

        )

        return df

    # ------------------------------------------------------

    @staticmethod
    def exportar_csv(
        df: pd.DataFrame,
        ruta: str
    ) -> None:

        Path(ruta).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(

            ruta,

            index=False,

            encoding="utf-8-sig"

        )

        logger.info(

            "Dataset exportado a %s",

            ruta

        )


# ==========================================================
# FUNCIÓN AUXILIAR
# ==========================================================

def cargar_datos() -> pd.DataFrame:
    """
    Genera el histórico y devuelve un DataFrame.
    """

    generator = DataGenerator()

    return generator.generar()


# ==========================================================
# EJECUCIÓN LOCAL
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(levelname)s - %(message)s"

    )

    datos = cargar_datos()

    print(datos.head())

    print()

    print(datos.describe())

    DataGenerator.exportar_csv(

        datos,

        "data/historico_satml.csv"

    )