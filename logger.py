"""
==========================================================
logger.py
----------------------------------------------------------
Sistema de registro de eventos para SAT-ML SCM

Permite registrar información, advertencias y errores
en archivo y consola.

Autor:
Wilson Andrés Carbajal Barreto

Versión:
2.0
==========================================================
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOG_DIR, LOG_FILE, LOG_LEVEL, LOG_FORMAT


class LoggerManager:
    """
    Configura y administra el sistema de logging.

    Uso:
        logger = LoggerManager().get_logger(__name__)
        logger.info("Aplicación iniciada")
    """

    _configured = False

    @classmethod
    def configure(cls) -> None:
        """
        Configura el sistema de logging solo una vez.
        """

        if cls._configured:
            return

        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger()
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

        # Evita agregar handlers duplicados
        logger.handlers.clear()

        formatter = logging.Formatter(LOG_FORMAT)

        # ==========================
        # Archivo con rotación
        # ==========================
        file_handler = RotatingFileHandler(
            filename=LOG_FILE,
            maxBytes=5 * 1024 * 1024,   # 5 MB
            backupCount=5,
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)

        # ==========================
        # Consola
        # ==========================
        console_handler = logging.StreamHandler()

        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        cls._configured = True

        logger.info("=" * 60)
        logger.info("SAT-ML SCM iniciado")
        logger.info("=" * 60)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Devuelve un logger listo para usar.

        Parameters
        ----------
        name : str
            Nombre del módulo.

        Returns
        -------
        logging.Logger
        """

        LoggerManager.configure()

        return logging.getLogger(name)