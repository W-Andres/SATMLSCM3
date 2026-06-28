"""
==========================================================
config.py
----------------------------------------------------------
Sistema Inteligente de Analítica Predictiva Logística
Modelo SAT-ML SCM

Trabajo de Grado
Ingeniería de Sistemas

Autor:
Wilson Andrés Carbajal Barreto

Versión:
2.0.0
==========================================================
"""

from pathlib import Path

# ==========================================================
# INFORMACIÓN GENERAL
# ==========================================================

APP_NAME = "SAT-ML SCM"

APP_VERSION = "2.0.0"

AUTHOR = "Wilson Andrés Carbajal Barreto"

UNIVERSIDAD = "Fundación Universitaria Compensar"

PROJECT_DESCRIPTION = (
    "Sistema Inteligente de Analítica Predictiva Logística "
    "basado en Machine Learning para optimización de inventarios."
)

# ==========================================================
# CONFIGURACIÓN STREAMLIT
# ==========================================================

PAGE_TITLE = "SAT-ML SCM"

PAGE_ICON = "📦"

LAYOUT = "wide"

INITIAL_SIDEBAR_STATE = "expanded"

# ==========================================================
# MACHINE LEARNING
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

TARGET_COLUMN = "Demanda_Real"

MODEL_NAME = "Random Forest Regressor"

# Random Forest

RF_PARAMS = {

    "n_estimators": 300,

    "max_depth": 12,

    "min_samples_split": 2,

    "min_samples_leaf": 1,

    "random_state": RANDOM_STATE,

    "n_jobs": -1

}

# ==========================================================
# DATOS
# ==========================================================

DIAS_HISTORICOS = 180

CEDIS = [

    "Soacha",

    "Tenjo"

]

TEMPERATURA_MEDIA = 6.5

TEMPERATURA_STD = 1.2

# ==========================================================
# PROYECCIÓN
# ==========================================================

HORAS_PROYECCION = 72

RIESGO_BAJO = 0.85

RIESGO_MEDIO = 1.00

# ==========================================================
# RUTAS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

LOG_DIR = BASE_DIR / "logs"

EXPORT_DIR = BASE_DIR / "exports"

ASSETS_DIR = BASE_DIR / "assets"

DATABASE_DIR = BASE_DIR / "database"

# Crear carpetas automáticamente

for folder in [

    DATA_DIR,

    MODEL_DIR,

    LOG_DIR,

    EXPORT_DIR,

    DATABASE_DIR,

]:

    folder.mkdir(exist_ok=True)

# ==========================================================
# ARCHIVOS
# ==========================================================

MODEL_FILE = MODEL_DIR / "random_forest.pkl"

DATABASE_FILE = DATABASE_DIR / "satml.db"

LOG_FILE = LOG_DIR / "satml.log"

# ==========================================================
# DASHBOARD
# ==========================================================

PRIMARY_COLOR = "#0F62FE"

SUCCESS_COLOR = "#24A148"

WARNING_COLOR = "#F1C21B"

DANGER_COLOR = "#DA1E28"

# ==========================================================
# AUTENTICACIÓN
# ==========================================================

DEFAULT_USERS = {

    "admin": {

        "password": "admin123",

        "role": "Administrador"

    },

    "analista": {

        "password": "analista123",

        "role": "Analista"

    },

    "coordinador": {

        "password": "coordinador123",

        "role": "Coordinador"

    }

}

# ==========================================================
# EXPORTACIÓN
# ==========================================================

EXPORT_EXCEL = True

EXPORT_PDF = True

EXPORT_CSV = True

# ==========================================================
# LOGGING
# ==========================================================

LOG_LEVEL = "INFO"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(module)s | "
    "%(message)s"
)

# ==========================================================
# MENSAJES DEL SISTEMA
# ==========================================================

WELCOME_MESSAGE = (
    "Bienvenido al Sistema Inteligente de Analítica "
    "Predictiva Logística SAT-ML SCM."
)

FOOTER = (
    "Trabajo de Grado | Ingeniería de Sistemas | "
    "Wilson Andrés Carbajal Barreto | SAT-ML SCM v2.0"
)