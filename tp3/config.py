"""Configuración global del proyecto: rutas, semilla y grupos de columnas."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "gpu_resource_reservations.csv"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.2  # 1 de cada 5 grupos va a test

TARGET = "canceled_job"

# Columnas que NO se pueden usar como features (data leakage):
#  - execution_status: estado final de la reserva (canceled / completed / no_show).
#    Es una re-codificación exacta del target.
#  - execution_status_date: fecha del último cambio de estado. Para las reservas
#    canceladas es la fecha de cancelación (anterior a la ejecución), por lo que
#    revela el target.
LEAKAGE_COLS = ["execution_status", "execution_status_date"]

# Identificadores de alta cardinalidad (se tratan como categóricos)
ID_COLS = ["orchestration_agent_id", "organization_id"]

CATEGORICAL_COLS = [
    "service_tier",
    "client_region",
    "client_type",
    "request_channel",
    "reserved_gpu_type",
    "assigned_gpu_type",
    "reservation_commitment",
    "workload_profile",
]

NUMERIC_COLS = [
    "hours_before_execution",
    "execution_year",
    "execution_month",
    "execution_week",
    "execution_day",
    "peak_gpu_hours",
    "offpeak_gpu_hours",
    "num_primary_processes",
    "num_auxiliary_processes",
    "num_background_processes",
    "returning_client",
    "previous_job_cancellations",
    "successful_previous_jobs",
    "resource_modifications",
    "queue_wait_time",
    "estimated_cost_per_hour",
    "reserved_storage_units",
    "special_resource_requests",
    "total_processes",
    "total_gpu_hours",
    "high_value_client",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ID_COLS

# Cohortes pedidas explícitamente por la consigna (sección 3.1)
COHORT_COLS = ["client_type", "service_tier", "reserved_gpu_type"]
