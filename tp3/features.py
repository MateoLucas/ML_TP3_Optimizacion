"""Feature engineering (Paso 4.1).

* `FeatureEngineer`: features nuevas calculadas fila a fila (sin estado).
* `GroupStats`: features que necesitan estadísticos del train (frecuencias,
  precio relativo). Se ajusta dentro del Pipeline -> sin leakage entre folds.
* `make_fe_pipeline`: Pipeline completo FE + preprocesamiento + modelo, con
  target encoding para las categóricas de alta cardinalidad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from .config import CATEGORICAL_COLS, ID_COLS, NUMERIC_COLS, RANDOM_STATE

HIGH_CARD = ["client_region"] + ID_COLS

ENGINEERED_NUMERIC = [
    "gpu_type_changed",       # GPU asignada != reservada
    "has_agent",              # la reserva pasó por un agente
    "has_organization",       # la reserva está asociada a una organización
    "has_history",            # el cliente tiene historial previo
    "prev_cancel_ratio",      # cancelaciones / (cancelaciones + éxitos) previos
    "log_lead_time",          # log1p(hours_before_execution)
    "peak_fraction",          # fracción de horas GPU en período pico
    "total_cost",             # costo/h * horas GPU totales
    "cost_per_process",       # costo/h por proceso
    "execution_dayofweek",    # día de la semana del inicio de ejecución
    "execution_dayofyear",
    "is_empty_workload",      # 0 procesos o 0 horas (anomalía detectada en el EDA)
    "has_special_requests",
    "is_domestic",            # región PRT (mercado local, 39 % del dataset)
]
GROUP_NUMERIC = [
    "region_freq", "agent_freq", "organization_freq",   # frecuencia en train
    "relative_cost",          # costo / mediana del costo del mismo tipo de GPU y mes
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Features derivadas fila a fila (no aprende nada)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["gpu_type_changed"] = (X.reserved_gpu_type != X.assigned_gpu_type).astype(int)
        X["has_agent"] = (X.orchestration_agent_id != "none").astype(int)
        X["has_organization"] = (X.organization_id != "none").astype(int)
        prev = X.previous_job_cancellations + X.successful_previous_jobs
        X["has_history"] = (prev > 0).astype(int)
        X["prev_cancel_ratio"] = np.where(prev > 0, X.previous_job_cancellations / prev.clip(lower=1), 0.0)
        X["log_lead_time"] = np.log1p(X.hours_before_execution)
        X["peak_fraction"] = X.peak_gpu_hours / X.total_gpu_hours.clip(lower=1)
        X["total_cost"] = X.estimated_cost_per_hour * X.total_gpu_hours
        X["cost_per_process"] = X.estimated_cost_per_hour / X.total_processes.fillna(1).clip(lower=1)
        date = pd.to_datetime(dict(year=X.execution_year, month=X.execution_month, day=X.execution_day),
                              errors="coerce")
        X["execution_dayofweek"] = date.dt.dayofweek.fillna(0).astype(int)
        X["execution_dayofyear"] = date.dt.dayofyear.fillna(1).astype(int)
        X["is_empty_workload"] = ((X.total_processes.fillna(0) == 0) | (X.total_gpu_hours == 0)).astype(int)
        X["has_special_requests"] = (X.special_resource_requests > 0).astype(int)
        X["is_domestic"] = (X.client_region == "PRT").astype(int)
        return X


class GroupStats(BaseEstimator, TransformerMixin):
    """Estadísticos aprendidos en train: frecuencia de región/agente/organización
    y costo relativo a la mediana del mismo (tipo de GPU, mes)."""

    def fit(self, X, y=None):
        self.freq_ = {c: X[c].value_counts(normalize=True) for c in HIGH_CARD}
        self.cost_med_ = X.groupby(["reserved_gpu_type", "execution_month"]).estimated_cost_per_hour.median()
        self.cost_med_gpu_ = X.groupby("reserved_gpu_type").estimated_cost_per_hour.median()
        self.cost_med_all_ = X.estimated_cost_per_hour.median()
        return self

    def transform(self, X):
        X = X.copy()
        for c, name in zip(HIGH_CARD, ["region_freq", "agent_freq", "organization_freq"]):
            X[name] = X[c].map(self.freq_[c]).astype(float).fillna(0.0)
        key = pd.MultiIndex.from_arrays([X.reserved_gpu_type, X.execution_month])
        med = pd.Series(self.cost_med_.reindex(key).to_numpy(), index=X.index)
        med = med.fillna(X.reserved_gpu_type.map(self.cost_med_gpu_)).fillna(self.cost_med_all_)
        X["relative_cost"] = X.estimated_cost_per_hour / med.clip(lower=0.01)
        return X


def make_fe_preprocessor(scale: bool = True, high_card: str = "target", drop=()):
    """Preprocesador con features nuevas.

    high_card="target": TargetEncoder (con cross-fitting interno, evita leakage)
    para region / agente / organización; "onehot": igual que el baseline.
    `drop`: columnas a excluir (selección de variables).
    """
    num = [c for c in NUMERIC_COLS + ENGINEERED_NUMERIC + GROUP_NUMERIC if c not in drop]
    low = [c for c in CATEGORICAL_COLS if c != "client_region" and c not in drop]
    hc = [c for c in HIGH_CARD if c not in drop]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    ohe = lambda f: OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=f, sparse_output=False)  # noqa: E731
    transformers = [("num", Pipeline(num_steps), num), ("cat", ohe(20), low)]
    if high_card == "target":
        te = TargetEncoder(target_type="binary",
                           cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE))
        transformers.append(("te", Pipeline([("te", te), ("scale", StandardScaler())]) if scale else te, hc))
    else:
        transformers.append(("region", ohe(50), [c for c in hc if c == "client_region"]))
        transformers.append(("ids", ohe(100), [c for c in hc if c in ID_COLS]))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_fe_pipeline(model, scale: bool = True, high_card: str = "target", drop=()) -> Pipeline:
    return Pipeline([
        ("fe", FeatureEngineer()),
        ("stats", GroupStats()),
        ("prep", make_fe_preprocessor(scale=scale, high_card=high_card, drop=drop)),
        ("model", model),
    ])
