"""Carga, limpieza determinística y partición train/test del dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import (
    CATEGORICAL_COLS,
    DATA_PATH,
    FEATURE_COLS,
    ID_COLS,
    LEAKAGE_COLS,
    N_FOLDS,
    RANDOM_STATE,
    TARGET,
)


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza fila a fila, sin estadísticos aprendidos (se puede aplicar antes
    del split y también en inferencia sin riesgo de leakage).

    - elimina las columnas con leakage (execution_status, execution_status_date)
    - categóricas: NaN -> "missing" (client_region tiene 20 nulos)
    - IDs (agente / organización): se convierten a categoría de texto; NaN
      significa "sin agente" / "sin organización" -> "none"
    - el único NaN numérico (num_auxiliary_processes) se deja para el
      SimpleImputer del pipeline, que aprende la mediana sólo con train.
    """
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns]).copy()
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("object").where(df[c].notna(), "missing").astype(str).str.strip()
    for c in ID_COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        df[c] = np.where(v.isna(), "none", v.fillna(-1).astype(int).astype(str))
    return df


def duplicate_groups(df: pd.DataFrame) -> np.ndarray:
    """Un id de grupo por cada combinación única de features.

    ~1/3 del dataset son filas idénticas en todas las features (reservas grupales
    cargadas en lote). Si una copia queda en train y otra en validación, el modelo
    "memoriza" la respuesta y la métrica se infla. Usamos estos ids como `groups`
    para que todas las copias caigan siempre del mismo lado del split.
    """
    h = pd.util.hash_pandas_object(df[FEATURE_COLS], index=False)
    return pd.factorize(h)[0]


def train_test_split_grouped(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """Hold-out 80/20 estratificado por target y agrupado por duplicados."""
    groups = duplicate_groups(df)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(df, df[TARGET], groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def get_data():
    """Devuelve (train, test) ya limpios. Es la única puerta de entrada a los
    datos para todos los pasos, así el test es siempre el mismo."""
    df = clean(load_raw())
    return train_test_split_grouped(df)


def xy(df: pd.DataFrame):
    return df[FEATURE_COLS], df[TARGET].to_numpy()
