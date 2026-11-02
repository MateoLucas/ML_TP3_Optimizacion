"""Pipelines de preprocesamiento (todo se ajusta dentro de cada fold -> sin leakage)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLS, ID_COLS, NUMERIC_COLS


def _ohe(min_frequency):
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        sparse_output=False,
    )


def make_preprocessor(scale: bool = True, numeric_cols=None, categorical_cols=None,
                      id_cols=None) -> ColumnTransformer:
    """Preprocesador base.

    - numéricas: imputación por mediana (+ StandardScaler si `scale`)
    - categóricas de baja cardinalidad: one-hot; categorías con < 20 casos se
      agrupan en "infrequent" (evita columnas casi vacías y categorías no vistas)
    - client_region (149 países): one-hot con min_frequency=50
    - IDs de agente / organización: one-hot con min_frequency=100 (sólo los
      agentes/organizaciones grandes tienen columna propia, el resto -> infrequent)
    """
    numeric_cols = NUMERIC_COLS if numeric_cols is None else numeric_cols
    categorical_cols = CATEGORICAL_COLS if categorical_cols is None else categorical_cols
    id_cols = ID_COLS if id_cols is None else id_cols

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    low_card = [c for c in categorical_cols if c != "client_region"]
    transformers = [("num", Pipeline(num_steps), numeric_cols),
                    ("cat", _ohe(20), low_card)]
    if "client_region" in categorical_cols:
        transformers.append(("region", _ohe(50), ["client_region"]))
    if id_cols:
        transformers.append(("ids", _ohe(100), id_cols))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def make_pipeline(model, scale: bool = True, **kw) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(scale=scale, **kw)), ("model", model)])
