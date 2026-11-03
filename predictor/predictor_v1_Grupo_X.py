"""Función predictora del grupo (entregable del Paso 5).

Uso
---
    from predictor.predictor_v1_Grupo_X import predictor_v1_Grupo_X

    y_pred = predictor_v1_Grupo_X("nuevos_datos.csv")          # ruta a un CSV
    y_pred = predictor_v1_Grupo_X(df)                           # o un DataFrame
    proba  = predictor_v1_Grupo_X(df, return_proba=True)        # probabilidad de cancelación

Entrada: datos con las mismas columnas que `gpu_resource_reservations.csv`.
`canceled_job`, `execution_status` y `execution_status_date` pueden estar o no:
se ignoran (las dos últimas tienen leakage del target).

Salida: np.ndarray con 1 = la reserva será cancelada, 0 = no será cancelada
(en el mismo orden que las filas de entrada).

Desde la terminal:
    python predictor/predictor_v1_Grupo_X.py nuevos_datos.csv [salida.csv]

Reemplazar la X del nombre por el número de grupo antes de entregar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # el modelo serializado usa las clases de tp3/
    sys.path.insert(0, str(_ROOT))

from tp3.config import FEATURE_COLS  # noqa: E402
from tp3.data import clean  # noqa: E402

_MODEL_PATH = _ROOT / "models" / "final_model.joblib"
_CONFIG_PATH = _ROOT / "models" / "final_config.json"
_cache = {}


def _load():
    if "model" not in _cache:
        _cache["model"] = joblib.load(_MODEL_PATH)
        _cache["threshold"] = json.loads(_CONFIG_PATH.read_text()).get("threshold", 0.5)
    return _cache["model"], _cache["threshold"]


def predictor_v1_Grupo_X(data, return_proba: bool = False) -> np.ndarray:
    """Predice si cada reserva será cancelada (1) o no (0)."""
    df = pd.read_csv(data) if isinstance(data, (str, Path)) else pd.DataFrame(data).copy()
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en la entrada: {missing}")
    X = clean(df)[FEATURE_COLS]
    model, threshold = _load()
    proba = model.predict_proba(X)[:, 1]
    return proba if return_proba else (proba >= threshold).astype(int)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("uso: python predictor/predictor_v1_Grupo_X.py datos.csv [salida.csv]")
    pred = predictor_v1_Grupo_X(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else "predicciones.csv"
    pd.DataFrame({"canceled_job_pred": pred}).to_csv(out, index=False)
    print(f"{len(pred)} predicciones guardadas en {out} (tasa de cancelación predicha: {pred.mean():.3f})")
