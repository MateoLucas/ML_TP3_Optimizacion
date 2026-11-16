"""Paso 6 – Exporta el modelo final a un predictor AUTOCONTENIDO.

Lee el pipeline entrenado en el Paso 5 (models/final_model.joblib) y escribe
todos sus parámetros aprendidos (medianas, frecuencias, categorías del one-hot,
target encoding y los árboles de LightGBM) dentro de
predictor/predictor_v1_Grupo_12.py, que así se puede ejecutar en cualquier
carpeta sólo con numpy y pandas (sin sklearn, sin lightgbm, sin pickle, sin
este repositorio).

Después verifica que el predictor exportado reproduce EXACTAMENTE al pipeline
original (probabilidades y predicciones), también con categorías no vistas y
valores faltantes.
"""
import base64
import importlib.util
import json
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from tp3.config import FEATURE_COLS, MODEL_DIR, ROOT
from tp3.data import clean, load_raw

PRED_PATH = ROOT / "predictor" / "predictor_v1_Grupo_12.py"

pipe = joblib.load(MODEL_DIR / "final_model.joblib")
cfg = json.loads((MODEL_DIR / "final_config.json").read_text())
stats, prep, lgbm = pipe.named_steps["stats"], pipe.named_steps["prep"], pipe.named_steps["model"]
assert [n for n, _ in pipe.steps] == ["fe", "stats", "prep", "model"]


def tolist(a):
    return [x.item() if hasattr(x, "item") else x for x in a]


# ------------------------------------------------------------------ preprocesamiento
num_pipe = prep.named_transformers_["num"]
num_cols = list(num_pipe.feature_names_in_)
assert [n for n, _ in num_pipe.steps] == ["impute"], "el modelo final no escala (LightGBM)"
num_medians = tolist(num_pipe.named_steps["impute"].statistics_)

ohe = prep.named_transformers_["cat"]
ohe_payload = [{"col": c, "categories": tolist(cats), "infrequent": [] if inf is None else tolist(inf)}
               for c, cats, inf in zip(ohe.feature_names_in_, ohe.categories_, ohe.infrequent_categories_)]

te = prep.named_transformers_["te"]
te_payload = [{"col": c, "categories": tolist(cats), "encodings": tolist(enc)}
              for c, cats, enc in zip(te.feature_names_in_, te.categories_, te.encodings_)]

order = [n for n, _, _ in prep.transformers_ if n != "remainder"]
assert order == ["num", "cat", "te"], order

freq = {c: {str(k): float(v) for k, v in s.items()} for c, s in stats.freq_.items()}
cost_med = {f"{g}|{int(mth)}": float(v) for (g, mth), v in stats.cost_med_.items()}
cost_med_gpu = {str(k): float(v) for k, v in stats.cost_med_gpu_.items()}

# ------------------------------------------------------------------ árboles de LightGBM
model_str = lgbm.booster_.model_to_string()
objective = re.search(r"^objective=(.*)$", model_str, re.M).group(1)
assert objective.startswith("binary"), objective
sigmoid = float(re.search(r"sigmoid:([0-9.]+)", objective).group(1))
assert "average_output" not in model_str


def parse_array(block, key, cast):
    m = re.search(rf"^{key}=(.*)$", block, re.M)
    return [cast(v) for v in m.group(1).split()] if m else []


trees = []
for block in re.split(r"^Tree=\d+$", model_str.split("end of trees")[0], flags=re.M)[1:]:
    assert int(re.search(r"^num_cat=(\d+)$", block, re.M).group(1)) == 0, "no se esperan splits categóricos"
    dt = parse_array(block, "decision_type", int)
    assert all(d & 1 == 0 for d in dt)
    trees.append({
        "feature": parse_array(block, "split_feature", int),
        "threshold": parse_array(block, "threshold", float),
        "default_left": [int(bool(d & 2)) for d in dt],
        "missing_type": [(d >> 2) & 3 for d in dt],
        "left": parse_array(block, "left_child", int),
        "right": parse_array(block, "right_child", int),
        "leaf_value": parse_array(block, "leaf_value", float),
    })
assert len(trees) == lgbm.booster_.num_trees(), (len(trees), lgbm.booster_.num_trees())

payload = {
    "meta": {"grupo": 12, "modelo": cfg["name"], "entrenado_con_filas": len(load_raw()), "n_arboles": len(trees),
             "features": list(prep.get_feature_names_out())},
    "threshold": float(cfg["threshold"]),
    "sigmoid": sigmoid,
    "num_cols": num_cols, "num_medians": num_medians,
    "ohe": ohe_payload,
    "te": te_payload, "te_default": float(te.target_mean_),
    "freq": freq, "cost_med": cost_med, "cost_med_gpu": cost_med_gpu,
    "cost_med_all": float(stats.cost_med_all_),
    "trees": trees,
}
blob = base64.b64encode(zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), 9)).decode()
lines = [blob[i:i + 120] for i in range(0, len(blob), 120)]
payload_src = "_PAYLOAD = [\n" + "".join(f'    "{ln}",\n' for ln in lines) + "]\n"

src = PRED_PATH.read_text()
src = re.sub(r"(# --- BEGIN MODEL PAYLOAD ---\n).*?(# --- END MODEL PAYLOAD ---)",
             lambda m: m.group(1) + payload_src + m.group(2), src, flags=re.S)
PRED_PATH.write_text(src)
print(f"predictor escrito: {PRED_PATH} ({PRED_PATH.stat().st_size / 1e6:.2f} MB, {len(trees)} árboles)")

# ------------------------------------------------------------------ verificación
spec = importlib.util.spec_from_file_location("pred_g12", PRED_PATH)
pred_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pred_mod)

raw = load_raw()
X_raw = raw.drop(columns=["canceled_job"])


def reference(df_raw):
    return pipe.predict_proba(clean(df_raw)[FEATURE_COLS])[:, 1]


def compare(name, df_raw):
    m = pred_mod._load_model()
    F_new = pred_mod._design_matrix(pred_mod._engineer(pred_mod._prepare_input(df_raw), m), m)
    F_ref = pipe[:-1].transform(clean(df_raw)[FEATURE_COLS])
    p_new = pred_mod.predict_proba_Grupo_12(df_raw)
    p_ref = reference(df_raw)
    y_new = pred_mod.predictor_v1_Grupo_12(df_raw)
    y_ref = (p_ref >= cfg["threshold"]).astype(int)
    r = {"caso": name, "filas": len(df_raw), "max |Δ features|": float(np.nanmax(np.abs(F_new - F_ref))),
         "max |Δ proba|": float(np.max(np.abs(p_new - p_ref))), "predicciones distintas": int((y_new != y_ref).sum())}
    print(r)
    assert r["max |Δ features|"] < 1e-9 and r["max |Δ proba|"] < 1e-9 and r["predicciones distintas"] == 0, r
    return r


checks = [compare("dataset completo", X_raw)]

# casos borde: categorías nunca vistas, IDs nuevos, faltantes en numéricas y categóricas
rng = np.random.default_rng(0)
E = X_raw.sample(500, random_state=0).reset_index(drop=True)
idx = lambda k: rng.choice(len(E), k, replace=False)  # noqa: E731
E.loc[idx(50), "client_region"] = "ZZZ"
E.loc[idx(50), "client_type"] = "nuevo_tipo"
E.loc[idx(50), "assigned_gpu_type"] = "GPU_T99"
E.loc[idx(50), "reserved_gpu_type"] = "GPU_T99"
E.loc[idx(50), "orchestration_agent_id"] = 99999
E.loc[idx(50), "organization_id"] = 88888
E.loc[idx(50), "service_tier"] = np.nan
for c in ["hours_before_execution", "estimated_cost_per_hour", "num_auxiliary_processes", "execution_month",
          "special_resource_requests", "total_gpu_hours"]:
    E.loc[idx(40), c] = np.nan
checks.append(compare("casos borde (categorías nuevas y faltantes)", E))
checks.append(compare("una sola fila", X_raw.iloc[[123]]))
shuffled = X_raw.sample(1000, random_state=1)            # índice desordenado
checks.append(compare("índice desordenado", shuffled))
pd.DataFrame(checks).to_csv(ROOT / "reports" / "tables" / "06_verificacion_predictor.csv", index=False)
(ROOT / "reports" / "tables" / "06_verificacion_predictor.md").write_text(
    pd.DataFrame(checks).to_markdown(index=False, floatfmt=".2e"))
print("OK: el predictor autocontenido reproduce exactamente al pipeline entrenado.")
