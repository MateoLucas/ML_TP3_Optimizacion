"""Paso 4 – Estrategia 5: Data-centered improvements.

Sobre el LightGBM tuneado se evalúan intervenciones sobre los DATOS DE
ENTRENAMIENTO (los folds de validación quedan intactos, así la comparación es justa):
  a. desbalance: class_weight="balanced"
  b. umbral de decisión optimizado para F1 (elegido sin mirar el fold evaluado)
  c. deduplicar el train (una fila por grupo de duplicados, con la etiqueta mayoritaria)
  d. eliminar del train las filas anómalas del EDA (0 procesos / 0 horas)
  e. sobre-ponderar la cohorte foco del Paso 3 (cloud_marketplace & no_commitment)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from tp3 import plotting as P
from tp3.config import MODEL_DIR, RANDOM_STATE
from tp3.data import duplicate_groups, get_data, xy
from tp3.evaluation import cohort_report, cross_validate, fmt_table, get_cv, save_table, summarize
from tp3.features import make_fe_pipeline

train, _ = get_data()
X, y = xy(train)
groups = duplicate_groups(train)
cv = get_cv()
splits = list(cv.split(X, y, groups))
DROP = json.loads((MODEL_DIR / "feature_selection.json").read_text())["drop"]
HPO = json.loads((MODEL_DIR / "hpo_params.json").read_text())
RS = RANDOM_STATE


def lgbm_pipe(**kw):
    params = {**HPO["lightgbm"], **kw}
    return make_fe_pipeline(LGBMClassifier(verbose=-1, random_state=RS, n_jobs=1, subsample_freq=1, **params),
                            scale=False, drop=DROP)


class WeightedFit(BaseEstimator, ClassifierMixin):
    """Envuelve un pipeline y le pasa sample_weight = w para la cohorte foco."""

    def __init__(self, estimator, weight=2.0):
        self.estimator = estimator
        self.weight = weight

    def fit(self, X, y):
        focus = ((X.client_type == "cloud_marketplace") & (X.reservation_commitment == "no_commitment")).to_numpy()
        self.estimator_ = clone(self.estimator).fit(X, y, model__sample_weight=np.where(focus, self.weight, 1.0))
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def dedup(Xtr, ytr):
    """Una fila por grupo de duplicados; etiqueta = mayoría del grupo."""
    g = duplicate_groups(Xtr)
    d = pd.DataFrame({"g": g, "y": ytr, "i": np.arange(len(ytr))})
    agg = d.groupby("g").agg(i=("i", "first"), y=("y", "mean"))
    return Xtr.iloc[agg.i.to_numpy()], (agg.y.to_numpy() >= 0.5).astype(int)


def drop_anomalies(Xtr, ytr):
    keep = ~((Xtr.total_processes.fillna(0) == 0) | (Xtr.total_gpu_hours == 0)).to_numpy()
    return Xtr[keep], ytr[keep]


focus = ((train.client_type == "cloud_marketplace") & (train.reservation_commitment == "no_commitment")).to_numpy()
rows, oofs, fold_auc, fold_f1 = [], {}, {}, {}


def run(name, est, train_transform=None):
    folds, oof = cross_validate(est, X, y, groups, cv=cv, n_jobs=5, train_transform=train_transform)
    r = summarize(name, folds)
    r["foco_roc_auc"] = roc_auc_score(y[focus], oof[focus])
    r["foco_f1"] = f1_score(y[focus], oof[focus] >= 0.5)
    rows.append(r)
    oofs[name] = oof
    fold_auc[name] = folds.val_roc_auc.to_numpy()
    fold_f1[name] = folds.val_f1.to_numpy()
    print(f"{name:50s} AUC={r['val_roc_auc']:.4f} F1={r['val_f1']:.4f} rec={r['val_recall']:.4f}", flush=True)
    return oof


REF = "LightGBM tuneado (referencia)"
oof_ref = run(REF, lgbm_pipe())
run("a. class_weight = balanced", lgbm_pipe(class_weight="balanced"))
run("c. train deduplicado", lgbm_pipe(), train_transform=dedup)
run("d. train sin filas anómalas", lgbm_pipe(), train_transform=drop_anomalies)
run("e. peso ×2 a la cohorte foco", WeightedFit(lgbm_pipe(), weight=2.0))
run("e'. peso ×3 a la cohorte foco", WeightedFit(lgbm_pipe(), weight=3.0))


# ------------------------------------------------ b. umbral óptimo para F1 (sin mirar el fold evaluado)
def best_threshold(yt, p):
    ths = np.linspace(0.2, 0.7, 101)
    return ths[np.argmax([f1_score(yt, p >= t) for t in ths])]


def threshold_eval(name, oof):
    """Para cada fold k, el umbral se elige con las predicciones OOF de los otros 4 folds
    y se aplica al fold k -> estimación honesta del F1 con umbral ajustado."""
    f1s, recs, precs, ths = [], [], [], []
    for _, va in splits:
        mask = np.ones(len(y), bool)
        mask[va] = False
        t = best_threshold(y[mask], oof[mask])
        ths.append(t)
        f1s.append(f1_score(y[va], oof[va] >= t))
        recs.append(recall_score(y[va], oof[va] >= t))
        precs.append(precision_score(y[va], oof[va] >= t))
    return np.array(f1s), np.array(recs), np.array(precs), np.array(ths)


f1s, recs, precs, ths = threshold_eval(REF, oof_ref)
thr_global = best_threshold(y, oof_ref)
base_row = [r for r in rows if r["model"] == REF][0]
rows.append({**base_row, "model": f"b. umbral ajustado para F1 (≈{ths.mean():.2f})",
             "val_f1": f1s.mean(), "val_f1_std": f1s.std(ddof=1), "val_recall": recs.mean(),
             "val_recall_std": recs.std(ddof=1), "val_precision": precs.mean(), "val_precision_std": precs.std(ddof=1),
             "val_accuracy": np.nan, "val_accuracy_std": np.nan,
             "foco_f1": f1_score(y[focus], oof_ref[focus] >= thr_global)})
fold_auc[rows[-1]["model"]] = fold_auc[REF]
fold_f1[rows[-1]["model"]] = f1s
print("umbrales por fold:", ths, "F1:", f1s.mean())

res = pd.DataFrame(rows)
res["delta_auc"] = [np.mean(fold_auc[n] - fold_auc[REF]) for n in res.model]
res["delta_f1"] = [np.mean(fold_f1[n] - fold_f1[REF]) for n in res.model]
res["folds_mejora_f1"] = [int((fold_f1[n] > fold_f1[REF]).sum()) for n in res.model]
save_table(res, "04e_data_centered_raw")
t = fmt_table(res, cols=["roc_auc", "f1", "precision", "recall"])
t["Δ AUC"] = res.delta_auc.map("{:+.4f}".format)
t["Δ F1"] = res.delta_f1.map("{:+.4f}".format)
t["folds mejora F1"] = res.folds_mejora_f1.map(lambda v: f"{v}/5")
t["AUC foco"] = res.foco_roc_auc.map("{:.4f}".format)
t["F1 foco"] = res.foco_f1.map("{:.4f}".format)
save_table(t, "04e_data_centered")
print(t.to_string(index=False))
json.dump({"threshold_f1_oof": float(thr_global), "threshold_per_fold": ths.tolist()},
          open(MODEL_DIR / "threshold.json", "w"), indent=2)

# curva F1 / precision / recall vs umbral
grid = np.linspace(0.05, 0.95, 91)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(grid, [f1_score(y, oof_ref >= g) for g in grid], color=P.ORANGE, label="F1")
ax.plot(grid, [precision_score(y, oof_ref >= g) for g in grid], color=P.BLUE, label="precision")
ax.plot(grid, [recall_score(y, oof_ref >= g) for g in grid], color=P.AQUA, label="recall")
ax.axvline(0.5, color=P.TEXT2, ls=":", lw=1)
ax.axvline(thr_global, color=P.ORANGE, ls="--", lw=1)
ax.text(thr_global, 0.42, f" umbral F1 = {thr_global:.2f}", color=P.TEXT2, fontsize=8)
ax.set_xlabel("umbral de decisión")
ax.set_ylim(0.4, 1.0)
ax.legend(loc="lower left")
ax.set_title("LightGBM tuneado: métricas según el umbral (OOF)")
P.save(fig, "04e_umbral")

# efecto por cohorte del umbral ajustado (¿global o específico?)
coh_rows = []
for col in ["client_type", "service_tier", "reserved_gpu_type"]:
    a = cohort_report(train, y, oof_ref, col, threshold=0.5)[[col, "n", "f1", "FNR", "FPR"]]
    b = cohort_report(train, y, oof_ref, col, threshold=thr_global)[[col, "f1", "FNR", "FPR"]]
    m = a.merge(b, on=col, suffixes=("_0.5", "_ajustado"))
    m.insert(0, "variable", col)
    coh_rows.append(m.rename(columns={col: "valor"}))
save_table(pd.concat(coh_rows), "04e_umbral_por_cohorte")
