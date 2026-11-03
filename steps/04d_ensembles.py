"""Paso 4 – Estrategia 4: Ensemble learning.

Con los modelos tuneados en 4.3 se prueban:
* bagging de semillas de LightGBM (promedio de 5 semillas),
* soft voting (LightGBM + XGBoost + RF) y (… + Regresión Logística),
* stacking con meta-modelo logístico y CV interno agrupado por duplicados.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

from tp3 import plotting as P
from tp3.config import MODEL_DIR, RANDOM_STATE
from tp3.data import duplicate_groups, get_data, xy
from tp3.ensembles import GroupStackingClassifier, SeedBaggingClassifier
from tp3.evaluation import cross_validate, fmt_table, get_cv, save_table, summarize
from tp3.features import make_fe_pipeline

train, _ = get_data()
X, y = xy(train)
groups = duplicate_groups(train)
cv = get_cv()
DROP = json.loads((MODEL_DIR / "feature_selection.json").read_text())["drop"]
HPO = json.loads((MODEL_DIR / "hpo_params.json").read_text())
RS = RANDOM_STATE


def lgbm_pipe():
    return make_fe_pipeline(LGBMClassifier(verbose=-1, random_state=RS, n_jobs=1, subsample_freq=1, **HPO["lightgbm"]),
                            scale=False, drop=DROP)


def xgb_pipe():
    return make_fe_pipeline(XGBClassifier(tree_method="hist", random_state=RS, n_jobs=1, eval_metric="logloss",
                                          **HPO["xgboost"]), scale=False, drop=DROP)


def rf_pipe():
    return make_fe_pipeline(RandomForestClassifier(n_jobs=1, random_state=RS, **HPO["random_forest"]),
                            scale=False, drop=DROP)


def lr_pipe():  # modelo lineal con TODAS las features de FE (es donde rinde mejor, ver 4.1)
    return make_fe_pipeline(LogisticRegression(C=0.1, max_iter=5000), high_card="target")


members = {"LightGBM (tuneado)": lgbm_pipe(), "XGBoost (tuneado)": xgb_pipe(),
           "Random Forest (tuneado)": rf_pipe(), "Regresión Logística + FE": lr_pipe()}

focus = ((train.client_type == "cloud_marketplace") & (train.reservation_commitment == "no_commitment")).to_numpy()
rows, oofs, fold_auc, fold_f1 = [], {}, {}, {}


def run(name, est, n_jobs=5):
    folds, oof = cross_validate(est, X, y, groups, cv=cv, n_jobs=n_jobs)
    r = summarize(name, folds)
    r["foco_roc_auc"] = roc_auc_score(y[focus], oof[focus])
    r["foco_f1"] = f1_score(y[focus], oof[focus] >= 0.5)
    rows.append(r)
    oofs[name] = oof
    fold_auc[name] = folds.val_roc_auc.to_numpy()
    fold_f1[name] = folds.val_f1.to_numpy()
    print(f"{name:45s} AUC={r['val_roc_auc']:.4f} F1={r['val_f1']:.4f}", flush=True)


for name, est in members.items():
    run(name, est)

run("Bagging de semillas LightGBM (×5)",
    SeedBaggingClassifier(lgbm_pipe(), seeds=(0, 1, 2, 3, 4), seed_param="model__random_state"))
run("Soft voting (LGBM + XGB + RF)",
    VotingClassifier([("lgbm", lgbm_pipe()), ("xgb", xgb_pipe()), ("rf", rf_pipe())], voting="soft"))
run("Soft voting (LGBM + XGB + RF + LR)",
    VotingClassifier([("lgbm", lgbm_pipe()), ("xgb", xgb_pipe()), ("rf", rf_pipe()), ("lr", lr_pipe())],
                     voting="soft"))
run("Stacking (LGBM + XGB + RF + LR → LogReg)",
    GroupStackingClassifier([("lgbm", lgbm_pipe()), ("xgb", xgb_pipe()), ("rf", rf_pipe()), ("lr", lr_pipe())],
                            final_estimator=LogisticRegression(C=1.0)))

res = pd.DataFrame(rows)
ref = "LightGBM (tuneado)"
res["delta_auc"] = [np.mean(fold_auc[n] - fold_auc[ref]) for n in res.model]
res["delta_f1"] = [np.mean(fold_f1[n] - fold_f1[ref]) for n in res.model]
res["folds_mejora_auc"] = [int((fold_auc[n] > fold_auc[ref]).sum()) for n in res.model]
save_table(res, "04d_ensembles_raw")
t = fmt_table(res)
t["Δ AUC vs LGBM"] = res.delta_auc.map("{:+.4f}".format)
t["Δ F1 vs LGBM"] = res.delta_f1.map("{:+.4f}".format)
t["folds mejora"] = res.folds_mejora_auc.map(lambda v: f"{v}/5")
t["AUC foco"] = res.foco_roc_auc.map("{:.4f}".format)
t["F1 foco"] = res.foco_f1.map("{:.4f}".format)
save_table(t, "04d_ensembles")
print(t.to_string(index=False))
np.savez_compressed(MODEL_DIR / "oof_ensembles.npz", **oofs)

# diversidad: correlación de las predicciones OOF de los miembros
names = list(members)
corr = pd.DataFrame(np.corrcoef([oofs[n] for n in names]), index=names, columns=names)
err_corr = pd.DataFrame(np.corrcoef([(oofs[n] >= 0.5) != y for n in names]), index=names, columns=names)
save_table(corr.round(4).reset_index(), "04d_correlacion_predicciones")
save_table(err_corr.round(4).reset_index(), "04d_correlacion_errores")
fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(err_corr, cmap=P.SEQ_CMAP, vmin=0, vmax=1)
short = ["LGBM", "XGB", "RF", "LR+FE"]
ax.set_xticks(range(4), short)
ax.set_yticks(range(4), short)
ax.grid(False)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{err_corr.iloc[i, j]:.2f}", ha="center", va="center",
                color="white" if err_corr.iloc[i, j] > 0.6 else P.TEXT, fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Correlación de errores entre miembros (OOF)")
P.save(fig, "04d_correlacion_errores")

fig, ax = plt.subplots(figsize=(8, 3.8))
order = res.sort_values("val_roc_auc")
ax.barh(order.model, order.val_roc_auc, xerr=order.val_roc_auc_std, height=0.6,
        color=[P.ORANGE if n not in members else P.BLUE for n in order.model],
        error_kw=dict(ecolor=P.TEXT2, lw=1, capsize=2))
for yy, (v, sd) in enumerate(zip(order.val_roc_auc, order.val_roc_auc_std)):
    ax.text(v + sd + 0.0005, yy, f"{v:.4f}", va="center", fontsize=7.5, color=P.TEXT2)
ax.set_xlim(0.90, 0.95)
ax.set_xlabel("ROC-AUC (5-fold CV)")
ax.grid(axis="y", visible=False)
ax.set_title("Modelos individuales (azul) vs ensambles (naranja)")
P.save(fig, "04d_ensembles")
