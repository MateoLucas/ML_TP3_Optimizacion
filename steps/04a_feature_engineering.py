"""Paso 4 – Estrategia 1: Feature engineering y selección de variables.

1. Se agregan features nuevas (tp3.features) + target encoding de las categóricas
   de alta cardinalidad, y se mide su efecto con el MISMO modelo baseline
   (Árbol depth=8), con Regresión Logística y con un LightGBM de referencia.
2. Selección de variables por permutation importance (calculada fuera de la
   muestra en cada fold) y re-evaluación con subconjuntos de features.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from tp3 import plotting as P
from tp3.config import MODEL_DIR, RANDOM_STATE
from tp3.data import duplicate_groups, get_data, xy
from tp3.evaluation import cross_validate, fmt_table, get_cv, save_table, summarize
from tp3.features import ENGINEERED_NUMERIC, GROUP_NUMERIC, make_fe_pipeline
from tp3.preprocessing import make_pipeline

train, _ = get_data()
X, y = xy(train)
groups = duplicate_groups(train)
cv = get_cv()


def lgbm():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1,
                          random_state=RANDOM_STATE, n_jobs=1)


tree = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
logreg = LogisticRegression(C=0.1, max_iter=5000)

experiments = {
    "Árbol d=8 | baseline (one-hot)": make_pipeline(tree, scale=False),
    "Árbol d=8 | + FE (one-hot IDs)": make_fe_pipeline(tree, scale=False, high_card="onehot"),
    "Árbol d=8 | + FE + target enc.": make_fe_pipeline(tree, scale=False, high_card="target"),
    "LogReg | baseline (one-hot)": make_pipeline(logreg),
    "LogReg | + FE + target enc.": make_fe_pipeline(logreg, high_card="target"),
    "LightGBM ref | baseline (one-hot)": make_pipeline(lgbm(), scale=False),
    "LightGBM ref | + FE + target enc.": make_fe_pipeline(lgbm(), scale=False, high_card="target"),
}
rows, oofs = [], {}
for name, est in experiments.items():
    folds, oof = cross_validate(est, X, y, groups, cv=cv, n_jobs=5)
    rows.append(summarize(name, folds))
    oofs[name] = oof
    print(name, f"AUC={rows[-1]['val_roc_auc']:.4f} F1={rows[-1]['val_f1']:.4f}")
fe = pd.DataFrame(rows)
# métricas dentro de la cohorte foco del Paso 3 (cloud_marketplace & no_commitment)
focus = ((train.client_type == "cloud_marketplace") & (train.reservation_commitment == "no_commitment")).to_numpy()
fe["foco_roc_auc"] = [roc_auc_score(y[focus], oofs[n][focus]) for n in fe.model]
fe["foco_f1"] = [f1_score(y[focus], oofs[n][focus] >= 0.5) for n in fe.model]
save_table(fe, "04a_feature_engineering_raw")
t = fmt_table(fe)
t["AUC cohorte foco"] = fe.foco_roc_auc.map("{:.4f}".format)
t["F1 cohorte foco"] = fe.foco_f1.map("{:.4f}".format)
save_table(t, "04a_feature_engineering")
print(t.to_string(index=False))

# ------------------------------------------------------------- permutation importance (OOF, por fold)
FE_COLS = None


def fold_importance(tr, va, k):
    pipe = make_fe_pipeline(lgbm(), scale=False, high_card="target")
    pipe.fit(X.iloc[tr], y[tr])
    Z = pipe[:2].transform(X.iloc[va])           # DataFrame después de FE + GroupStats
    tail = pipe[2:]                               # preprocesador + modelo
    used = list(tail.named_steps["prep"].feature_names_in_)
    base = roc_auc_score(y[va], tail.predict_proba(Z)[:, 1])
    rng = np.random.default_rng(RANDOM_STATE + k)
    out = []
    for c in used:
        drops = []
        for _ in range(3):
            Zp = Z.copy()
            Zp[c] = rng.permutation(Zp[c].values)
            drops.append(base - roc_auc_score(y[va], tail.predict_proba(Zp)[:, 1]))
        out.append({"feature": c, "fold": k, "auc_drop": float(np.mean(drops))})
    return out


res = Parallel(n_jobs=5)(delayed(fold_importance)(tr, va, k) for k, (tr, va) in enumerate(cv.split(X, y, groups)))
imp = pd.DataFrame([r for fold in res for r in fold])
imp = imp.groupby("feature").auc_drop.agg(["mean", "std"]).sort_values("mean", ascending=False).reset_index()
new_feats = set(ENGINEERED_NUMERIC + GROUP_NUMERIC)
imp["tipo"] = np.where(imp.feature.isin(new_feats), "nueva (FE)", "original")
save_table(imp, "04a_permutation_importance")
print(imp.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 10))
d = imp.iloc[::-1]
colors = [P.ORANGE if t == "nueva (FE)" else P.BLUE for t in d.tipo]
ax.barh(d.feature, d["mean"], xerr=d["std"], color=colors, height=0.65,
        error_kw=dict(ecolor=P.TEXT2, lw=0.8, capsize=2))
ax.axvline(0, color=P.TEXT2, lw=0.8)
ax.set_xlabel("caída de ROC-AUC al permutar la variable (media ± std en 5 folds)")
ax.set_title("Permutation importance (LightGBM ref + FE, validación)")
ax.grid(axis="y", visible=False)
ax.tick_params(axis="y", labelsize=7.5)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=P.BLUE, label="original"), Patch(color=P.ORANGE, label="nueva (FE)")],
          loc="lower right")
P.save(fig, "04a_permutation_importance")

# ------------------------------------------------------------- selección de variables
negligible = imp.loc[imp["mean"] <= 0.0002, "feature"].tolist()
ranked = imp.feature.tolist()
selections = {
    "todas las features": [],
    f"sin las {len(negligible)} de importancia ≤ 0.0002": negligible,
    "top 20": ranked[20:],
    "top 12": ranked[12:],
}
sel_rows, sel_folds = [], {}
for name, drop in selections.items():
    folds, _ = cross_validate(make_fe_pipeline(lgbm(), scale=False, drop=drop), X, y, groups, cv=cv, n_jobs=5)
    sel_folds[name] = folds
    r = summarize(name, folds)
    r["n_features_entrada"] = len(ranked) - len(drop)
    sel_rows.append(r)
sel = pd.DataFrame(sel_rows)
# comparación PAREADA contra "todas las features" (mismos folds): diferencia media de AUC
ref = sel_folds["todas las features"].val_roc_auc.to_numpy()
sel["delta_auc_pareado"] = [np.mean(sel_folds[n].val_roc_auc.to_numpy() - ref) for n in sel.model]
sel["delta_auc_pareado_std"] = [np.std(sel_folds[n].val_roc_auc.to_numpy() - ref, ddof=1) for n in sel.model]
save_table(sel, "04a_seleccion_variables_raw")
t = fmt_table(sel)
t.insert(1, "n features", sel["n_features_entrada"])
t["Δ AUC pareado vs todas"] = [f"{a:+.4f} ± {b:.4f}" for a, b in zip(sel.delta_auc_pareado, sel.delta_auc_pareado_std)]
save_table(t, "04a_seleccion_variables")
print(t.to_string(index=False))

# Regla: el subconjunto más chico que no pierda más de 0.001 de AUC (pareado) contra usar todas
ok = sel[sel.delta_auc_pareado >= -0.001].sort_values("n_features_entrada")
chosen = ok.iloc[0]
chosen_drop = selections[chosen.model]
(MODEL_DIR / "feature_selection.json").write_text(json.dumps(
    {"selection": chosen.model, "drop": chosen_drop}, indent=2, ensure_ascii=False))
print("SELECCIÓN:", chosen.model, "drop:", chosen_drop)

# figura resumen FE
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
for ax, m, color in zip(axes, ["roc_auc", "f1"], [P.BLUE, P.ORANGE]):
    ax.barh(fe.model, fe[f"val_{m}"], xerr=fe[f"val_{m}_std"], color=color, height=0.55,
            error_kw=dict(ecolor=P.TEXT2, lw=1, capsize=3))
    for yy, (v, sd) in enumerate(zip(fe[f"val_{m}"], fe[f"val_{m}_std"])):
        ax.text(v + sd + 0.004, yy, f"{v:.3f}", va="center", fontsize=8, color=P.TEXT2)
    ax.set_xlim(0.7 if m == "roc_auc" else 0.6, 1.0)
    ax.set_title({"roc_auc": "ROC-AUC", "f1": "F1"}[m])
    ax.grid(axis="y", visible=False)
axes[0].invert_yaxis()
fig.suptitle("Efecto del feature engineering (mismo modelo, mismos folds)", fontweight="bold")
P.save(fig, "04a_feature_engineering")
