"""Paso 2 – Model-centered ML: baseline, validación cruzada y métricas.

* 2.1 Calibración breve de hiperparámetros de 3 clasificadores vistos en clase
      (Regresión Logística, k-NN, Árbol de Decisión) con GridSearchCV.
* 2.2 k-fold CV (k=5) estratificado y agrupado por duplicados, sólo sobre train.
* 2.3 Métricas: ROC-AUC (criterio de refit) y F1 de la clase "cancelada".
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from tp3 import plotting as P
from tp3.config import MODEL_DIR, RANDOM_STATE
from tp3.data import duplicate_groups, get_data, xy
from tp3.evaluation import cross_validate, fmt_table, get_cv, save_table, summarize
from tp3.preprocessing import make_pipeline

train, _ = get_data()          # el test NO se toca en este paso
X, y = xy(train)
groups = duplicate_groups(train)
cv = get_cv()
SCORING = {"roc_auc": "roc_auc", "f1": "f1"}

candidates = {
    "Regresión Logística": (
        make_pipeline(LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]},
        "model__C", True),
    "k-NN": (
        make_pipeline(KNeighborsClassifier(n_jobs=1)),
        {"model__n_neighbors": [5, 15, 31, 51, 101], "model__weights": ["uniform", "distance"]},
        "model__n_neighbors", False),
    "Árbol de Decisión": (
        make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE), scale=False),
        {"model__max_depth": [3, 5, 8, 12, 16, 20, None], "model__min_samples_leaf": [1, 5, 20, 50]},
        "model__max_depth", False),
}

search_rows, best = [], {}
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
for ax, (name, (pipe, grid, xparam, logx)) in zip(axes, candidates.items()):
    gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=4,
                      return_train_score=True)
    gs.fit(X, y, groups=groups)
    res = pd.DataFrame(gs.cv_results_)
    for _, r in res.iterrows():
        search_rows.append({"modelo": name, "params": json.dumps({k.replace("model__", ""): v for k, v in r["params"].items()}),
                            "val_roc_auc": r["mean_test_roc_auc"], "val_roc_auc_std": r["std_test_roc_auc"],
                            "val_f1": r["mean_test_f1"], "train_roc_auc": r["mean_train_roc_auc"]})
    best[name] = gs.best_params_
    print(name, gs.best_params_, f"AUC={gs.best_score_:.4f}")

    # curva de validación sobre el hiperparámetro principal (resto fijo en el mejor valor)
    others = {k: v for k, v in gs.best_params_.items() if k != xparam}
    mask = np.ones(len(res), bool)
    for k, v in others.items():
        mask &= res[f"param_{k}"].astype(str) == str(v)
    sub = res[mask].copy()
    xs = sub[f"param_{xparam}"].apply(lambda v: 40 if v is None else v).astype(float)
    order = np.argsort(xs.values)
    xs = xs.values[order]
    for metric, color, label in [("roc_auc", P.BLUE, "ROC-AUC"), ("f1", P.ORANGE, "F1")]:
        ax.plot(xs, sub[f"mean_test_{metric}"].values[order], marker="o", ms=5, color=color, label=f"{label} validación")
        ax.plot(xs, sub[f"mean_train_{metric}"].values[order], ls="--", lw=1.5, color=color, label=f"{label} train")
    if logx:
        ax.set_xscale("log")
    if xparam == "model__max_depth":
        ax.set_xticks(xs)
        ax.set_xticklabels(["None" if v == 40 else int(v) for v in xs])
    ax.set_xlabel(xparam.replace("model__", ""))
    ax.set_title(f"{name}\n({', '.join(f'{k.replace(chr(109)+"odel__", "")}={v}' for k, v in others.items()) or 'sin otros params'})", fontsize=10)
    ax.set_ylim(0.6, 1.0)
axes[0].set_ylabel("score (5-fold CV)")
axes[-1].legend(loc="lower right", fontsize=7.5)
fig.suptitle("Calibración de hiperparámetros del baseline (curvas de validación)", fontweight="bold")
P.save(fig, "02_curvas_validacion_baseline")
save_table(pd.DataFrame(search_rows).sort_values(["modelo", "val_roc_auc"], ascending=[True, False]),
           "02_grid_search_baseline")

# ---------------------------------------------------- CV final de cada candidato (mismos folds)
final_models = {
    "Dummy (clase mayoritaria)": make_pipeline(DummyClassifier(strategy="prior")),
    **{name: candidates[name][0].set_params(**best[name]) for name in candidates},
}
rows, oofs = [], {}
for name, est in final_models.items():
    folds, oof = cross_validate(est, X, y, groups, cv=cv, n_jobs=5)
    rows.append(summarize(name, folds))
    oofs[name] = oof
comp = pd.DataFrame(rows)
save_table(comp, "02_comparacion_baseline_raw")
save_table(fmt_table(comp), "02_comparacion_baseline")
print(fmt_table(comp).to_string(index=False))

# Elección del baseline: mejor ROC-AUC (y se verifica que también tenga el mejor F1)
cands = comp[~comp.model.str.startswith("Dummy")]
baseline_name = cands.sort_values("val_roc_auc", ascending=False).iloc[0]["model"]
assert baseline_name == cands.sort_values("val_f1", ascending=False).iloc[0]["model"], \
    "ROC-AUC y F1 no coinciden en el mejor modelo -> revisar la decisión"
MODEL_DIR.mkdir(exist_ok=True)
(MODEL_DIR / "baseline.json").write_text(json.dumps(
    {"name": baseline_name, "params": {k: v for k, v in best[baseline_name].items()}}, indent=2, default=str))
print("BASELINE:", baseline_name, best[baseline_name])

# barras de comparación
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
for ax, m, color in zip(axes, ["roc_auc", "f1"], [P.BLUE, P.ORANGE]):
    ax.barh(comp.model, comp[f"val_{m}"], xerr=comp[f"val_{m}_std"], color=color, height=0.55,
            error_kw=dict(ecolor=P.TEXT2, lw=1, capsize=3))
    for yy, v in enumerate(comp[f"val_{m}"]):
        ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=8, color=P.TEXT2)
    ax.set_xlim(0, 1.08)
    ax.set_title({"roc_auc": "ROC-AUC", "f1": "F1 (clase cancelada)"}[m])
    ax.grid(axis="y", visible=False)
axes[0].invert_yaxis()
fig.suptitle("Comparación de clasificadores baseline (5-fold CV agrupado, media ± std)", fontweight="bold")
P.save(fig, "02_comparacion_baseline")
