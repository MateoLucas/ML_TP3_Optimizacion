"""Paso 4 – Estrategia 2: Model selection.

Compara familias de clasificadores con configuración estándar (sin tunear),
todas sobre el mismo pipeline (FE + target encoding + variables seleccionadas)
y los mismos folds.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from tp3 import plotting as P
from tp3.config import MODEL_DIR, RANDOM_STATE
from tp3.data import duplicate_groups, get_data, xy
from tp3.evaluation import cross_validate, fmt_table, get_cv, save_table, summarize
from tp3.features import make_fe_pipeline
from tp3.preprocessing import make_pipeline

train, _ = get_data()
X, y = xy(train)
groups = duplicate_groups(train)
cv = get_cv()
DROP = json.loads((MODEL_DIR / "feature_selection.json").read_text())["drop"]
RS = RANDOM_STATE

models = {
    "Naive Bayes (Gaussiano)": GaussianNB(),
    "Regresión Logística": LogisticRegression(C=0.1, max_iter=5000),
    "k-NN (k=31)": KNeighborsClassifier(n_neighbors=31, weights="distance"),
    "Árbol de Decisión (d=8)": DecisionTreeClassifier(max_depth=8, random_state=RS),
    "SVM (RBF)": SVC(C=1.0, gamma="scale", random_state=RS),
    "Red neuronal (MLP 64-32)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                              max_iter=300, random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=RS),
    "Extra Trees": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=1, random_state=RS),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=RS),
    "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1,
                               random_state=RS, n_jobs=1),
    "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, tree_method="hist",
                             random_state=RS, n_jobs=1, eval_metric="logloss"),
}

focus = ((train.client_type == "cloud_marketplace") & (train.reservation_commitment == "no_commitment")).to_numpy()
rows, oofs = [], {}
# referencia: baseline exacto del Paso 2 (sin FE)
folds, oof = cross_validate(make_pipeline(DecisionTreeClassifier(max_depth=8, random_state=RS), scale=False),
                            X, y, groups, cv=cv, n_jobs=5)
rows.append(summarize("BASELINE (Árbol d=8, sin FE)", folds))
oofs["BASELINE (Árbol d=8, sin FE)"] = oof
for name, model in models.items():
    folds, oof = cross_validate(make_fe_pipeline(model, drop=DROP), X, y, groups, cv=cv, n_jobs=5)
    rows.append(summarize(name, folds))
    oofs[name] = oof
    print(f"{name:28s} AUC={rows[-1]['val_roc_auc']:.4f} F1={rows[-1]['val_f1']:.4f} "
          f"fit={rows[-1]['fit_time_s']:.1f}s", flush=True)

ms = pd.DataFrame(rows)
ms["foco_roc_auc"] = [roc_auc_score(y[focus], oofs[n][focus]) for n in ms.model]
ms["foco_f1"] = [f1_score(y[focus], oofs[n][focus] >= 0.5) for n in ms.model]
ms = ms.sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
save_table(ms, "04b_model_selection_raw")
t = fmt_table(ms)
t["AUC cohorte foco"] = ms.foco_roc_auc.map("{:.4f}".format)
t["F1 cohorte foco"] = ms.foco_f1.map("{:.4f}".format)
save_table(t, "04b_model_selection")
print(t.to_string(index=False))
np.savez_compressed(MODEL_DIR / "oof_model_selection.npz", **{k: v for k, v in oofs.items()})

# ------------------------------------------------------------------ figuras
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 0.7]))
for ax, m, color, lo in zip(axes[:2], ["roc_auc", "f1"], [P.BLUE, P.ORANGE], [0.75, 0.6]):
    cols = [P.TEXT2 if n.startswith("BASELINE") else color for n in ms.model]
    ax.barh(ms.model, ms[f"val_{m}"], xerr=ms[f"val_{m}_std"], color=cols, height=0.6,
            error_kw=dict(ecolor=P.TEXT2, lw=1, capsize=2))
    for yy, (v, sd) in enumerate(zip(ms[f"val_{m}"], ms[f"val_{m}_std"])):
        ax.text(v + sd + 0.003, yy, f"{v:.3f}", va="center", fontsize=7.5, color=P.TEXT2)
    ax.set_xlim(lo, 1.0)
    ax.set_title({"roc_auc": "ROC-AUC", "f1": "F1"}[m])
    ax.grid(axis="y", visible=False)
ax = axes[2]
ax.barh(ms.model, ms.fit_time_s, color=P.AQUA, height=0.6)
ax.set_xscale("log")
ax.set_title("fit por fold (s, log)")
ax.grid(axis="y", visible=False)
axes[0].invert_yaxis()
fig.suptitle("Model selection: familias de clasificadores (FE + selección, 5-fold CV agrupado)",
             fontweight="bold")
P.save(fig, "04b_model_selection")

fig, ax = plt.subplots(figsize=(5.4, 5))
show = ["BASELINE (Árbol d=8, sin FE)", "Regresión Logística", "Random Forest", "LightGBM", "XGBoost"]
for i, n in enumerate(show):
    fpr, tpr, _ = roc_curve(y, oofs[n])
    ax.plot(fpr, tpr, lw=2, color=P.SERIES[i], label=f"{n} (AUC={roc_auc_score(y, oofs[n]):.3f})")
ax.plot([0, 1], [0, 1], color=P.GRID, lw=1, ls="--")
ax.set_xlabel("FPR")
ax.set_ylabel("TPR (recall)")
ax.set_title("Curvas ROC (predicciones OOF)")
ax.legend(fontsize=7.5, loc="lower right")
P.save(fig, "04b_curvas_roc")
