"""Paso 3 – Error analysis y enfoque data-centered.

* 3.1 Análisis de errores del baseline con predicciones out-of-fold (OOF) en train:
      cohortes (client_type, service_tier, reserved_gpu_type y otras), FP vs FN,
      y búsqueda de regiones problemáticas (slices).
* 3.2 Curvas de aprendizaje (baseline vs. modelos de distinta capacidad).
* 3.3 / 3.4 Evidencia cuantitativa para la estrategia data-centered y la adquisición de datos.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier, export_text

from tp3 import plotting as P
from tp3.config import MODEL_DIR, NUMERIC_COLS, RANDOM_STATE, TARGET
from tp3.data import duplicate_groups, get_data, xy
from tp3.evaluation import cohort_report, cross_validate, get_cv, save_table, score_proba
from tp3.preprocessing import make_pipeline

train, _ = get_data()
X, y = xy(train)
groups = duplicate_groups(train)
cv = get_cv()

base_cfg = json.loads((MODEL_DIR / "baseline.json").read_text())
baseline = make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE), scale=False).set_params(**base_cfg["params"])
folds, oof = cross_validate(baseline, X, y, groups, cv=cv, n_jobs=5)
pred = (oof >= 0.5).astype(int)
np.save(MODEL_DIR / "oof_baseline.npy", oof)
print("OOF global:", {k: round(v, 4) for k, v in score_proba(y, oof).items()})

# ------------------------------------------------------------------ columnas auxiliares de análisis
dup_count = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy()
A = train.assign(
    y=y, proba=oof, pred=pred,
    outcome=np.select([(pred == 1) & (y == 1), (pred == 0) & (y == 0), (pred == 1) & (y == 0)],
                      ["TP", "TN", "FP"], "FN"),
    lead_time_bin=pd.cut(train.hours_before_execution, [-1, 7, 30, 90, 180, 365, 1e9],
                         labels=["0–7", "8–30", "31–90", "91–180", "181–365", ">365"]),
    region=train.client_region.where(train.client_region.isin(["PRT", "FRA", "DEU", "GBR", "ESP", "ITA"]), "otras"),
    has_history=np.where((train.previous_job_cancellations + train.successful_previous_jobs) > 0, "con historial", "sin historial"),
    special_req=np.where(train.special_resource_requests > 0, "≥1 pedido especial", "sin pedidos"),
    gpu_changed=np.where(train.reserved_gpu_type != train.assigned_gpu_type, "GPU cambiada", "misma GPU"),
    in_dup_group=np.where(dup_count > 1, "duplicada (lote)", "única"),
    has_agent=np.where(train.orchestration_agent_id == "none", "sin agente", "con agente"),
)
A["error"] = (A.y != A.pred).astype(int)

# ------------------------------------------------------------------ matriz de confusión
cm = confusion_matrix(y, pred)
fig, ax = plt.subplots(figsize=(3.8, 3.4))
ax.imshow(cm, cmap=P.SEQ_CMAP)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j]/cm.sum():.1%})", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else P.TEXT, fontsize=10)
ax.set_xticks([0, 1], ["pred. no cancela", "pred. cancela"])
ax.set_yticks([0, 1], ["real no cancela", "real cancela"])
ax.grid(False)
ax.set_title("Matriz de confusión OOF (baseline)")
P.save(fig, "03_matriz_confusion")

# distribución de probabilidades OOF por clase
fig, ax = plt.subplots(figsize=(6.5, 3.2))
bins = np.linspace(0, 1, 41)
for k in [0, 1]:
    ax.hist(oof[y == k], bins=bins, alpha=0.6, color=P.CLASS_COLORS[k],
            label=["no cancelada (real)", "cancelada (real)"][k])
ax.axvline(0.5, color=P.TEXT2, ls="--", lw=1)
ax.set_xlabel("probabilidad predicha de cancelación (OOF)")
ax.set_ylabel("reservas")
ax.legend()
ax.set_title("Separación de clases del baseline")
P.save(fig, "03_distribucion_probabilidades")

# ------------------------------------------------------------------ cohortes
cohort_vars = ["client_type", "service_tier", "reserved_gpu_type", "reservation_commitment",
               "request_channel", "workload_profile", "region", "lead_time_bin", "has_history",
               "special_req", "gpu_changed", "in_dup_group", "has_agent"]
all_cohorts = []
for col in cohort_vars:
    rep = cohort_report(A, y, oof, col)
    save_table(rep, f"03_cohorte_{col}")
    all_cohorts.append(rep.rename(columns={col: "valor"}).assign(variable=col))
coh = pd.concat(all_cohorts, ignore_index=True)
coh["valor"] = coh["valor"].astype(str)
save_table(coh[["variable", "valor", "n", "share", "cancel_rate", "pred_rate", "roc_auc", "f1",
                "accuracy", "FPR", "FNR", "FP", "FN", "errors_share"]], "03_cohortes_todas")

# figura: las tres cohortes pedidas por la consigna
fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), gridspec_kw=dict(width_ratios=[1, 1, 1]))
for r, col in enumerate(["client_type", "service_tier", "reserved_gpu_type"]):
    rep = coh[coh.variable == col].sort_values("n")
    yy = np.arange(len(rep))
    # (a) F1 y AUC
    ax = axes[r, 0]
    ax.barh(yy - 0.2, rep.roc_auc, height=0.38, color=P.BLUE, label="ROC-AUC")
    ax.barh(yy + 0.2, rep.f1, height=0.38, color=P.ORANGE, label="F1")
    ax.axvline(score_proba(y, oof)["roc_auc"], color=P.BLUE, ls="--", lw=1)
    ax.axvline(score_proba(y, oof)["f1"], color=P.ORANGE, ls="--", lw=1)
    ax.set_yticks(yy, [f"{v}  (n={n:,})" for v, n in zip(rep.valor, rep.n)])
    ax.set_xlim(0, 1)
    ax.set_title(f"{col}: ROC-AUC y F1")
    ax.grid(axis="y", visible=False)
    if r == 0:
        ax.legend(loc="lower left", fontsize=7.5)
    # (b) FPR y FNR
    ax = axes[r, 1]
    ax.barh(yy - 0.2, rep.FPR, height=0.38, color=P.AQUA, label="FPR (falsa alarma)")
    ax.barh(yy + 0.2, rep.FNR, height=0.38, color=P.VIOLET, label="FNR (cancelación no detectada)")
    ax.set_yticks(yy, [""] * len(yy))
    ax.set_xlim(0, 1)
    ax.set_title(f"{col}: tasas de error")
    ax.grid(axis="y", visible=False)
    if r == 0:
        ax.legend(loc="lower right", fontsize=7.5)
    # (c) % de datos vs % de errores
    ax = axes[r, 2]
    ax.barh(yy - 0.2, rep.share, height=0.38, color=P.TEXT2, alpha=0.45, label="% de las reservas")
    ax.barh(yy + 0.2, rep.errors_share, height=0.38, color=P.RED, label="% de los errores")
    ax.set_yticks(yy, [""] * len(yy))
    ax.set_xlim(0, 1)
    ax.set_title(f"{col}: peso en datos vs en errores")
    ax.grid(axis="y", visible=False)
    if r == 0:
        ax.legend(loc="lower right", fontsize=7.5)
fig.suptitle("Análisis de errores por cohorte (baseline, predicciones OOF, cohortes con n ≥ 30)",
             fontweight="bold")
P.save(fig, "03_cohortes_errores")

# figura: otras cohortes relevantes (F1 por subgrupo)
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.flat, ["reservation_commitment", "region", "lead_time_bin", "has_history",
                               "special_req", "in_dup_group"]):
    rep = coh[coh.variable == col]
    if col == "lead_time_bin":
        rep = rep.set_index("valor").loc[["0–7", "8–30", "31–90", "91–180", "181–365", ">365"]].reset_index()
    else:
        rep = rep.sort_values("n")
    yy = np.arange(len(rep))
    ax.barh(yy - 0.2, rep.FPR, height=0.38, color=P.AQUA, label="FPR")
    ax.barh(yy + 0.2, rep.FNR, height=0.38, color=P.VIOLET, label="FNR")
    for i, (a, b, f1) in enumerate(zip(rep.FPR, rep.FNR, rep.f1)):
        ax.text(max(a, b) + 0.02, i, f"F1={f1:.2f}", va="center", fontsize=7.5, color=P.TEXT2)
    ax.set_yticks(yy, [f"{v}  (n={n:,})" for v, n in zip(rep.valor, rep.n)], fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_title(col)
    ax.grid(axis="y", visible=False)
axes.flat[0].legend(loc="lower right", fontsize=7.5)
fig.suptitle("Tasas de error en otros subgrupos (baseline, OOF)", fontweight="bold")
P.save(fig, "03_subgrupos_extra")

# ------------------------------------------------------------------ FP y FN
num_cmp = ["hours_before_execution", "special_resource_requests", "previous_job_cancellations",
           "successful_previous_jobs", "resource_modifications", "queue_wait_time",
           "estimated_cost_per_hour", "total_gpu_hours", "total_processes"]
prof = A.groupby("outcome")[num_cmp].mean().T[["TP", "FN", "TN", "FP"]]
cat_cmp = {
    "% PRT": A.client_region.eq("PRT"),
    "% locked_allocation": A.reservation_commitment.eq("locked_allocation"),
    "% cloud_marketplace": A.client_type.eq("cloud_marketplace"),
    "% research_consortium": A.client_type.eq("research_consortium"),
    "% managed_account": A.client_type.eq("managed_account"),
    "% broker_platform": A.request_channel.eq("broker_platform"),
    "% con pedidos especiales": A.special_resource_requests.gt(0),
    "% duplicada (lote)": A.in_dup_group.eq("duplicada (lote)"),
    "% GPU cambiada": A.gpu_changed.eq("GPU cambiada"),
    "% con historial": A.has_history.eq("con historial"),
}
for k, v in cat_cmp.items():
    prof.loc[k] = v.groupby(A.outcome).mean()[["TP", "FN", "TN", "FP"]] * 100
prof.loc["n"] = A.outcome.value_counts()[["TP", "FN", "TN", "FP"]]
prof.loc["prob. media OOF"] = A.groupby("outcome").proba.mean()[["TP", "FN", "TN", "FP"]]
save_table(prof.round(3).reset_index().rename(columns={"index": "variable"}), "03_perfil_fp_fn")
print(prof.round(2).to_string())

# confianza de los errores
conf = A[A.error == 1].assign(dist=lambda d: (d.proba - 0.5).abs())
conf_tab = pd.DataFrame({
    "FN con p < 0.2 (muy confiados)": [((A.outcome == "FN") & (A.proba < 0.2)).sum()],
    "FN totales": [(A.outcome == "FN").sum()],
    "FP con p > 0.8 (muy confiados)": [((A.outcome == "FP") & (A.proba > 0.8)).sum()],
    "FP totales": [(A.outcome == "FP").sum()],
    "errores con 0.4 ≤ p ≤ 0.6 (zona dudosa)": [((A.error == 1) & A.proba.between(0.4, 0.6)).sum()],
})
save_table(conf_tab, "03_confianza_errores")
print(conf_tab.T)

# ------------------------------------------------------------------ regiones problemáticas (slices)
slice_vars = ["client_type", "service_tier", "reserved_gpu_type", "reservation_commitment",
              "request_channel", "region", "lead_time_bin", "special_req", "has_history", "in_dup_group"]
base_err = A.error.mean()
rows = []
for k in [1, 2]:
    for cols in combinations(slice_vars, k):
        for key, d in A.groupby(list(cols), observed=True):
            if len(d) < 300:
                continue
            key = key if isinstance(key, tuple) else (key,)
            rows.append({"slice": " & ".join(f"{c}={v}" for c, v in zip(cols, key)), "n": len(d),
                         "cancel_rate": d.y.mean(), "error_rate": d.error.mean(),
                         "lift_error": d.error.mean() / base_err, "errores": d.error.sum(),
                         "FP": ((d.pred == 1) & (d.y == 0)).sum(), "FN": ((d.pred == 0) & (d.y == 1)).sum(),
                         "errors_share": d.error.sum() / A.error.sum()})
slices = pd.DataFrame(rows)
top_rate = slices.sort_values("error_rate", ascending=False).drop_duplicates("slice").head(15)
top_count = slices[slices.slice.str.count("&") == 1].sort_values("errores", ascending=False).head(15)
save_table(top_rate, "03_slices_mayor_tasa_error")
save_table(top_count, "03_slices_mas_errores")
print(top_rate.head(10).to_string(index=False))
print(top_count.head(10).to_string(index=False))

# árbol "explicador" de errores: ¿qué reglas describen dónde se equivoca el modelo?
E = pd.get_dummies(A[["client_type", "reservation_commitment", "request_channel", "region",
                      "service_tier", "workload_profile"]], dtype=int)
E = pd.concat([E, A[["hours_before_execution", "special_resource_requests", "previous_job_cancellations",
                     "estimated_cost_per_hour", "total_gpu_hours", "resource_modifications"]]], axis=1)
expl = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, random_state=RANDOM_STATE).fit(E, A.error)
rules = export_text(expl, feature_names=list(E.columns), show_weights=True)
(Path(__file__).resolve().parents[1] / "reports" / "tables" / "03_arbol_errores.txt").write_text(rules)
print(rules)

# ------------------------------------------------------------------ 3.2 curvas de aprendizaje
models_lc = {
    "Árbol depth=8 (baseline)": baseline,
    "Regresión Logística (C=0.1)": make_pipeline(LogisticRegression(C=0.1, max_iter=3000)),
    "Árbol sin límite de profundidad": make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE), scale=False),
    "LightGBM (alta capacidad, default)": make_pipeline(
        LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, verbose=-1,
                       random_state=RANDOM_STATE, n_jobs=1), scale=False),
}
sizes = np.linspace(0.1, 1.0, 8)
lc_rows = []
fig, axes = plt.subplots(1, 4, figsize=(16, 3.7), sharey=True)
for ax, (name, est) in zip(axes, models_lc.items()):
    n, tr_s, va_s = learning_curve(est, X, y, groups=groups, cv=cv, train_sizes=sizes,
                                   scoring="roc_auc", n_jobs=5, shuffle=True, random_state=RANDOM_STATE)
    for i in range(len(n)):
        lc_rows.append({"modelo": name, "n_train": n[i], "train_auc": tr_s[i].mean(),
                        "val_auc": va_s[i].mean(), "val_auc_std": va_s[i].std()})
    ax.plot(n, tr_s.mean(1), ls="--", color=P.BLUE, label="train")
    ax.plot(n, va_s.mean(1), marker="o", ms=5, color=P.ORANGE, label="validación")
    ax.fill_between(n, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), color=P.ORANGE, alpha=0.15, lw=0)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("filas de entrenamiento")
    ax.set_ylim(0.75, 1.005)
    ax.text(n[-1], va_s.mean(1)[-1] - 0.012, f"{va_s.mean(1)[-1]:.3f}", ha="right", fontsize=8, color=P.TEXT2)
axes[0].set_ylabel("ROC-AUC")
axes[0].legend(loc="lower left")
fig.suptitle("Curvas de aprendizaje (5-fold CV agrupado)", fontweight="bold")
P.save(fig, "03_curvas_aprendizaje")
lc = pd.DataFrame(lc_rows)
save_table(lc, "03_curvas_aprendizaje")
print(lc.to_string(index=False))

# ------------------------------------------------------------------ 3.3 / 3.4 evidencia adicional
target_slice = (A.client_type == "cloud_marketplace") & (A.reservation_commitment == "no_commitment")
focus = pd.DataFrame([
    {"subconjunto": "cloud_marketplace & no_commitment", "n": target_slice.sum(),
     "share": target_slice.mean(), "cancel_rate": A.y[target_slice].mean(),
     "error_rate": A.error[target_slice].mean(), "errors_share": A.error[target_slice].sum() / A.error.sum(),
     **{f"oof_{k}": v for k, v in score_proba(y[target_slice], oof[target_slice]).items()}},
    {"subconjunto": "resto", "n": (~target_slice).sum(), "share": (~target_slice).mean(),
     "cancel_rate": A.y[~target_slice].mean(), "error_rate": A.error[~target_slice].mean(),
     "errors_share": A.error[~target_slice].sum() / A.error.sum(),
     **{f"oof_{k}": v for k, v in score_proba(y[~target_slice], oof[~target_slice]).items()}},
])
save_table(focus, "03_cohorte_foco")
print(focus.T)

# ¿la cohorte foco está limitada por datos o por modelo? curva de aprendizaje restringida a ella
idx = np.where(target_slice.values)[0]
Xs, ys, gs = X.iloc[idx], y[idx], groups[idx]
n, tr_s, va_s = learning_curve(models_lc["LightGBM (alta capacidad, default)"], Xs, ys, groups=gs, cv=cv,
                               train_sizes=sizes, scoring="roc_auc", n_jobs=5, shuffle=True,
                               random_state=RANDOM_STATE)
n2, tr2, va2 = learning_curve(baseline, Xs, ys, groups=gs, cv=cv, train_sizes=sizes, scoring="roc_auc",
                              n_jobs=5, shuffle=True, random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(n2, va2.mean(1), marker="o", ms=5, color=P.ORANGE, label="Árbol depth=8 – validación")
ax.plot(n2, tr2.mean(1), ls="--", color=P.ORANGE, label="Árbol depth=8 – train")
ax.plot(n, va_s.mean(1), marker="o", ms=5, color=P.BLUE, label="LightGBM – validación")
ax.plot(n, tr_s.mean(1), ls="--", color=P.BLUE, label="LightGBM – train")
ax.set_xlabel("filas de entrenamiento (sólo cloud_marketplace & no_commitment)")
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0.7, 1.005)
ax.legend(fontsize=7.5, loc="lower right")
ax.set_title("Curva de aprendizaje dentro de la cohorte foco")
P.save(fig, "03_curva_aprendizaje_cohorte_foco")
save_table(pd.DataFrame({"n_train": n, "lgbm_train": tr_s.mean(1), "lgbm_val": va_s.mean(1),
                         "tree_train": tr2.mean(1), "tree_val": va2.mean(1)}), "03_curva_cohorte_foco")
