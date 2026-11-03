"""Paso 5 – Modelo final.

1. Entrena baseline y modelo final con TODO el train y los evalúa UNA vez en test.
2. Intervalos de confianza por bootstrap (por grupos de duplicados) y diferencia
   pareada final − baseline.
3. Mejora por cohorte en test: ¿global o específica?
4. Contribución de cada estrategia (a partir de los resultados de CV del Paso 4).
5. Re-entrena el modelo final con train + test y lo guarda para la función predictora.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from tp3 import plotting as P
from tp3.config import MODEL_DIR, RANDOM_STATE, TAB_DIR, TARGET
from tp3.data import clean, duplicate_groups, get_data, load_raw, xy
from tp3.evaluation import cohort_report, save_table, score_proba
from tp3.final import build_final_model
from tp3.preprocessing import make_pipeline

cfg = json.loads((MODEL_DIR / "final_config.json").read_text())
THR = cfg.get("threshold", 0.5)
train, test = get_data()
Xtr, ytr = xy(train)
Xte, yte = xy(test)

baseline = make_pipeline(DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE), scale=False).fit(Xtr, ytr)
final = build_final_model(cfg).fit(Xtr, ytr)
p_base = baseline.predict_proba(Xte)[:, 1]
p_final = final.predict_proba(Xte)[:, 1]

# ------------------------------------------------------------------ 1. métricas en test
rows = [
    {"modelo": "Baseline (Árbol d=8)", "umbral": 0.5, **score_proba(yte, p_base, 0.5)},
    {"modelo": f"Final: {cfg['name']}", "umbral": 0.5, **score_proba(yte, p_final, 0.5)},
]
if THR != 0.5:
    rows.append({"modelo": f"Final: {cfg['name']}", "umbral": THR, **score_proba(yte, p_final, THR)})
test_tab = pd.DataFrame(rows)

# ------------------------------------------------------------------ 2. bootstrap por grupos
g = duplicate_groups(test)
uniq = np.unique(g)
idx_by_group = pd.Series(np.arange(len(g))).groupby(g).apply(np.array)
rng = np.random.default_rng(RANDOM_STATE)
boot = []
for _ in range(1000):
    gs = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate(idx_by_group.loc[gs].to_numpy())
    b, f = score_proba(yte[idx], p_base[idx], 0.5), score_proba(yte[idx], p_final[idx], THR)
    boot.append({**{f"final_{k}": v for k, v in f.items()}, **{f"base_{k}": v for k, v in b.items()},
                 **{f"delta_{k}": f[k] - b[k] for k in f}})
boot = pd.DataFrame(boot)
ci = []
for m in ["roc_auc", "f1", "accuracy", "precision", "recall"]:
    ci.append({"métrica": m,
               "final (test)": score_proba(yte, p_final, THR)[m],
               "IC95% final": f"[{boot[f'final_{m}'].quantile(.025):.4f}, {boot[f'final_{m}'].quantile(.975):.4f}]",
               "baseline (test)": score_proba(yte, p_base, 0.5)[m],
               "Δ final − baseline": score_proba(yte, p_final, THR)[m] - score_proba(yte, p_base, 0.5)[m],
               "IC95% Δ": f"[{boot[f'delta_{m}'].quantile(.025):+.4f}, {boot[f'delta_{m}'].quantile(.975):+.4f}]"})
ci = pd.DataFrame(ci)
save_table(test_tab, "05_test_metricas")
save_table(ci, "05_test_ic_bootstrap")
print(test_tab.round(4).to_string(index=False))
print(ci.round(4).to_string(index=False))

# CV (Paso 4) vs test
cv_rows = pd.read_csv(TAB_DIR / cfg["cv_table"])
cv_row = cv_rows[cv_rows.model == cfg["cv_row"]].iloc[0]
cv_thr = cv_rows[cv_rows.model == cfg.get("cv_row_threshold", cfg["cv_row"])].iloc[0]
cv_vs_test = pd.DataFrame([
    {"métrica": m, "CV (train, 5 folds)": f"{r[f'val_{m}']:.4f} ± {r[f'val_{m}_std']:.4f}",
     "test": score_proba(yte, p_final, THR)[m]}
    for m, r in [("roc_auc", cv_row), ("f1", cv_thr), ("precision", cv_thr), ("recall", cv_thr)]
])
save_table(cv_vs_test, "05_cv_vs_test")
print(cv_vs_test.to_string(index=False))

# matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for ax, (name, p, t) in zip(axes, [("Baseline", p_base, 0.5), ("Final", p_final, THR)]):
    cm = confusion_matrix(yte, (p >= t).astype(int))
    ax.imshow(cm, cmap=P.SEQ_CMAP, vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else P.TEXT, fontsize=11)
    ax.set_xticks([0, 1], ["pred. 0", "pred. 1"])
    ax.set_yticks([0, 1], ["real 0", "real 1"])
    ax.grid(False)
    ax.set_title(f"{name} (test, umbral {t:.2f})")
P.save(fig, "05_matrices_confusion_test")

# ------------------------------------------------------------------ 3. por cohorte en test
T = test.assign(
    cohorte_foco=np.where((test.client_type == "cloud_marketplace") & (test.reservation_commitment == "no_commitment"),
                          "cloud_marketplace & no_commitment", "resto"),
    lead_time_bin=pd.cut(test.hours_before_execution, [-1, 7, 30, 90, 180, 365, 1e9],
                         labels=["0–7", "8–30", "31–90", "91–180", "181–365", ">365"]).astype(str),
)
coh_rows = []
for col in ["cohorte_foco", "client_type", "service_tier", "reserved_gpu_type", "reservation_commitment",
            "lead_time_bin"]:
    a = cohort_report(T, yte, p_base, col, 0.5)
    b = cohort_report(T, yte, p_final, col, THR)
    m = a[[col, "n", "cancel_rate", "roc_auc", "f1", "FNR", "FPR", "FP", "FN"]].merge(
        b[[col, "roc_auc", "f1", "FNR", "FPR", "FP", "FN"]], on=col, suffixes=("_base", "_final"))
    m["delta_f1"] = m.f1_final - m.f1_base
    m["delta_auc"] = m.roc_auc_final - m.roc_auc_base
    m["errores_evitados"] = (m.FP_base + m.FN_base) - (m.FP_final + m.FN_final)
    m.insert(0, "variable", col)
    coh_rows.append(m.rename(columns={col: "valor"}))
coh = pd.concat(coh_rows, ignore_index=True)
total_avoided = ((p_base >= 0.5) != yte).sum() - ((p_final >= THR) != yte).sum()
coh["share_errores_evitados"] = coh.errores_evitados / total_avoided
save_table(coh, "05_test_por_cohorte")
print(coh[["variable", "valor", "n", "f1_base", "f1_final", "delta_f1", "roc_auc_base", "roc_auc_final",
           "errores_evitados"]].round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True)
for ax, col in zip(axes, ["client_type", "service_tier", "reserved_gpu_type"]):
    d = coh[coh.variable == col].sort_values("n")
    yy = np.arange(len(d))
    for i, (a, b) in enumerate(zip(d.f1_base, d.f1_final)):
        ax.plot([a, b], [i, i], color=P.GRID, lw=3, zorder=1)
    ax.scatter(d.f1_base, yy, color=P.TEXT2, s=40, zorder=2, label="baseline")
    ax.scatter(d.f1_final, yy, color=P.ORANGE, s=40, zorder=3, label="final")
    for i, (b, dd) in enumerate(zip(d.f1_final, d.delta_f1)):
        ax.text(max(b, d.f1_base.iloc[i]) + 0.02, i, f"{dd:+.3f}", va="center", fontsize=8, color=P.TEXT2)
    ax.set_yticks(yy, [f"{v} (n={n:,})" for v, n in zip(d.valor, d.n)], fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_title(f"F1 por {col} (test)")
    ax.grid(axis="y", visible=False)
axes[0].legend(loc="lower right")
fig.suptitle("¿La mejora es global o específica? F1 baseline → final por cohorte (test)", fontweight="bold")
P.save(fig, "05_mejora_por_cohorte")

# ------------------------------------------------------------------ 4. contribución de cada estrategia (CV)
def cv_metric(table, model):
    r = pd.read_csv(TAB_DIR / table)
    r = r[r.model == model].iloc[0]
    return r.val_roc_auc, r.val_f1


steps = [
    ("Baseline: Árbol d=8", "02_comparacion_baseline_raw.csv", "Árbol de Decisión"),
    ("4.1 + FE (mismo árbol)", "04a_feature_engineering_raw.csv", "Árbol d=8 | + FE (one-hot IDs)"),
    ("4.2 LightGBM + FE + selección", "04b_model_selection_raw.csv", "LightGBM"),
    ("4.3 + HPO (Optuna)", "04c_hpo_raw.csv", "LightGBM – Optuna TPE"),
] + [tuple(s) for s in cfg.get("extra_steps", [])]
prog = []
for label, table, model in steps:
    auc, f1 = cv_metric(table, model)
    prog.append({"paso": label, "roc_auc": auc, "f1": f1})
if THR != 0.5:
    r = pd.read_csv(TAB_DIR / "04e_data_centered_raw.csv")
    r = r[r.model.str.startswith("b. umbral")].iloc[0]
    prog.append({"paso": f"4.5 + umbral {THR:.2f}", "roc_auc": prog[-1]["roc_auc"], "f1": r.val_f1})
prog = pd.DataFrame(prog)
prog["Δ roc_auc"] = prog.roc_auc.diff()
prog["Δ f1"] = prog.f1.diff()
save_table(prog, "05_contribucion_estrategias")
print(prog.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
for ax, m, color in zip(axes, ["roc_auc", "f1"], [P.BLUE, P.ORANGE]):
    vals = prog[m].to_numpy()
    base = vals[0]
    for i, v in enumerate(vals):
        prev = vals[i - 1] if i else base
        if i == 0:
            ax.bar(i, v - (base - 0.02), bottom=base - 0.02, color=P.TEXT2, width=0.6)
        else:
            ax.bar(i, v - prev, bottom=prev, color=color if v >= prev else P.RED, width=0.6)
        ax.text(i, max(v, prev) + 0.0015, f"{v:.4f}", ha="center", fontsize=7.5, color=P.TEXT2)
    ax.set_xticks(range(len(vals)), prog.paso, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylim(base - 0.02, vals.max() + 0.01)
    ax.set_title({"roc_auc": "ROC-AUC (CV)", "f1": "F1 (CV)"}[m])
    ax.grid(axis="x", visible=False)
fig.suptitle("Contribución de cada estrategia (5-fold CV en train)", fontweight="bold")
P.save(fig, "05_contribucion_estrategias")

# ------------------------------------------------------------------ 5. modelo entregable: train + test
full = clean(load_raw())
Xall, yall = xy(full)
final_full = build_final_model(cfg).fit(Xall, yall)
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(final_full, MODEL_DIR / "final_model.joblib", compress=3)
print("modelo guardado:", (MODEL_DIR / "final_model.joblib").stat().st_size / 1e6, "MB")

# sanity check de la función predictora (sobre el CSV crudo)
from predictor.predictor_v1_Grupo_X import predictor_v1_Grupo_X  # noqa: E402

raw = load_raw()
pred = predictor_v1_Grupo_X(raw.drop(columns=[TARGET]))
print("predictor OK:", pred.shape, "tasa predicha:", pred.mean().round(4),
      "acc. (in-sample, sólo chequeo):", (pred == raw[TARGET].to_numpy()).mean().round(4))
