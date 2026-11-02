"""Paso 1 – Preprocesamiento y análisis exploratorio.

Genera:
  reports/tables/01_*.md|csv   (calidad de datos, leakage, cohortes)
  reports/figures/01_*.png
Todo el análisis exploratorio se hace SÓLO sobre train (después del split),
salvo el chequeo de calidad global del CSV crudo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tp3 import plotting as P
from tp3.config import CATEGORICAL_COLS, FEATURE_COLS, LEAKAGE_COLS, NUMERIC_COLS, TARGET
from tp3.data import clean, duplicate_groups, load_raw, train_test_split_grouped
from tp3.evaluation import save_table

raw = load_raw()

# ---------------------------------------------------------------- calidad de datos
quality = pd.DataFrame({
    "dtype": raw.dtypes.astype(str),
    "n_missing": raw.isna().sum(),
    "pct_missing": raw.isna().mean() * 100,
    "n_unique": raw.nunique(),
}).rename_axis("columna").reset_index()
save_table(quality, "01_calidad_columnas")

dup_all = raw.duplicated().sum()
feat_raw = [c for c in raw.columns if c not in LEAKAGE_COLS + [TARGET]]
dup_feat = raw[feat_raw].duplicated().sum()
is_dup = raw[feat_raw].duplicated(keep=False)
summary = pd.DataFrame([
    ("filas", len(raw)),
    ("columnas", raw.shape[1]),
    ("tasa de cancelación", f"{raw[TARGET].mean():.3f}"),
    ("filas duplicadas exactas (todas las columnas)", dup_all),
    ("filas duplicadas en features (sin target ni leakage)", dup_feat),
    ("% filas que pertenecen a un grupo de duplicados", f"{is_dup.mean()*100:.1f}%"),
    ("tasa de cancelación en filas duplicadas", f"{raw.loc[is_dup, TARGET].mean():.3f}"),
    ("tasa de cancelación en filas únicas", f"{raw.loc[~is_dup, TARGET].mean():.3f}"),
    ("grupos de duplicados con etiquetas contradictorias",
     int(raw.groupby(feat_raw, dropna=False)[TARGET].agg(lambda s: s.nunique() > 1).sum())),
], columns=["chequeo", "valor"])
save_table(summary, "01_resumen_dataset")

# ---------------------------------------------------------------- leakage
leak = pd.crosstab(raw["execution_status"], raw[TARGET]).rename(columns={0: "canceled_job=0", 1: "canceled_job=1"})
save_table(leak.reset_index(), "01_leakage_execution_status")
ed = pd.to_datetime(dict(year=raw.execution_year, month=raw.execution_month, day=raw.execution_day))
sd = pd.to_datetime(raw.execution_status_date)
gap = (ed - sd).dt.days
leak_date = raw.assign(dias_estado_antes_de_ejecucion=gap).groupby(TARGET)["dias_estado_antes_de_ejecucion"].describe()
save_table(leak_date.reset_index(), "01_leakage_status_date")

# columnas derivadas (no documentadas en la consigna)
derived = pd.DataFrame([
    ("total_processes == primary + auxiliary + background",
     bool(np.allclose(raw.total_processes, raw[["num_primary_processes", "num_auxiliary_processes",
                                                "num_background_processes"]].sum(axis=1)))),
    ("total_gpu_hours == peak + offpeak",
     bool((raw.total_gpu_hours == raw.peak_gpu_hours + raw.offpeak_gpu_hours).all())),
    ("high_value_client: costo/h medio (0 vs 1)",
     " vs ".join(f"{v:.2f}" for v in raw.groupby("high_value_client").estimated_cost_per_hour.mean())),
], columns=["chequeo", "resultado"])
save_table(derived, "01_columnas_derivadas")

# anomalías / valores sospechosos
anom = pd.DataFrame([
    ("total_processes == 0 (workload sin procesos)", (raw.total_processes == 0).sum(),
     raw.loc[raw.total_processes == 0, TARGET].mean()),
    ("num_primary_processes == 0", (raw.num_primary_processes == 0).sum(),
     raw.loc[raw.num_primary_processes == 0, TARGET].mean()),
    ("total_gpu_hours == 0 (reserva de 0 horas)", (raw.total_gpu_hours == 0).sum(),
     raw.loc[raw.total_gpu_hours == 0, TARGET].mean()),
    ("estimated_cost_per_hour == 0", (raw.estimated_cost_per_hour == 0).sum(),
     raw.loc[raw.estimated_cost_per_hour == 0, TARGET].mean()),
    ("estimated_cost_per_hour > 20 (outlier extremo)", (raw.estimated_cost_per_hour > 20).sum(),
     raw.loc[raw.estimated_cost_per_hour > 20, TARGET].mean()),
    ("client_type / request_channel == 'unknown'",
     ((raw.client_type == "unknown") | (raw.request_channel == "unknown")).sum(),
     raw.loc[(raw.client_type == "unknown") | (raw.request_channel == "unknown"), TARGET].mean()),
], columns=["anomalía", "filas", "tasa_cancelación"])
save_table(anom, "01_anomalias")

# ---------------------------------------------------------------- split (antes del EDA)
df = clean(raw)
train, test = train_test_split_grouped(df)
split = pd.DataFrame([
    ("train", len(train), train[TARGET].mean(), len(np.unique(duplicate_groups(train)))),
    ("test", len(test), test[TARGET].mean(), len(np.unique(duplicate_groups(test)))),
], columns=["conjunto", "filas", "tasa_cancelación", "grupos_únicos"])
save_table(split, "01_split")
tr = train  # alias: TODO lo que sigue es sólo train

# ---------------------------------------------------------------- figuras
# 1) distribución del target
fig, ax = plt.subplots(figsize=(4.2, 3))
vc = tr[TARGET].value_counts().sort_index()
ax.bar(["no cancelada (0)", "cancelada (1)"], vc.values, color=[P.BLUE, P.ORANGE], width=0.55)
for i, v in enumerate(vc.values):
    ax.text(i, v, f"{v:,}\n({v/len(tr):.1%})", ha="center", va="bottom", fontsize=8.5, color=P.TEXT2)
ax.set_ylim(0, vc.max() * 1.25)
ax.set_title("Distribución del target (train)")
ax.grid(axis="x", visible=False)
P.save(fig, "01_target")

# 2) tasa de cancelación por variable categórica (cohortes)
tr_plot = tr.assign(
    gpu_type_changed=np.where(tr.reserved_gpu_type != tr.assigned_gpu_type, "distinto", "igual"),
    has_agent=np.where(tr.orchestration_agent_id == "none", "sin agente", "con agente"),
    has_organization=np.where(tr.organization_id == "none", "sin organización", "con organización"),
    returning=np.where(tr.returning_client == 1, "recurrente", "nuevo"),
    region=tr.client_region.where(tr.client_region.isin(tr.client_region.value_counts().index[:8]), "otras"),
)
cat_vars = ["client_type", "service_tier", "reserved_gpu_type", "reservation_commitment",
            "request_channel", "workload_profile", "region", "gpu_type_changed",
            "has_agent", "has_organization", "returning"]
base_rate = tr[TARGET].mean()
fig, axes = plt.subplots(4, 3, figsize=(13, 13))
cohort_rows = []
for ax, col in zip(axes.flat, cat_vars):
    g = tr_plot.groupby(col)[TARGET].agg(["mean", "size"])
    g = g[g["size"] >= 20].sort_values("mean")
    for name, r in g.iterrows():
        cohort_rows.append({"variable": col, "valor": name, "n": int(r["size"]),
                            "share": r["size"] / len(tr), "tasa_cancelación": r["mean"]})
    ax.barh(g.index.astype(str), g["mean"], color=P.BLUE, height=0.6)
    ax.axvline(base_rate, color=P.TEXT2, ls="--", lw=1)
    for y, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(m + 0.01, y, f"{m:.0%}  (n={n:,})", va="center", fontsize=7.5, color=P.TEXT2)
    ax.set_xlim(0, 1.25)
    ax.set_title(col)
    ax.grid(axis="y", visible=False)
axes.flat[-1].axis("off")
axes.flat[-1].text(0.05, 0.5, f"línea punteada = tasa global\n({base_rate:.1%})\n\nse muestran categorías\ncon n ≥ 20",
                   fontsize=10, color=P.TEXT2, va="center")
fig.suptitle("Tasa de cancelación por cohorte (train)", fontweight="bold")
P.save(fig, "01_cancel_rate_cohortes")
save_table(pd.DataFrame(cohort_rows), "01_cohortes_tasa_cancelacion")

# 3) variables numéricas: tasa de cancelación por rango de valores
def binned(x, spec):
    """spec = ('q', n) -> cuantiles; ('cap', k) -> 0,1,..,k-1,'k+'; ('edges', [...]) -> cortes fijos."""
    kind, arg = spec
    if kind == "cap":
        b = x.clip(upper=arg).astype(int)
        return b.map(lambda v: f"{v}+" if v == arg else str(v)), list(range(arg + 1))
    if kind == "q":
        edges = np.unique(np.quantile(x, np.linspace(0, 1, arg + 1)))
    else:
        edges = np.array(arg, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    cats = pd.cut(x, edges, right=True)
    fmt = (lambda v: f"{v:.1f}") if x.median() < 20 and kind == "q" else (lambda v: f"{v:.0f}")
    labels = {iv: (f"≤{fmt(iv.right)}" if np.isinf(iv.left) else
                   f">{fmt(iv.left)}" if np.isinf(iv.right) else f"{fmt(iv.left)}–{fmt(iv.right)}")
              for iv in cats.cat.categories}
    return cats.map(labels), None


num_specs = {
    "hours_before_execution": ("q", 10),
    "estimated_cost_per_hour": ("q", 10),
    "total_gpu_hours": ("edges", [0, 0, 8, 16, 24, 32, 48, 80, 1e9]),
    "special_resource_requests": ("cap", 4),
    "resource_modifications": ("cap", 4),
    "previous_job_cancellations": ("cap", 3),
    "successful_previous_jobs": ("cap", 3),
    "queue_wait_time": ("edges", [0, 0, 15, 30, 60, 1e9]),
    "execution_week": ("q", 10),
}
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, (col, spec) in zip(axes.flat, num_specs.items()):
    b, _ = binned(tr[col], spec)
    g = tr.groupby(b, observed=True, sort=False)[TARGET].agg(["mean", "size"])
    order = tr.groupby(b, observed=True)[col].median().sort_values().index
    g = g.loc[order]
    ax.plot(range(len(g)), g["mean"], marker="o", ms=5, color=P.ORANGE)
    ax.axhline(base_rate, color=P.TEXT2, ls="--", lw=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{i}\n(n={n:,})" if len(g) <= 6 else str(i) for i, n in zip(g.index, g["size"])],
                       rotation=0 if len(g) <= 6 else 45, ha="center" if len(g) <= 6 else "right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("tasa de cancelación")
    ax.set_title(col)
fig.suptitle("Tasa de cancelación según variables numéricas (train)", fontweight="bold")
P.save(fig, "01_cancel_rate_numericas")

# 4) distribuciones por clase (variables continuas clave)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, col, clip in zip(axes, ["hours_before_execution", "estimated_cost_per_hour", "total_gpu_hours"],
                         [500, 15, 120]):
    for k in [0, 1]:
        ax.hist(tr.loc[tr[TARGET] == k, col].clip(upper=clip), bins=40, density=True, alpha=0.55,
                color=P.CLASS_COLORS[k], label=["no cancelada", "cancelada"][k])
    ax.set_title(f"{col} (recortado en {clip})")
    ax.legend()
P.save(fig, "01_distribuciones_por_clase")

# 5) correlaciones (numéricas + target)
corr = tr[NUMERIC_COLS + [TARGET]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(corr, cmap=P.DIV_CMAP, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7.5)
ax.set_yticklabels(corr.columns, fontsize=7.5)
ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)):
        if abs(corr.iloc[i, j]) >= 0.3 and i != j:
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color=P.TEXT)
fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman")
ax.set_title("Correlación de Spearman (train)")
P.save(fig, "01_correlaciones")
save_table(corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).rename("spearman_vs_target")
           .reset_index().rename(columns={"index": "feature"}), "01_corr_target")

# 6) temporal: tasa de cancelación por mes de ejecución y año
fig, ax = plt.subplots(figsize=(7, 3.4))
for i, (yr, d) in enumerate(tr.groupby("execution_year")):
    g = d.groupby("execution_month")[TARGET].mean()
    ax.plot(g.index, g.values, marker="o", ms=5, label=str(yr), color=P.SERIES[i])
ax.set_xticks(range(1, 13))
ax.set_xlabel("mes de ejecución")
ax.set_ylabel("tasa de cancelación")
ax.set_ylim(0, 0.8)
ax.legend(title="año")
ax.set_title("Estacionalidad de la cancelación (train)")
P.save(fig, "01_temporal")

# 7) sub-grupos cruzados: client_type x reservation_commitment
ct = tr.pivot_table(index="client_type", columns="reservation_commitment", values=TARGET,
                    aggfunc=["mean", "size"])
save_table(ct["mean"].round(3).reset_index(), "01_client_type_x_commitment_tasa")
save_table(ct["size"].fillna(0).astype(int).reset_index(), "01_client_type_x_commitment_n")

print(summary.to_string(index=False))
print(anom.to_string(index=False))
print(split.to_string(index=False))
print(corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(12))
print(ct["mean"].round(3))
print(ct["size"])
