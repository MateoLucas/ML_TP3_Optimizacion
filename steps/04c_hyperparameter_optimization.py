"""Paso 4 – Estrategia 3: Hyperparameter optimization.

* LightGBM (mejor familia del Paso 4.2): Random Search (RandomizedSearchCV, 40
  configuraciones) vs. optimización bayesiana (Optuna TPE, 40 trials) sobre el
  mismo espacio y los mismos folds -> se compara convergencia y resultado.
* XGBoost (Optuna, 30 trials) y Random Forest (Random Search, 10 configs) se
  tunean también, para usarlos después como miembros de ensambles (4.4).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from tp3 import plotting as P
from tp3.config import MODEL_DIR, RANDOM_STATE
from tp3.data import duplicate_groups, get_data, xy
from tp3.evaluation import cross_validate, fmt_table, get_cv, save_table, summarize
from tp3.features import make_fe_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)
train, _ = get_data()
X, y = xy(train)
groups = duplicate_groups(train)
cv = get_cv()
DROP = json.loads((MODEL_DIR / "feature_selection.json").read_text())["drop"]
RS = RANDOM_STATE
N_LGBM = 40


def lgbm(**kw):
    return LGBMClassifier(verbose=-1, random_state=RS, n_jobs=1, subsample_freq=1, **kw)


def cv_auc(pipe):
    folds, _ = cross_validate(pipe, X, y, groups, cv=cv, n_jobs=5)
    return folds.val_roc_auc.mean()


# ======================================================= LightGBM: Random Search
space_rs = {
    "model__n_estimators": randint(200, 1500),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__num_leaves": randint(15, 256),
    "model__min_child_samples": randint(5, 100),
    "model__subsample": uniform(0.5, 0.5),
    "model__colsample_bytree": uniform(0.4, 0.6),
    "model__reg_lambda": loguniform(1e-3, 10),
    "model__reg_alpha": loguniform(1e-3, 10),
}
t0 = time.perf_counter()
rs = RandomizedSearchCV(make_fe_pipeline(lgbm(), scale=False, drop=DROP), space_rs, n_iter=N_LGBM,
                        scoring={"roc_auc": "roc_auc", "f1": "f1"}, refit=False, cv=cv, n_jobs=4,
                        random_state=RS)
rs.fit(X, y, groups=groups)
t_rs = time.perf_counter() - t0
rs_res = pd.DataFrame(rs.cv_results_)
rs_curve = rs_res.mean_test_roc_auc.to_numpy()
best_rs = rs_res.loc[rs_res.mean_test_roc_auc.idxmax()]
best_rs_params = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in best_rs.params.items()}
print(f"Random Search: best AUC={best_rs.mean_test_roc_auc:.4f} en {t_rs/60:.1f} min", best_rs_params, flush=True)


# ======================================================= LightGBM: Optuna (TPE)
def lgbm_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    }


t0 = time.perf_counter()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS, n_startup_trials=10))
study.optimize(lambda tr: cv_auc(make_fe_pipeline(lgbm(**lgbm_params(tr)), scale=False, drop=DROP)),
               n_trials=N_LGBM)
t_tpe = time.perf_counter() - t0
tpe_curve = np.array([t.value for t in study.trials])
print(f"Optuna TPE: best AUC={study.best_value:.4f} en {t_tpe/60:.1f} min", study.best_params, flush=True)

# convergencia
fig, ax = plt.subplots(figsize=(7, 3.8))
it = np.arange(1, N_LGBM + 1)
ax.scatter(it, rs_curve, s=14, color=P.BLUE, alpha=0.35)
ax.scatter(it, tpe_curve, s=14, color=P.ORANGE, alpha=0.35)
ax.plot(it, np.maximum.accumulate(rs_curve), color=P.BLUE, label="Random Search (mejor hasta i)")
ax.plot(it, np.maximum.accumulate(tpe_curve), color=P.ORANGE, label="Optuna TPE (mejor hasta i)")
ax.axhline(0.9390, color=P.TEXT2, ls="--", lw=1)
ax.text(N_LGBM, 0.9391, "LightGBM sin tunear", ha="right", va="bottom", fontsize=7.5, color=P.TEXT2)
ax.set_xlabel("iteración / trial")
ax.set_ylabel("ROC-AUC (5-fold CV)")
lo = min(np.percentile(rs_curve, 5), np.percentile(tpe_curve, 5))
ax.set_ylim(lo - 0.001, max(rs_curve.max(), tpe_curve.max()) + 0.002)
ax.legend(loc="lower right")
ax.set_title("Búsqueda de hiperparámetros de LightGBM")
P.save(fig, "04c_convergencia_hpo")

# ======================================================= XGBoost (Optuna) y RF (Random Search)
def xgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1, log=True),
    }


def xgb(**kw):
    return XGBClassifier(tree_method="hist", random_state=RS, n_jobs=1, eval_metric="logloss", **kw)


study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS, n_startup_trials=8))
study_xgb.optimize(lambda tr: cv_auc(make_fe_pipeline(xgb(**xgb_params(tr)), scale=False, drop=DROP)), n_trials=30)
print(f"XGBoost TPE: best AUC={study_xgb.best_value:.4f}", study_xgb.best_params, flush=True)

space_rf = {
    "model__max_features": uniform(0.1, 0.6),
    "model__min_samples_leaf": randint(1, 10),
    "model__max_depth": [None, 20, 30],
    "model__class_weight": [None, "balanced_subsample"],
}
rs_rf = RandomizedSearchCV(make_fe_pipeline(RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=RS),
                                            scale=False, drop=DROP),
                           space_rf, n_iter=10, scoring="roc_auc", cv=cv, n_jobs=4, random_state=RS)
rs_rf.fit(X, y, groups=groups)
best_rf_params = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in rs_rf.best_params_.items()}
print(f"RF Random Search: best AUC={rs_rf.best_score_:.4f}", best_rf_params, flush=True)

# ======================================================= evaluación final (mismos folds, métricas completas)
best_lgbm_params = study.best_params if study.best_value >= best_rs.mean_test_roc_auc else best_rs_params
hpo = {
    "lightgbm": best_lgbm_params,
    "lightgbm_random_search": best_rs_params,
    "lightgbm_tpe": study.best_params,
    "xgboost": study_xgb.best_params,
    "random_forest": {**best_rf_params, "n_estimators": 500},
}
(MODEL_DIR / "hpo_params.json").write_text(json.dumps(hpo, indent=2, default=str))

focus = ((train.client_type == "cloud_marketplace") & (train.reservation_commitment == "no_commitment")).to_numpy()
final = {
    "LightGBM sin tunear": make_fe_pipeline(lgbm(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=1.0),
                                            scale=False, drop=DROP),
    "LightGBM – Random Search": make_fe_pipeline(lgbm(**best_rs_params), scale=False, drop=DROP),
    "LightGBM – Optuna TPE": make_fe_pipeline(lgbm(**study.best_params), scale=False, drop=DROP),
    "XGBoost sin tunear": make_fe_pipeline(xgb(n_estimators=400, learning_rate=0.05, max_depth=6), scale=False, drop=DROP),
    "XGBoost – Optuna TPE": make_fe_pipeline(xgb(**study_xgb.best_params), scale=False, drop=DROP),
    "Random Forest sin tunear": make_fe_pipeline(RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=RS),
                                                 scale=False, drop=DROP),
    "Random Forest – Random Search": make_fe_pipeline(
        RandomForestClassifier(n_jobs=1, random_state=RS, **hpo["random_forest"]), scale=False, drop=DROP),
}
rows, fold_auc = [], {}
for name, est in final.items():
    folds, oof = cross_validate(est, X, y, groups, cv=cv, n_jobs=5)
    r = summarize(name, folds)
    r["foco_roc_auc"] = roc_auc_score(y[focus], oof[focus])
    r["foco_f1"] = f1_score(y[focus], oof[focus] >= 0.5)
    rows.append(r)
    fold_auc[name] = folds.val_roc_auc.to_numpy()
    print(f"{name:32s} AUC={r['val_roc_auc']:.4f} F1={r['val_f1']:.4f}", flush=True)
res = pd.DataFrame(rows)
pairs = {"LightGBM – Random Search": "LightGBM sin tunear", "LightGBM – Optuna TPE": "LightGBM sin tunear",
         "XGBoost – Optuna TPE": "XGBoost sin tunear", "Random Forest – Random Search": "Random Forest sin tunear"}
res["delta_auc_vs_sin_tunear"] = [np.mean(fold_auc[n] - fold_auc[pairs[n]]) if n in pairs else np.nan
                                  for n in res.model]
res["folds_que_mejora"] = [int((fold_auc[n] > fold_auc[pairs[n]]).sum()) if n in pairs else np.nan
                           for n in res.model]
save_table(res, "04c_hpo_raw")
t = fmt_table(res)
t["Δ AUC pareado"] = res.delta_auc_vs_sin_tunear.map(lambda v: "" if np.isnan(v) else f"{v:+.4f}")
t["folds que mejora"] = res.folds_que_mejora.map(lambda v: "" if np.isnan(v) else f"{int(v)}/5")
t["AUC cohorte foco"] = res.foco_roc_auc.map("{:.4f}".format)
t["F1 cohorte foco"] = res.foco_f1.map("{:.4f}".format)
save_table(t, "04c_hpo")
print(t.to_string(index=False))
save_table(pd.DataFrame([
    {"técnica": "Random Search", "evaluaciones": N_LGBM, "mejor AUC (CV)": best_rs.mean_test_roc_auc,
     "minutos": t_rs / 60, "iteración del mejor": int(np.argmax(rs_curve)) + 1},
    {"técnica": "Optuna TPE", "evaluaciones": N_LGBM, "mejor AUC (CV)": study.best_value,
     "minutos": t_tpe / 60, "iteración del mejor": int(np.argmax(tpe_curve)) + 1},
]), "04c_random_vs_tpe")
pd.DataFrame([{"param": k, "Random Search": best_rs_params.get(k), "Optuna TPE": study.best_params.get(k)}
              for k in study.best_params]).pipe(save_table, "04c_mejores_params_lgbm")
