"""Paso 4 – Estrategia 3: Hyperparameter optimization.

* LightGBM (mejor familia del Paso 4.2): Random Search (30 configuraciones al
  azar) vs. optimización bayesiana (Optuna TPE, 30 trials) sobre el MISMO espacio
  de búsqueda y los mismos folds -> se compara convergencia y resultado.
  Ambas búsquedas se implementan con Optuna (RandomSampler vs TPESampler) y se
  guardan en una base SQLite (models/optuna.db), así una corrida interrumpida
  se puede reanudar sin perder lo ya evaluado.
* XGBoost (TPE, 20 trials) y Random Forest (Random Search, 8 configs) se
  tunean también, para usarlos después como miembros de ensambles (4.4).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
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
N_LGBM = 30


def lgbm(**kw):
    return LGBMClassifier(verbose=-1, random_state=RS, n_jobs=1, subsample_freq=1, **kw)


def cv_auc(pipe):
    folds, _ = cross_validate(pipe, X, y, groups, cv=cv, n_jobs=5)
    return folds.val_roc_auc.mean()


STORAGE = f"sqlite:///{MODEL_DIR / 'optuna.db'}"


def run_study(name, make_sampler, objective, n_trials):
    """Crea (o reanuda) un estudio persistente y completa los trials que falten.
    La semilla del sampler se desplaza según los trials ya hechos, así al reanudar
    no se repiten las mismas configuraciones."""
    study = optuna.create_study(study_name=name, storage=STORAGE, direction="maximize", load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    study.sampler = make_sampler(RS + done)
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done,
                       callbacks=[lambda st, tr: print(f"  {name} #{tr.number}: {tr.value:.4f}", flush=True)])
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE][:n_trials]
    print(f"{name}: best AUC={max(t.value for t in trials):.4f}", flush=True)
    return study, trials, sum(t.duration.total_seconds() for t in trials)


# ======================================================= LightGBM: Random Search vs Optuna TPE
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


def lgbm_objective(tr):
    return cv_auc(make_fe_pipeline(lgbm(**lgbm_params(tr)), scale=False, drop=DROP))


_, rs_trials, t_rs = run_study("lgbm_random", lambda sd: optuna.samplers.RandomSampler(seed=sd), lgbm_objective, N_LGBM)
study, tpe_trials, t_tpe = run_study("lgbm_tpe", lambda sd: optuna.samplers.TPESampler(seed=sd, n_startup_trials=10),
                                     lgbm_objective, N_LGBM)
rs_curve = np.array([t.value for t in rs_trials])
tpe_curve = np.array([t.value for t in tpe_trials])
best_rs_trial = max(rs_trials, key=lambda t: t.value)
best_tpe_trial = max(tpe_trials, key=lambda t: t.value)
best_rs_params, best_tpe_params = best_rs_trial.params, best_tpe_trial.params

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


study_xgb, xgb_trials, _ = run_study(
    "xgb_tpe", lambda sd: optuna.samplers.TPESampler(seed=sd, n_startup_trials=8),
    lambda tr: cv_auc(make_fe_pipeline(xgb(**xgb_params(tr)), scale=False, drop=DROP)), 20)
best_xgb_params = max(xgb_trials, key=lambda t: t.value).params


def rf_params(trial):
    return {
        "max_features": trial.suggest_float("max_features", 0.1, 0.7),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_depth": trial.suggest_categorical("max_depth", [None, 20, 30]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced_subsample"]),
    }


_, rf_trials, _ = run_study(
    "rf_random", lambda sd: optuna.samplers.RandomSampler(seed=sd),
    lambda tr: cv_auc(make_fe_pipeline(RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=RS,
                                                              **rf_params(tr)), scale=False, drop=DROP)), 8)
best_rf_params = max(rf_trials, key=lambda t: t.value).params

# ======================================================= evaluación final (mismos folds, métricas completas)
best_lgbm_params = best_tpe_params if best_tpe_trial.value >= best_rs_trial.value else best_rs_params
hpo = {
    "lightgbm": best_lgbm_params,
    "lightgbm_random_search": best_rs_params,
    "lightgbm_tpe": best_tpe_params,
    "xgboost": best_xgb_params,
    "random_forest": {**best_rf_params, "n_estimators": 500},
}
(MODEL_DIR / "hpo_params.json").write_text(json.dumps(hpo, indent=2, default=str))

focus = ((train.client_type == "cloud_marketplace") & (train.reservation_commitment == "no_commitment")).to_numpy()
final = {
    "LightGBM sin tunear": make_fe_pipeline(lgbm(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=1.0),
                                            scale=False, drop=DROP),
    "LightGBM – Random Search": make_fe_pipeline(lgbm(**best_rs_params), scale=False, drop=DROP),
    "LightGBM – Optuna TPE": make_fe_pipeline(lgbm(**best_tpe_params), scale=False, drop=DROP),
    "XGBoost sin tunear": make_fe_pipeline(xgb(n_estimators=400, learning_rate=0.05, max_depth=6), scale=False, drop=DROP),
    "XGBoost – Optuna TPE": make_fe_pipeline(xgb(**best_xgb_params), scale=False, drop=DROP),
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
    {"técnica": "Random Search", "evaluaciones": N_LGBM, "mejor AUC (CV)": best_rs_trial.value,
     "minutos": t_rs / 60, "iteración del mejor": int(np.argmax(rs_curve)) + 1},
    {"técnica": "Optuna TPE", "evaluaciones": N_LGBM, "mejor AUC (CV)": best_tpe_trial.value,
     "minutos": t_tpe / 60, "iteración del mejor": int(np.argmax(tpe_curve)) + 1},
]), "04c_random_vs_tpe")
pd.DataFrame([{"param": k, "Random Search": best_rs_params.get(k), "Optuna TPE": best_tpe_params.get(k)}
              for k in best_tpe_params]).pipe(save_table, "04c_mejores_params_lgbm")
