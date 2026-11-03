"""Construcción del modelo final a partir de models/final_config.json (Paso 5)."""
from __future__ import annotations

import json

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import MODEL_DIR, RANDOM_STATE
from .ensembles import GroupStackingClassifier, SeedBaggingClassifier
from .features import make_fe_pipeline


def _members(hpo, drop):
    lgbm = make_fe_pipeline(LGBMClassifier(verbose=-1, random_state=RANDOM_STATE, n_jobs=1, subsample_freq=1,
                                           **hpo["lightgbm"]), scale=False, drop=drop)
    xgb = make_fe_pipeline(XGBClassifier(tree_method="hist", random_state=RANDOM_STATE, n_jobs=1,
                                         eval_metric="logloss", **hpo["xgboost"]), scale=False, drop=drop)
    rf = make_fe_pipeline(RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE, **hpo["random_forest"]),
                          scale=False, drop=drop)
    lr = make_fe_pipeline(LogisticRegression(C=0.1, max_iter=5000), high_card="target")
    return {"lgbm": lgbm, "xgb": xgb, "rf": rf, "lr": lr}


def build_final_model(cfg: dict | None = None):
    cfg = cfg or json.loads((MODEL_DIR / "final_config.json").read_text())
    hpo = json.loads((MODEL_DIR / "hpo_params.json").read_text())
    drop = json.loads((MODEL_DIR / "feature_selection.json").read_text())["drop"]
    m = _members(hpo, drop)
    kind = cfg["type"]
    if kind == "lgbm":
        return m["lgbm"]
    if kind == "lgbm_seed_bagging":
        return SeedBaggingClassifier(m["lgbm"], seeds=tuple(cfg.get("seeds", (0, 1, 2, 3, 4))),
                                     seed_param="model__random_state")
    if kind == "voting":
        return VotingClassifier([(k, m[k]) for k in cfg["members"]], voting="soft")
    if kind == "stacking":
        return GroupStackingClassifier([(k, m[k]) for k in cfg["members"]],
                                       final_estimator=LogisticRegression(C=1.0))
    raise ValueError(kind)
