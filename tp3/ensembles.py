"""Ensambles propios (Paso 4.4).

`GroupStackingClassifier`: stacking cuyo CV interno (el que genera las
predicciones OOF con las que se entrena el meta-modelo) también está agrupado
por duplicados. Con el `StackingClassifier` de sklearn y un KFold común, las
copias de una misma reserva quedarían repartidas entre los folds internos: los
miembros que "memorizan" (RF) parecerían mejores de lo que son y el meta-modelo
les daría demasiado peso.

`SeedBaggingClassifier`: promedia el mismo modelo entrenado con distintas
semillas (bagging de la aleatoriedad de subsample / colsample).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from .config import RANDOM_STATE
from .data import duplicate_groups


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class GroupStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, final_estimator=None, n_splits=5, random_state=RANDOM_STATE, n_jobs=1):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        groups = duplicate_groups(X)
        cv = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        splits = list(cv.split(X, y, groups))
        meta_X = np.column_stack([
            cross_val_predict(clone(est), X, y, cv=splits, method="predict_proba", n_jobs=self.n_jobs)[:, 1]
            for _, est in self.estimators
        ])
        self.final_estimator_ = clone(self.final_estimator or LogisticRegression(C=1.0))
        self.final_estimator_.fit(_logit(meta_X), y)
        self.estimators_ = [clone(est).fit(X, y) for _, est in self.estimators]
        self.classes_ = np.unique(y)
        return self

    def member_proba(self, X):
        return np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(_logit(self.member_proba(X)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SeedBaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, seeds=(0, 1, 2, 3, 4), seed_param="model__random_state"):
        self.estimator = estimator
        self.seeds = seeds
        self.seed_param = seed_param

    def fit(self, X, y, **fit_params):
        self.estimators_ = [clone(self.estimator).set_params(**{self.seed_param: s}).fit(X, y, **fit_params)
                            for s in self.seeds]
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.mean([e.predict_proba(X) for e in self.estimators_], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
