"""Validación cruzada, métricas y utilidades de reporte."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .config import N_FOLDS, RANDOM_STATE, TAB_DIR

# Métricas principales elegidas (ver README, sección 2.3)
PRIMARY = ["roc_auc", "f1"]


def get_cv(n_splits: int = N_FOLDS, random_state: int = RANDOM_STATE):
    """K-fold estratificado y agrupado por duplicados, sólo sobre train."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def score_proba(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "f1": f1_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
    }


def predict_score(model, X):
    """Probabilidad de la clase positiva (o decision_function reescalada)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return 1 / (1 + np.exp(-d))


def _fit_fold(estimator, X, y, tr, va, train_transform=None, fit_params=None):
    est = clone(estimator)
    Xtr, ytr = X.iloc[tr], y[tr]
    if train_transform is not None:  # p.ej. deduplicar / re-muestrear sólo el fold de train
        Xtr, ytr = train_transform(Xtr, ytr)
    t0 = time.perf_counter()
    est.fit(Xtr, ytr, **(fit_params or {}))
    fit_time = time.perf_counter() - t0
    p_tr = predict_score(est, Xtr)
    p_va = predict_score(est, X.iloc[va])
    return va, p_va, score_proba(ytr, p_tr), fit_time


def cross_validate(estimator, X: pd.DataFrame, y: np.ndarray, groups, cv=None,
                   n_jobs: int = 1, train_transform=None, fit_params=None):
    """CV propia: devuelve métricas por fold (train y validación) y las
    probabilidades out-of-fold (OOF) para el análisis de errores."""
    cv = cv or get_cv()
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_fold)(estimator, X, y, tr, va, train_transform, fit_params)
        for tr, va in cv.split(X, y, groups)
    )
    oof = np.zeros(len(y))
    rows = []
    for k, (va, p_va, tr_scores, fit_time) in enumerate(out):
        oof[va] = p_va
        va_scores = score_proba(y[va], p_va)
        rows.append({"fold": k, "fit_time": fit_time,
                     **{f"val_{m}": v for m, v in va_scores.items()},
                     **{f"train_{m}": v for m, v in tr_scores.items()}})
    return pd.DataFrame(rows), oof


def summarize(name: str, folds: pd.DataFrame) -> dict:
    """mean ± std de cada métrica a lo largo de los folds."""
    d = {"model": name}
    for m in ["roc_auc", "f1", "accuracy", "precision", "recall"]:
        d[f"val_{m}"] = folds[f"val_{m}"].mean()
        d[f"val_{m}_std"] = folds[f"val_{m}"].std()
    d["train_roc_auc"] = folds["train_roc_auc"].mean()
    d["train_f1"] = folds["train_f1"].mean()
    d["fit_time_s"] = folds["fit_time"].mean()
    return d


def fmt_table(df: pd.DataFrame, cols=None, std=True) -> pd.DataFrame:
    """Tabla compacta 'media ± std' para el README."""
    cols = cols or ["roc_auc", "f1", "accuracy", "precision", "recall"]
    out = pd.DataFrame({"modelo": df["model"]})
    for m in cols:
        if std and f"val_{m}_std" in df:
            out[m] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(df[f"val_{m}"], df[f"val_{m}_std"])]
        else:
            out[m] = df[f"val_{m}"].map("{:.4f}".format)
    if "train_roc_auc" in df:
        out["train roc_auc"] = df["train_roc_auc"].map("{:.4f}".format)
    if "fit_time_s" in df:
        out["fit (s)"] = df["fit_time_s"].map("{:.1f}".format)
    return out


def save_table(df: pd.DataFrame, name: str, index: bool = False):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{name}.csv", index=index)
    (TAB_DIR / f"{name}.md").write_text(df.to_markdown(index=index, floatfmt=".4f"))


def cohort_report(df: pd.DataFrame, y, proba, col: str, threshold: float = 0.5,
                  min_size: int = 30) -> pd.DataFrame:
    """Métricas por subgrupo: tamaño, tasa de cancelación, AUC, F1, FPR, FNR,
    cantidad de errores y % del total de errores que aporta el subgrupo."""
    pred = (proba >= threshold).astype(int)
    tmp = pd.DataFrame({"g": df[col].values, "y": y, "p": proba, "pred": pred})
    total_err = (tmp.y != tmp.pred).sum()
    rows = []
    for g, d in tmp.groupby("g"):
        if len(d) < min_size:
            continue
        fp = ((d.pred == 1) & (d.y == 0)).sum()
        fn = ((d.pred == 0) & (d.y == 1)).sum()
        neg, pos = (d.y == 0).sum(), (d.y == 1).sum()
        rows.append({
            col: g,
            "n": len(d),
            "share": len(d) / len(tmp),
            "cancel_rate": d.y.mean(),
            "pred_rate": d.pred.mean(),
            "roc_auc": roc_auc_score(d.y, d.p) if 0 < pos < len(d) else np.nan,
            "f1": f1_score(d.y, d.pred, zero_division=0),
            "accuracy": (d.y == d.pred).mean(),
            "FPR": fp / neg if neg else np.nan,
            "FNR": fn / pos if pos else np.nan,
            "FP": fp,
            "FN": fn,
            "errors_share": (fp + fn) / total_err,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
