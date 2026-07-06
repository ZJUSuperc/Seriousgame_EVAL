from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats

from .labels import SubjectLabel


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r, p = stats.pearsonr(y_pred, y_true)
        r, p = float(r), float(p)
    else:
        r, p = float("nan"), float("nan")
    std_y = y_true.std(ddof=0)
    normalized_mae = mae / std_y if std_y > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": r,
        "pearson_p": p,
        "normalized_mae": normalized_mae,
    }


def mean_baseline(
    labels: dict[str, SubjectLabel], train_subject_ids: Iterable[str]
) -> tuple[float, float]:
    ids = [s for s in train_subject_ids if s in labels]
    cog = float(np.mean([labels[s].cognitive_score for s in ids]))
    emo = float(np.mean([labels[s].emotion_score for s in ids]))
    return cog, emo
