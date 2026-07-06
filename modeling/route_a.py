from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from .dataset import VideoRecord
from .features import extract_features
from .loso import PredictFn


def _design_matrix(records, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, y_cog, y_emo = [], [], []
    for r in records:
        if not r.has_label or r.subject_id not in labels:
            continue
        rows.append(extract_features(r))
        y_cog.append(labels[r.subject_id].cognitive_score)
        y_emo.append(labels[r.subject_id].emotion_score)
    if rows:
        X = np.vstack(rows).astype(np.float32)
    else:
        X = np.zeros((0, 0), dtype=np.float32)
    return X, np.array(y_cog, dtype=np.float32), np.array(y_emo, dtype=np.float32)


def make_sklearn_predictor(factory: Callable[[], object]) -> PredictFn:
    def predict(train_records, labels, test_records):
        Xtr, y_cog, y_emo = _design_matrix(train_records, labels)
        m_cog = factory().fit(Xtr, y_cog)
        m_emo = factory().fit(Xtr, y_emo)
        if test_records:
            Xte = np.vstack([extract_features(r) for r in test_records]).astype(np.float32)
        else:
            Xte = np.zeros((0, Xtr.shape[1]), dtype=np.float32)
        pc = m_cog.predict(Xte)
        pe = m_emo.predict(Xte)
        return {r.video_id: (float(pc[i]), float(pe[i])) for i, r in enumerate(test_records)}

    return predict


def make_ridge_predictor(alpha: float = 1.0) -> PredictFn:
    return make_sklearn_predictor(lambda: Ridge(alpha=alpha))


def make_svr_predictor(C: float = 1.0) -> PredictFn:
    return make_sklearn_predictor(lambda: SVR(C=C))


def make_rf_predictor(n_estimators: int = 300, random_state: int = 0) -> PredictFn:
    return make_sklearn_predictor(
        lambda: RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    )


MODEL_FACTORIES: dict[str, Callable[[], PredictFn]] = {
    "ridge": make_ridge_predictor,
    "svr": make_svr_predictor,
    "rf": make_rf_predictor,
}
