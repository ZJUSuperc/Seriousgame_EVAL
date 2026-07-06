from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .dataset import VideoRecord
from .eval import regression_metrics
from .labels import SubjectLabel
from .splits import loso_folds

PredictFn = Callable[
    [Sequence[VideoRecord], dict[str, SubjectLabel], Sequence[VideoRecord]],
    dict[str, tuple[float, float]],
]


def run_loso(
    records: Sequence[VideoRecord],
    labels: dict[str, SubjectLabel],
    predict_fn: PredictFn,
    return_predictions: bool = False,
) -> dict:
    subject_ids = sorted({r.subject_id for r in records})
    folds = loso_folds(subject_ids)

    subject_preds: dict[str, list[tuple[float, float]]] = {}
    for train_ids, test_id in folds:
        train_recs = [r for r in records if r.subject_id in train_ids]
        test_recs = [r for r in records if r.subject_id == test_id]
        if not test_recs:
            continue
        preds = predict_fn(train_recs, labels, test_recs)
        subject_preds[test_id] = [preds[r.video_id] for r in test_recs]

    cog_true, cog_pred, emo_true, emo_pred = [], [], [], []
    aggregated: dict[str, tuple[float, float]] = {}
    for sid, plist in subject_preds.items():
        cps = np.array([p[0] for p in plist], dtype=float)
        eps = np.array([p[1] for p in plist], dtype=float)
        cp = float(cps.mean())
        ep = float(eps.mean())
        aggregated[sid] = (cp, ep)
        cog_true.append(labels[sid].cognitive_score)
        emo_true.append(labels[sid].emotion_score)
        cog_pred.append(cp)
        emo_pred.append(ep)

    result = {
        "cognitive": regression_metrics(np.array(cog_true), np.array(cog_pred)),
        "emotion": regression_metrics(np.array(emo_true), np.array(emo_pred)),
        "n_test_subjects": len(subject_preds),
    }
    if return_predictions:
        result["predictions"] = aggregated
    return result


def permutation_test(
    records: Sequence[VideoRecord],
    labels: dict[str, SubjectLabel],
    predict_fn: PredictFn,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """置换检验:打乱被试→分数映射作为零假设,返回每个目标 pearson_r 的 p 值。"""
    rng = np.random.default_rng(seed)
    observed = run_loso(records, labels, predict_fn)
    obs_r = {t: observed[t]["pearson_r"] for t in ("cognitive", "emotion")}

    sids = sorted(labels.keys())
    cog = np.array([labels[s].cognitive_score for s in sids], dtype=float)
    emo = np.array([labels[s].emotion_score for s in sids], dtype=float)

    null = {"cognitive": [], "emotion": []}
    for _ in range(n_permutations):
        perm = rng.permutation(len(sids))
        shuffled = {
            sids[i]: SubjectLabel(sids[i], float(cog[perm[i]]), float(emo[perm[i]]))
            for i in range(len(sids))
        }
        res = run_loso(records, shuffled, predict_fn)
        null["cognitive"].append(res["cognitive"]["pearson_r"])
        null["emotion"].append(res["emotion"]["pearson_r"])

    p: dict[str, float] = {}
    for t in ("cognitive", "emotion"):
        arr = np.array([v for v in null[t] if not np.isnan(v)])
        if arr.size == 0 or np.isnan(obs_r[t]):
            p[t] = float("nan")
        else:
            p[t] = float(np.mean(arr >= obs_r[t]))
    return {"observed_pearson_r": obs_r, "p_value": p, "n_permutations": n_permutations}
