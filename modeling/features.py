from __future__ import annotations

import numpy as np

from .dataset import AU_COLUMNS, VideoRecord

AU_THRESHOLD = 0.5
_STAT_NAMES = ("mean", "std", "act", "median", "p90")


def feature_names() -> list[str]:
    aus = [c.replace("_r", "") for c in AU_COLUMNS]
    names = [f"{a}_{s}" for a in aus for s in _STAT_NAMES]
    names += [f"{a}_delta" for a in aus]
    names += ["global_mean_intensity", "any_active_frame_ratio", "valid_ratio"]
    return names


def _per_au_stats(au: np.ndarray) -> np.ndarray:
    if au.shape[0] == 0:
        return np.zeros((au.shape[1], len(_STAT_NAMES)), dtype=np.float32)
    mean = au.mean(axis=0)
    std = au.std(axis=0)
    act = (au > AU_THRESHOLD).mean(axis=0)
    median = np.median(au, axis=0)
    p90 = np.quantile(au, 0.9, axis=0)
    return np.stack([mean, std, act, median, p90], axis=1).astype(np.float32)


def _temporal_dynamics(au: np.ndarray) -> np.ndarray:
    if au.shape[0] < 2:
        return np.zeros(au.shape[1], dtype=np.float32)
    return np.mean(np.abs(np.diff(au, axis=0)), axis=0).astype(np.float32)


def extract_features(record: VideoRecord) -> np.ndarray:
    au = record.au
    if au.ndim != 2 or au.shape[0] == 0:
        return np.zeros(len(feature_names()), dtype=np.float32)
    stats = _per_au_stats(au).reshape(-1)
    dyn = _temporal_dynamics(au)
    global_feats = np.array(
        [
            float(au.mean()),
            float((au > AU_THRESHOLD).any(axis=1).mean()),
            float(record.valid_ratio),
        ],
        dtype=np.float32,
    )
    return np.concatenate([stats, dyn, global_feats]).astype(np.float32)
