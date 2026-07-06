from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .labels import SubjectLabel


@dataclass
class TargetNormalizer:
    cog_mean: float
    cog_std: float
    emo_mean: float
    emo_std: float

    def transform(self, cognitive: float, emotion: float) -> tuple[float, float]:
        return (
            (cognitive - self.cog_mean) / self.cog_std,
            (emotion - self.emo_mean) / self.emo_std,
        )

    def inverse_cog(self, z):
        return z * self.cog_std + self.cog_mean

    def inverse_emo(self, z):
        return z * self.emo_std + self.emo_mean


def fit_normalizer(labels: dict[str, SubjectLabel], train_subject_ids: Iterable[str]) -> TargetNormalizer:
    train_ids = [s for s in train_subject_ids if s in labels]
    cog = np.array([labels[s].cognitive_score for s in train_ids], dtype=float)
    emo = np.array([labels[s].emotion_score for s in train_ids], dtype=float)
    return TargetNormalizer(
        cog_mean=float(cog.mean()),
        cog_std=float(cog.std(ddof=0)),
        emo_mean=float(emo.mean()),
        emo_std=float(emo.std(ddof=0)),
    )
