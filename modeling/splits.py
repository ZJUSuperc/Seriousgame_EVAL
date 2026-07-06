from __future__ import annotations

from typing import Sequence


def loso_folds(subject_ids: Sequence[str]) -> list[tuple[set[str], str]]:
    sids = sorted(subject_ids)
    return [({s for s in sids if s != test}, test) for test in sids]
