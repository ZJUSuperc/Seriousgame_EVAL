from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

SUBJECT_LABELS: dict[str, dict[str, int]] = {
    "1901": {"cognitive_score": 20, "emotion_score": 1},
    "1902": {"cognitive_score": 15, "emotion_score": 9},
    "1903": {"cognitive_score": 18, "emotion_score": 5},
    "1904": {"cognitive_score": 21, "emotion_score": 5},
    "1905": {"cognitive_score": 6, "emotion_score": 6},
    "1906": {"cognitive_score": 20, "emotion_score": 3},
    "1907": {"cognitive_score": 26, "emotion_score": 3},
    "1908": {"cognitive_score": 12, "emotion_score": 5},
    "1909": {"cognitive_score": 29, "emotion_score": 0},
    "1910": {"cognitive_score": 28, "emotion_score": 0},
    "1911": {"cognitive_score": 25, "emotion_score": 3},
    "1912": {"cognitive_score": 27, "emotion_score": 9},
}


@dataclass(frozen=True)
class SubjectLabel:
    subject_id: str
    cognitive_score: float
    emotion_score: float


def get_labels() -> dict[str, SubjectLabel]:
    return {
        sid: SubjectLabel(sid, v["cognitive_score"], v["emotion_score"])
        for sid, v in SUBJECT_LABELS.items()
    }


def labeled_subject_ids() -> list[str]:
    return sorted(SUBJECT_LABELS.keys())


def write_labels_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "cognitive_score", "emotion_score"])
        for sid in labeled_subject_ids():
            lbl = get_labels()[sid]
            w.writerow([sid, lbl.cognitive_score, lbl.emotion_score])


def load_labels_csv(path: Path) -> dict[str, SubjectLabel]:
    out: dict[str, SubjectLabel] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["subject_id"]] = SubjectLabel(
                row["subject_id"],
                float(row["cognitive_score"]),
                float(row["emotion_score"]),
            )
    return out
