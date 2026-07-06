from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

AU_COLUMNS: list[str] = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r",
    "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]


@dataclass
class VideoRecord:
    video_id: str
    subject_id: str
    au: np.ndarray            # (T, 17) float32
    timestamps: np.ndarray    # (T,) float32, 秒
    valid_ratio: float
    has_label: bool


def discover_videos(reports_root: Path, subjects: Sequence[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for sid in subjects:
        subj_dir = reports_root / sid
        if not subj_dir.is_dir():
            continue
        for csv_path in sorted(subj_dir.rglob("*.csv")):
            out.append((sid, csv_path))
    return out


def load_video_au(
    csv_path: Path,
    confidence_threshold: float = 0.8,
    enable_quality_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    n_total = len(df)
    if n_total == 0:
        empty = np.empty((0, len(AU_COLUMNS)), dtype=np.float32)
        return empty, np.empty((0,), dtype=np.float32), 0.0

    if enable_quality_filter and "confidence" in df.columns:
        df = df.loc[df["confidence"].astype(float) >= confidence_threshold].copy()

    sub = df[AU_COLUMNS].apply(pd.to_numeric, errors="coerce")
    ts = pd.to_numeric(df.get("timestamp", pd.Series([np.nan] * len(df))), errors="coerce")
    valid = sub.notna().all(axis=1) & ts.notna()
    sub = sub.loc[valid]
    ts = ts.loc[valid]

    au = sub.to_numpy(dtype=np.float32)
    timestamps = ts.to_numpy(dtype=np.float32)
    valid_ratio = float(len(au)) / n_total
    return au, timestamps, valid_ratio


def _video_id(subject_id: str, csv_path: Path) -> str:
    return f"{subject_id}/{csv_path.stem}"


def build_dataset(
    reports_root: Path,
    subjects: Sequence[str],
    labeled: set[str],
    confidence_threshold: float = 0.8,
    enable_quality_filter: bool = True,
) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    for sid, csv_path in discover_videos(reports_root, subjects):
        au, ts, vr = load_video_au(csv_path, confidence_threshold, enable_quality_filter)
        records.append(
            VideoRecord(
                video_id=_video_id(sid, csv_path),
                subject_id=sid,
                au=au,
                timestamps=ts,
                valid_ratio=vr,
                has_label=sid in labeled,
            )
        )
    return records
