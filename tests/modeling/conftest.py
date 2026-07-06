from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling.dataset import AU_COLUMNS  # 单一真值来源,避免重复定义


@pytest.fixture
def make_openface_csv():
    """返回一个写入合成 OpenFace CSV 的函数(供需要单独造 CSV 的测试用)。"""
    def _make(path: Path, n_frames: int = 50, confidence: float = 0.95, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        cols: dict = {
            "frame": np.arange(n_frames),
            " timestamp": np.arange(n_frames) / 30.0,   # 故意带前导空格,测试 strip
            "confidence": np.full(n_frames, confidence),
            "face_id": np.zeros(n_frames, dtype=int),
        }
        for au in AU_COLUMNS:
            cols[au] = rng.uniform(0.0, 2.0, size=n_frames)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cols).to_csv(path, index=False)
    return _make


@pytest.fixture
def synthetic_reports(tmp_path: Path, make_openface_csv) -> Path:
    reports = tmp_path / "reports"
    for sid in ["1901", "1902", "1903"]:
        make_openface_csv(
            reports / sid / f"{sid}_20251010" / f"{sid}_20251010.csv",
            n_frames=40, seed=int(sid),
        )
    return reports
