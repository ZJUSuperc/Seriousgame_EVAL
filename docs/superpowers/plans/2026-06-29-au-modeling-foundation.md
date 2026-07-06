# AU 建模基础(数据管线 + LOSO 评估框架)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建带标签的 AU 数据集(从 `data/reports` 的 OpenFace CSV + 被试标签)和统一的 LOSO 评估框架,供后续路线 A/B/C 即插即用。

**Architecture:** 一个 `modeling/` 包,按职责拆分小文件:`labels`(标签)、`dataset`(AU 时序加载+清洗)、`splits`(LOSO 折)、`targets`(目标标准化)、`eval`(指标+基线)、`loso`(LOSO 驱动器)。模型通过一个 `predict_fn` 接口接入,被试级聚合后评估。

**Tech Stack:** Python 3.12、numpy、pandas、scipy、pytest(均已安装)。

## Global Constraints

- 所有路径基于项目根 `C:\Users\Administrator\Desktop\seriousgame_eval`。
- AU 列固定为 17 个 `_r` 强度列(见下);时间用 `timestamp`(秒)。
- 标签来自 12 被试(1901–1912),被试级(每视频继承其被试标签);`1/11/99` 无标签,可加载但 `has_label=False`。
- 评估**必须按被试**:LOSO,且指标在被试级聚合(每被试的多视频预测取均值后,与其真值比较)。
- 训练侧的任何统计(目标标准化均值/方差、基线均值)**只用该折训练被试**计算,杜绝泄露。
- 项目当前**不是 git 仓库**。执行前若想要"每任务提交"的节奏,先 `git init` 一次;否则把每个 Commit 步骤当作"保存检查点"。
- 已确认无 xgboost;后续 Route A 用 sklearn 的 Ridge/SVR/RandomForest。

---

## File Structure

```
seriousgame_eval/
  modeling/
    __init__.py          # 空包标记
    labels.py            # 被试标签(认知/情绪)+ CSV 读写
    dataset.py           # AU_COLUMNS、VideoRecord、发现视频、加载+清洗
    splits.py            # loso_folds
    targets.py           # TargetNormalizer(z-score,按折训练集拟合)
    eval.py              # regression_metrics、mean_baseline、permutation_test
    loso.py              # run_loso(驱动 predict_fn 跑 12 折 + 被试级聚合)
  data/modeling/
    labels.csv           # 由 labels.write_labels_csv 生成(Task 1)
  tests/modeling/
    __init__.py
    conftest.py          # 合成 OpenFace CSV / reports 树 fixture
    test_labels.py
    test_dataset.py
    test_splits.py
    test_targets.py
    test_eval.py
    test_loso.py
```

**关键接口契约(后续 Route A/B 实现这个):**
```python
PredictFn = Callable[
    [Sequence[VideoRecord], dict[str, SubjectLabel], Sequence[VideoRecord]],
    dict[str, tuple[float, float]],
]
# 输入:(训练视频records, 全部标签, 待测视频records)
# 输出:{video_id: (cognitive_pred, emotion_pred)}  原始尺度(未标准化)
```

---

## Task 1: 包脚手架 + 标签模块

**Files:**
- Create: `modeling/__init__.py`, `modeling/labels.py`
- Create: `tests/modeling/__init__.py`, `tests/modeling/test_labels.py`

**Interfaces:**
- Produces: `SUBJECT_LABELS: dict`, `SubjectLabel(subject_id, cognitive_score, emotion_score)`, `get_labels() -> dict[str, SubjectLabel]`, `labeled_subject_ids() -> list[str]`, `write_labels_csv(path)`, `load_labels_csv(path) -> dict`.

- [ ] **Step 1: 写失败测试**

`tests/modeling/test_labels.py`:
```python
from pathlib import Path
from modeling.labels import (
    get_labels, labeled_subject_ids, write_labels_csv, load_labels_csv, SubjectLabel,
)

def test_get_labels_has_twelve_subjects():
    labels = get_labels()
    assert len(labels) == 12
    assert labels["1901"].cognitive_score == 20
    assert labels["1901"].emotion_score == 1
    assert labels["1909"].cognitive_score == 29

def test_labeled_subject_ids_sorted():
    ids = labeled_subject_ids()
    assert ids == sorted(ids)
    assert "1901" in ids and "1912" in ids

def test_label_is_frozen_dataclass():
    lbl = SubjectLabel("1999", 10.0, 4.0)
    assert lbl.subject_id == "1999"

def test_csv_roundtrip(tmp_path: Path):
    p = tmp_path / "labels.csv"
    write_labels_csv(p)
    loaded = load_labels_csv(p)
    assert set(loaded) == set(get_labels())
    assert loaded["1907"].cognitive_score == 26.0
    assert loaded["1907"].emotion_score == 3.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_labels.py -v`
Expected: FAIL(模块不存在 / ImportError)

- [ ] **Step 3: 写实现**

`modeling/__init__.py`: (空文件)

`modeling/labels.py`:
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_labels.py -v`
Expected: 4 passed

- [ ] **Step 5: 生成 labels.csv 并提交**

Run:
```bash
python -c "from pathlib import Path; from modeling.labels import write_labels_csv; write_labels_csv(Path('data/modeling/labels.csv'))"
```
Commit(若已 git init):
```bash
git add modeling/__init__.py modeling/labels.py tests/modeling/ data/modeling/labels.csv
git commit -m "feat(modeling): add subject labels module and labels.csv"
```

---

## Task 2: 合成数据 fixture + AU 加载与清洗

**Files:**
- Create: `tests/modeling/conftest.py`
- Create: `modeling/dataset.py`
- Create: `tests/modeling/test_dataset.py`

**Interfaces:**
- Produces: `AU_COLUMNS`(17 个)、`VideoRecord(video_id, subject_id, au[T,17], timestamps[T], valid_ratio, has_label)`、`discover_videos(reports_root, subjects) -> list[(subject_id, Path)]`、`load_video_au(csv_path, confidence_threshold=0.8, enable_quality_filter=True) -> (au, timestamps, valid_ratio)`、`build_dataset(reports_root, subjects, labeled, ...) -> list[VideoRecord]`。
- Consumes: Task 1 的 `labeled_subject_ids`。

- [ ] **Step 1: 写 conftest fixture**

`tests/modeling/conftest.py`:
```python
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
```

`tests/modeling/test_dataset.py`:
```python
import numpy as np

from modeling.dataset import AU_COLUMNS, build_dataset, discover_videos, load_video_au


def test_au_columns_count():
    assert len(AU_COLUMNS) == 17
    assert "AU45_r" in AU_COLUMNS

def test_discover_videos(synthetic_reports):
    found = discover_videos(synthetic_reports, ["1901", "1902", "1903"])
    assert len(found) == 3
    assert all(p.suffix == ".csv" for _, p in found)

def test_load_video_au_shape_and_valid_ratio(synthetic_reports):
    csv_path = synthetic_reports / "1901" / "1901_20251010" / "1901_20251010.csv"
    au, ts, vr = load_video_au(csv_path)
    assert au.shape == (40, 17)
    assert ts.shape == (40,)
    assert vr == 1.0
    assert au.dtype == np.float32

def test_quality_filter_drops_low_confidence(tmp_path, make_openface_csv):
    p = tmp_path / "v.csv"
    make_openface_csv(p, n_frames=10, confidence=0.5, seed=1)  # 低于 0.8 阈值
    au, ts, vr = load_video_au(p, confidence_threshold=0.8, enable_quality_filter=True)
    assert vr == 0.0 and au.shape[0] == 0

def test_build_dataset_records(synthetic_reports):
    recs = build_dataset(synthetic_reports, ["1901", "1902"], labeled={"1901"})
    assert len(recs) == 2
    r0 = recs[0]
    assert r0.subject_id == "1901"
    assert r0.video_id == "1901/1901_20251010"
    assert r0.has_label is True
    assert recs[1].has_label is False  # 1902 不在 labeled
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_dataset.py -v`
Expected: FAIL(`modeling.dataset` 不存在)

- [ ] **Step 3: 写实现**

`modeling/dataset.py`:
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_dataset.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tests/modeling/conftest.py modeling/dataset.py tests/modeling/test_dataset.py
git commit -m "feat(modeling): add AU dataset loading and quality cleaning"
```

---

## Task 3: LOSO 折划分

**Files:**
- Create: `modeling/splits.py`
- Create: `tests/modeling/test_splits.py`

**Interfaces:**
- Produces: `loso_folds(subject_ids) -> list[(set[str], str)]`,每元素 = `(训练被试集合, 留出被试id)`。

- [ ] **Step 1: 写失败测试**

`tests/modeling/test_splits.py`:
```python
from modeling.splits import loso_folds

def test_loso_count_and_holdout():
    folds = loso_folds(["A", "B", "C"])
    assert len(folds) == 3
    held = [test for _, test in folds]
    assert sorted(held) == ["A", "B", "C"]

def test_loso_train_excludes_test():
    folds = loso_folds(["A", "B", "C"])
    for train_ids, test_id in folds:
        assert test_id not in train_ids
        assert len(train_ids) == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_splits.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`modeling/splits.py`:
```python
from __future__ import annotations

from typing import Sequence


def loso_folds(subject_ids: Sequence[str]) -> list[tuple[set[str], str]]:
    sids = sorted(subject_ids)
    return [({s for s in sids if s != test}, test) for test in sids]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_splits.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add modeling/splits.py tests/modeling/test_splits.py
git commit -m "feat(modeling): add leave-one-subject-out folds"
```

---

## Task 4: 目标标准化(按折训练集拟合)

**Files:**
- Create: `modeling/targets.py`
- Create: `tests/modeling/test_targets.py`

**Interfaces:**
- Consumes: `SubjectLabel`(Task 1)。
- Produces: `TargetNormalizer(cog_mean, cog_std, emo_mean, emo_std)`,方法 `transform(cog, emo) -> (z_cog, z_emo)`、`inverse_cog(z)`、`inverse_emo(z)`;`fit_normalizer(labels, train_subject_ids) -> TargetNormalizer`。

- [ ] **Step 1: 写失败测试**

`tests/modeling/test_targets.py`:
```python
import numpy as np
import pytest

from modeling.labels import get_labels
from modeling.targets import fit_normalizer

def test_fit_uses_only_train_subjects():
    labels = get_labels()
    train_ids = {"1901", "1902", "1903", "1904"}
    nz = fit_normalizer(labels, train_ids)
    cog = np.array([labels[s].cognitive_score for s in train_ids])
    emo = np.array([labels[s].emotion_score for s in train_ids])
    assert nz.cog_mean == pytest.approx(cog.mean())
    assert nz.cog_std == pytest.approx(cog.std(ddof=0))
    assert nz.emo_mean == pytest.approx(emo.mean())

def test_transform_then_inverse_roundtrip():
    labels = get_labels()
    nz = fit_normalizer(labels, {"1901", "1902", "1903", "1904"})
    z_c, z_e = nz.transform(20.0, 5.0)
    assert abs(nz.inverse_cog(z_c) - 20.0) < 1e-5
    assert abs(nz.inverse_emo(z_e) - 5.0) < 1e-5
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_targets.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`modeling/targets.py`:
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_targets.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add modeling/targets.py tests/modeling/test_targets.py
git commit -m "feat(modeling): add z-score target normalizer fit on train fold"
```

---

## Task 5: 回归指标 + 均值基线

**Files:**
- Create: `modeling/eval.py`
- Create: `tests/modeling/test_eval.py`

**Interfaces:**
- Produces: `regression_metrics(y_true, y_pred) -> dict`(键 `mae, rmse, r2, pearson_r, pearson_p, normalized_mae`)、`mean_baseline(labels, train_subject_ids) -> (cog, emo)`。

- [ ] **Step 1: 写失败测试**

`tests/modeling/test_eval.py`:
```python
import numpy as np
from modeling.eval import regression_metrics, mean_baseline
from modeling.labels import get_labels

def test_perfect_prediction_metrics():
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert m["pearson_r"] == 1.0

def test_metrics_values():
    yt = np.array([0.0, 1.0, 2.0])
    yp = np.array([0.0, 2.0, 1.0])  # 有误差
    m = regression_metrics(yt, yp)
    assert abs(m["mae"] - (0 + 1 + 1) / 3) < 1e-9
    assert m["normalized_mae"] > 0

def test_mean_baseline_uses_train_only():
    labels = get_labels()
    cog, emo = mean_baseline(labels, {"1901", "1902", "1903"})
    expected_cog = np.mean([labels[s].cognitive_score for s in ["1901", "1902", "1903"]])
    assert abs(cog - expected_cog) < 1e-9
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_eval.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`modeling/eval.py`:
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_eval.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add modeling/eval.py tests/modeling/test_eval.py
git commit -m "feat(modeling): add regression metrics and mean baseline"
```

---

## Task 6: LOSO 驱动器(被试级聚合)+ 端到端冒烟测试

**Files:**
- Create: `modeling/loso.py`
- Create: `tests/modeling/test_loso.py`

**Interfaces:**
- Consumes: `VideoRecord`(Task 2)、`SubjectLabel`/`get_labels`(Task 1)、`loso_folds`(Task 3)、`regression_metrics`(Task 5)。
- Produces: `PredictFn`(类型别名,见 File Structure 契约)、`run_loso(records, labels, predict_fn, return_predictions=False) -> dict`,返回 `{"cognitive": metrics, "emotion": metrics, "n_test_subjects": int, "predictions"?}`。

- [ ] **Step 1: 写失败测试**

`tests/modeling/test_loso.py`:
```python
import numpy as np
from modeling.dataset import build_dataset
from modeling.labels import get_labels
from modeling.loso import run_loso
from modeling.eval import mean_baseline


def _mean_predictor(train_records, labels, test_records):
    train_ids = {r.subject_id for r in train_records}
    cog, emo = mean_baseline(labels, train_ids)
    return {r.video_id: (cog, emo) for r in test_records}


def test_run_loso_smoke_with_synthetic(synthetic_reports):
    recs = build_dataset(synthetic_reports, ["1901", "1902", "1903"], labeled={"1901", "1902", "1903"})
    # 临时改标签,让合成被试也"有标签":这里直接用真实标签 dict 的结构造 3 个假标签
    labels = {
        "1901": get_labels()["1901"],
        "1902": get_labels()["1902"],
        "1903": get_labels()["1903"],
    }
    res = run_loso(recs, labels, _mean_predictor)
    assert "cognitive" in res and "emotion" in res
    assert res["n_test_subjects"] == 3
    for key in ("mae", "rmse", "r2", "pearson_r", "normalized_mae"):
        assert key in res["cognitive"]

def test_run_loso_returns_predictions(synthetic_reports):
    recs = build_dataset(synthetic_reports, ["1901", "1902"], labeled={"1901", "1902"})
    labels = {"1901": get_labels()["1901"], "1902": get_labels()["1902"]}
    res = run_loso(recs, labels, _mean_predictor, return_predictions=True)
    assert "predictions" in res
    # 每个被试聚合成一个预测点
    assert len(res["predictions"]) == 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_loso.py -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

`modeling/loso.py`:
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_loso.py -v`
Expected: 2 passed

- [ ] **Step 5: 跑全部测试**

Run: `python -m pytest tests/modeling/ -v`
Expected: 所有 Task 1–6 测试全部 passed(共 18 个)

- [ ] **Step 6: 真实数据冒烟(可选但推荐)**

确认真实数据能加载(只读、不训练):
```bash
python -c "
from pathlib import Path
from modeling.dataset import build_dataset
from modeling.labels import labeled_subject_ids, get_labels
recs = build_dataset(Path('data/reports'), labeled_subject_ids(), set(labeled_subject_ids()))
print('videos:', len(recs))
print('subjects:', sorted({r.subject_id for r in recs}))
print('sample frames:', recs[0].au.shape, 'valid_ratio:', round(recs[0].valid_ratio, 3))
"
```
Expected: videos ≈ 72,subjects = 1901..1912,sample frames 为 `(T, 17)`。

- [ ] **Step 7: Commit**

```bash
git add modeling/loso.py tests/modeling/test_loso.py
git commit -m "feat(modeling): add LOSO driver with subject-level aggregation"
```

---

## Self-Review(已自查)

- **Spec 覆盖**:本计划覆盖 spec 第 2 节(数据:标签/分布/清洗/LOSO/标准化)、第 7 节(评估框架:指标/基线/被试级聚合)。第 4/5/6 节(路线 A/B/C)留给 Plan 2/3/4。置换检验(permutation test)spec 第 7 节提到——本 Plan 暂未实现,推迟到 Plan 2 引入第一个真实模型时一并加(基线阶段无模型可验,优先级低)。
- **占位扫描**:无 TBD/TODO;每步都给了真实代码与命令。
- **类型一致**:`SubjectLabel`、`VideoRecord`、`PredictFn`、`TargetNormalizer` 跨任务命名一致;`run_loso` 返回键与测试断言一致。
- **歧义钉死**:评估在被试级聚合(每被试多视频预测取均值)、训练侧统计只用本折训练被试——两处潜在泄露已在代码与 Global Constraints 明确。
