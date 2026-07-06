# 路线 A(特征工程 + 浅层多输出回归)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用可解释的 AU 时序特征 + 浅层回归模型(Ridge/SVR/RandomForest),在 Plan 1 的 LOSO 框架上产出认知/情绪分数的第一个真实基线。

**Architecture:** 新增 `modeling/features.py`(逐视频 AU 统计+时序动态特征)、`modeling/route_a.py`(把 sklearn 回归器封装成 Plan 1 的 `PredictFn`,认知/情绪各训一个)、给 `modeling/loso.py` 加 `permutation_test`(置换检验)、`scripts/run_route_a.py`(跑三模型 LOSO + 置换 + 特征重要性,落 JSON)。每个模型独立训练 cognitive 与 emotion 两个目标(避免尺度均衡问题)。

**Tech Stack:** Python 3.12、numpy、scikit-learn 1.8.0、pytest(均已装)。

## Global Constraints

- 复用 Plan 1 的接口:`VideoRecord`、`SubjectLabel`/`get_labels`、`labeled_subject_ids`、`run_loso`、`PredictFn` 契约 `(train_records, labels, test_records) -> {video_id:(cog,emo)}` 原始尺度。
- 训练**只用有标签训练被试**的视频(`has_label and subject_id in labels`);评估由 `run_loso` 做被试级聚合(已防泄露)。
- 认知/情绪**各训独立模型**(单输出),不做 z-score 均衡(浅层单输出模型不需要)。
- 路径相对项目根 `C:/Users/Administrator/Desktop/seriousgame_eval`。
- **非 git 仓库**:跳过所有 `git add/commit`,当作保存检查点。
- 已确认无 xgboost;用 sklearn 的 Ridge/SVR/RandomForestRegressor。
- 语义级 7 类异常事件特征**本期不做**(会耦合 `app/`,且参考论文 FDASSNN 直接用 AU 强度);以 AU 统计+时序动态为主特征集,论文对齐、解耦。记录为有意范围决定。

---

## File Structure

```
seriousgame_eval/
  modeling/
    features.py        # AU_COLUMNS 命名;extract_features(record)->vec(105维);feature_names()
    route_a.py         # make_sklearn_predictor(factory);ridge/svr/rf 工厂;MODEL_FACTORIES
    loso.py            # [修改] 新增 permutation_test(records,labels,predict_fn,...)
  scripts/
    run_route_a.py     # 跑三模型 LOSO + 置换 + RF 特征重要性,写 data/modeling/route_a_results.json
  data/modeling/
    route_a_results.json   # 生成
  tests/modeling/
    test_features.py
    test_route_a.py
    test_permutation.py
```

**契约**:`PredictFn` 由 `route_a` 实现并交给 `run_loso`/`permutation_test`。特征维度 = 17×5(均值/标准差/激活率/中位数/p90)+ 17(delta)+ 3(全局)= **105**。

---

## Task 1: 特征提取模块

**Files:**
- Create: `modeling/features.py`
- Create: `tests/modeling/test_features.py`

**Interfaces:**
- Consumes: `VideoRecord`(Plan 1)、`AU_COLUMNS`(Plan 1 `dataset.py`)。
- Produces: `AU_THRESHOLD=0.5`、`extract_features(record: VideoRecord) -> np.ndarray`(105维 float32)、`feature_names() -> list[str]`(105)。

- [ ] **Step 1: 写失败测试**

`tests/modeling/test_features.py`:
```python
import numpy as np
from modeling.dataset import VideoRecord, AU_COLUMNS
from modeling.features import extract_features, feature_names, AU_THRESHOLD


def _record(au, valid_ratio=1.0):
    return VideoRecord(video_id="x/1", subject_id="x", au=au.astype(np.float32),
                       timestamps=np.arange(au.shape[0], dtype=np.float32),
                       valid_ratio=valid_ratio, has_label=True)


def test_feature_dim_matches_names():
    names = feature_names()
    assert len(names) == 105
    au = np.random.default_rng(0).uniform(0, 2, size=(50, len(AU_COLUMNS)))
    vec = extract_features(_record(au))
    assert vec.shape == (105,)
    assert vec.dtype == np.float32


def test_features_deterministic_and_finite():
    au = np.random.default_rng(1).uniform(0, 2, size=(40, len(AU_COLUMNS)))
    r = _record(au)
    a, b = extract_features(r), extract_features(r)
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(a))


def test_empty_record_returns_zeros_no_crash():
    r = _record(np.zeros((0, len(AU_COLUMNS))))
    vec = extract_features(r)
    assert vec.shape == (105,)
    assert np.all(vec == 0.0)


def test_activation_ratio_feature():
    # 第一个 AU: 一半帧 > 阈值
    n = len(AU_COLUMNS)
    au = np.zeros((10, n))
    au[:5, 0] = AU_THRESHOLD + 0.1  # 5/10 帧 active
    vec = extract_features(_record(au))
    names = feature_names()
    idx = names.index("AU01_act")
    assert abs(vec[idx] - 0.5) < 1e-6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_features.py -v` → FAIL(模块不存在)

- [ ] **Step 3: 写实现**

`modeling/features.py`:
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_features.py -v` → 4 passed

- [ ] **Step 5: 跑全套件确认无回归**

Run: `python -m pytest tests/modeling/ -q` → 全部通过(Plan1 的 18 + 本任务 4 = 22)

- [ ] **Step 6: Commit(无 git → 保存检查点)**

---

## Task 2: 路线 A 预测器(封装 sklearn 回归器为 PredictFn)

**Files:**
- Create: `modeling/route_a.py`
- Create: `tests/modeling/test_route_a.py`

**Interfaces:**
- Consumes: `extract_features`(Task 1)、`PredictFn`(Plan 1 `loso.py`)、`VideoRecord`/`SubjectLabel`。
- Produces: `make_sklearn_predictor(factory) -> PredictFn`、`make_ridge_predictor/make_svr_predictor/make_rf_predictor`、`MODEL_FACTORIES: dict`、`_design_matrix(records, labels) -> (X, y_cog, y_emo)`。

- [ ] **Step 1: 写失败测试**

`tests/modeling/test_route_a.py`:
```python
import numpy as np
from modeling.dataset import VideoRecord, AU_COLUMNS
from modeling.labels import SubjectLabel
from modeling.route_a import make_ridge_predictor, make_rf_predictor, _design_matrix


def _rec(vid, sid, seed):
    au = np.random.default_rng(seed).uniform(0, 2, size=(30, len(AU_COLUMNS))).astype(np.float32)
    return VideoRecord(video_id=vid, subject_id=sid, au=au,
                       timestamps=np.arange(30, dtype=np.float32), valid_ratio=1.0, has_label=True)


def test_design_matrix_shapes():
    recs = [_rec("a/1", "a", 0), _rec("a/2", "a", 1), _rec("b/1", "b", 2)]
    labels = {"a": SubjectLabel("a", 20.0, 3.0), "b": SubjectLabel("b", 10.0, 7.0)}
    X, yc, ye = _design_matrix(recs, labels)
    assert X.shape == (3, 105) and yc.shape == (3,) and ye.shape == (3,)
    assert yc[0] == 20.0 and yc[2] == 10.0  # 视频继承被试标签


def test_ridge_predictor_returns_all_test_preds_finite():
    train = [_rec(f"a/{i}", "a", i) for i in range(5)] + [_rec(f"b/{i}", "b", 100 + i) for i in range(5)]
    test = [_rec("c/1", "c", 200), _rec("c/2", "c", 201)]
    labels = {"a": SubjectLabel("a", 20.0, 3.0), "b": SubjectLabel("b", 10.0, 7.0)}
    preds = make_ridge_predictor()(train, labels, test)
    assert set(preds) == {"c/1", "c/2"}
    for v in preds.values():
        assert len(v) == 2 and all(np.isfinite(v))


def test_rf_predictor_smoke():
    train = [_rec(f"a/{i}", "a", i) for i in range(5)] + [_rec(f"b/{i}", "b", 100 + i) for i in range(5)]
    test = [_rec("c/1", "c", 200)]
    labels = {"a": SubjectLabel("a", 20.0, 3.0), "b": SubjectLabel("b", 10.0, 7.0)}
    preds = make_rf_predictor(n_estimators=10)(train, labels, test)
    assert "c/1" in preds
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_route_a.py -v` → FAIL

- [ ] **Step 3: 写实现**

`modeling/route_a.py`:
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_route_a.py -v` → 3 passed

- [ ] **Step 5: 全套件无回归**

Run: `python -m pytest tests/modeling/ -q` → 25 passed

- [ ] **Step 6: Commit(无 git → 检查点)**

---

## Task 3: 置换检验(permutation_test)

**Files:**
- Modify: `modeling/loso.py`(在文件末尾追加 `permutation_test`)
- Create: `tests/modeling/test_permutation.py`

**Interfaces:**
- Produces: `permutation_test(records, labels, predict_fn, n_permutations=200, seed=0) -> {"observed_pearson_r": {cog,emo}, "p_value": {cog,emo}, "n_permutations": int}`。
- 零假设:打乱被试→分数映射(保留 AU 数据与折结构),重跑 `run_loso`,统计置换 r ≥ 观测 r 的比例。

- [ ] **Step 1: 写失败测试**

`tests/modeling/test_permutation.py`:
```python
import numpy as np
from modeling.dataset import VideoRecord, AU_COLUMNS
from modeling.labels import SubjectLabel
from modeling.loso import permutation_test, run_loso
from modeling.eval import mean_baseline


def _rec(vid, sid, seed):
    au = np.random.default_rng(seed).uniform(0, 2, size=(30, len(AU_COLUMNS))).astype(np.float32)
    return VideoRecord(video_id=vid, subject_id=sid, au=au,
                       timestamps=np.arange(30, dtype=np.float32), valid_ratio=1.0, has_label=True)


def _mean_predictor(train_records, labels, test_records):
    cog, emo = mean_baseline(labels, {r.subject_id for r in train_records})
    return {r.video_id: (cog, emo) for r in test_records}


def test_permutation_test_structure():
    recs = [_rec(f"s{i}/1", f"s{i}", i) for i in range(5)]
    labels = {f"s{i}": SubjectLabel(f"s{i}", float(i * 5), float(i)) for i in range(5)}
    res = permutation_test(recs, labels, _mean_predictor, n_permutations=5, seed=0)
    assert set(res) == {"observed_pearson_r", "p_value", "n_permutations"}
    assert res["n_permutations"] == 5
    for t in ("cognitive", "emotion"):
        assert t in res["observed_pearson_r"] and t in res["p_value"]
        p = res["p_value"][t]
        assert (np.isnan(p) or 0.0 <= p <= 1.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_permutation.py -v` → FAIL(`permutation_test` 不存在)

- [ ] **Step 3: 写实现(追加到 `modeling/loso.py` 末尾)**

在 `modeling/loso.py` 文件末尾追加(不要改动已有 `run_loso`):
```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_permutation.py -v` → 1 passed

- [ ] **Step 5: 全套件无回归**

Run: `python -m pytest tests/modeling/ -q` → 26 passed

- [ ] **Step 6: Commit(无 git → 检查点)**

---

## Task 4: 运行脚本(三模型 LOSO + 置换 + 特征重要性)

**Files:**
- Create: `scripts/run_route_a.py`
- Create: `tests/modeling/test_run_route_a.py`(轻量冒烟:用合成小数据跑通)

**Interfaces:**
- Produces: `main(reports_root, out_path, n_perm=200)`;写 JSON 结构 `{model: {metrics:{cognitive,emotion}, permutation:{...}}, feature_importance:{name:weight,...}(RF 全数据拟合), n_videos, n_subjects}`。

- [ ] **Step 1: 写失败测试(合成小数据冒烟)**

`tests/modeling/test_run_route_a.py`:
```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from modeling.dataset import AU_COLUMNS
from scripts.run_route_a import main


def _write_csv(path, n_frames, seed):
    rng = np.random.default_rng(seed)
    cols = {"frame": np.arange(n_frames), "timestamp": np.arange(n_frames) / 30.0,
            "confidence": np.full(n_frames, 0.95)}
    for au in AU_COLUMNS:
        cols[au] = rng.uniform(0, 2, size=n_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(path, index=False)


def test_main_runs_on_synthetic_and_writes_json(tmp_path):
    reports = tmp_path / "reports"
    # 仅 2 个被试,每个 2 视频(够跑通;真实 LOSO 在主程序用 12 被试)
    for sid in ["1901", "1902"]:
        for k in range(2):
            _write_csv(reports / sid / f"{sid}_v{k}" / f"{sid}_v{k}.csv", n_frames=40, seed=int(sid) + k)
    out = tmp_path / "out.json"
    main(reports_root=reports, out_path=out, n_perm=3)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "ridge" in data and "rf" in data and "svr" in data
    assert "feature_importance" in data and len(data["feature_importance"]) == 105
    for m in ("ridge", "rf", "svr"):
        assert "cognitive" in data[m]["metrics"] and "mae" in data[m]["metrics"]["cognitive"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/modeling/test_run_route_a.py -v` → FAIL(模块不存在)

- [ ] **Step 3: 写实现**

`scripts/run_route_a.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from modeling.dataset import build_dataset
from modeling.features import extract_features, feature_names
from modeling.labels import get_labels, labeled_subject_ids
from modeling.loso import permutation_test, run_loso
from modeling.route_a import MODEL_FACTORIES, _design_matrix

ROOT = Path(__file__).resolve().parents[1]


def _feature_importance(records, labels) -> dict[str, float]:
    X, y_cog, _ = _design_matrix(records, labels)
    if X.shape[0] < 2:
        return {n: 0.0 for n in feature_names()}
    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y_cog)
    return {n: float(w) for n, w in zip(feature_names(), rf.feature_importances_)}


def main(reports_root: Path, out_path: Path, n_perm: int = 200) -> dict:
    subjects = labeled_subject_ids()
    records = build_dataset(reports_root, subjects, set(subjects))
    labels = get_labels()

    results: dict = {}
    for name, factory in MODEL_FACTORIES.items():
        fn = factory()
        metrics = run_loso(records, labels, fn)
        perm = permutation_test(records, labels, fn, n_permutations=n_perm, seed=0)
        results[name] = {
            "metrics": {t: metrics[t] for t in ("cognitive", "emotion")},
            "permutation": perm,
        }

    results["feature_importance"] = _feature_importance(records, labels)
    results["n_videos"] = len(records)
    results["n_subjects"] = len(subjects)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    res = main(REPORTS_ROOT := ROOT / "data" / "reports", ROOT / "data" / "modeling" / "route_a_results.json")
    for m in ("ridge", "svr", "rf"):
        c = res[m]["metrics"]["cognitive"]
        e = res[m]["metrics"]["emotion"]
        print(f"{m}: cog r={c['pearson_r']:.3f} R2={c['r2']:.3f} MAE={c['mae']:.2f} | "
              f"emo r={e['pearson_r']:.3f} R2={e['r2']:.3f} MAE={e['mae']:.2f}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/modeling/test_run_route_a.py -v` → 1 passed

- [ ] **Step 5: 全套件无回归**

Run: `python -m pytest tests/modeling/ -q` → 27 passed

- [ ] **Step 6: Commit(无 git → 检查点)**

---

## Task 5: 真实数据跑批 + 结果落盘

**Files:**
- 无新代码;运行 `scripts/run_route_a.py` 产出 `data/modeling/route_a_results.json`。

- [ ] **Step 1: 在真实数据上跑(12 被试 / 72 视频)**

Run:
```bash
python scripts/run_route_a.py
```
Expected:打印 ridge/svr/rf 的 cog/emo 的 pearson_r / R² / MAE;写出 `data/modeling/route_a_results.json`。耗时取决于置换次数(默认 200 × 3 模型 × 12 折)。若太慢可临时把 `__main__` 里改小 n_perm,但**交付用 n_perm=200**。

- [ ] **Step 2: 校验产出**

Run:
```bash
python -c "import json; d=json.load(open('data/modeling/route_a_results.json',encoding='utf-8')); print('models:', [k for k in ('ridge','svr','rf') if k in d]); print('n_videos', d['n_videos'], 'n_subjects', d['n_subjects']); import pprint; pprint.pprint({m:{t:d[m]['metrics'][t] for t in ('cognitive','emotion')} for m in ('ridge','svr','rf')})"
```
Expected:`n_videos=72, n_subjects=12`;三模型各有 cog/emo 的 mae/rmse/r2/pearson_r/normalized_mae;feature_importance 共 105 项。

- [ ] **Step 3: 记录关键结果到 ledger**

把三模型 cog/emo 的 pearson_r、R²、MAE 与置换 p 值摘抄进 `.superpowers/sdd/progress.md` 的 Plan 2 小节。

- [ ] **Step 4: Commit(无 git → 检查点)**

---

## Self-Review(已自查)

- **Spec 覆盖**:Plan 2 覆盖 spec 第 4 节(路线 A:特征+浅层模型+LOSO)与第 7 节置换检验(从 Plan 1 推迟到此处)。第 5/6 节(路线 B/C)留给 Plan 3/4。
- **范围决定(记录)**:7 类语义异常事件特征**本期不做**(耦合 `app/`、且参考论文直接用 AU 强度),以 AU 统计+时序动态(105 维)为主,论文对齐、解耦。
- **占位扫描**:无 TBD;每步含真实代码与命令。
- **类型一致**:`PredictFn`、`extract_features`、`_design_matrix`、`permutation_test` 跨任务签名一致;`run_loso`/`mean_baseline`(Plan 1)复用无误。
- **歧义钉死**:认知/情绪各训独立单输出模型(不做 z-score);训练只用有标签训练被试视频;置换检验打乱被试→分数映射(保留 AU 与折结构)。
