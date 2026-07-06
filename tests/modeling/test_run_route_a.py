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
