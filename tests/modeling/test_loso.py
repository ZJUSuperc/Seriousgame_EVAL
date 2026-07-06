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
