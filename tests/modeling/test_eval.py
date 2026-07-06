import numpy as np
from modeling.eval import regression_metrics, mean_baseline
from modeling.labels import get_labels

def test_perfect_prediction_metrics():
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert abs(m["pearson_r"] - 1.0) < 1e-9

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
