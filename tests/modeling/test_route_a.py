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
