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
