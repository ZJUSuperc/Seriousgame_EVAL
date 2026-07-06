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
