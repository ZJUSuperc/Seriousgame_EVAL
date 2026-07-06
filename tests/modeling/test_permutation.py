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
