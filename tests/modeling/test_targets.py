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
