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
