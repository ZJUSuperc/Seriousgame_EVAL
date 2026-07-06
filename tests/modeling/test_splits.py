from modeling.splits import loso_folds

def test_loso_count_and_holdout():
    folds = loso_folds(["A", "B", "C"])
    assert len(folds) == 3
    held = [test for _, test in folds]
    assert sorted(held) == ["A", "B", "C"]

def test_loso_train_excludes_test():
    folds = loso_folds(["A", "B", "C"])
    for train_ids, test_id in folds:
        assert test_id not in train_ids
        assert len(train_ids) == 2
