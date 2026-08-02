from metrics import compute_auc, compute_tdr


def test_compute_auc_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    assert compute_auc(y_true, y_scores) == 1.0


def test_compute_auc_single_class_returns_zero():
    # sklearn's roc_auc_score raises ValueError when only one class is present.
    assert compute_auc([1, 1, 1], [0.1, 0.5, 0.9]) == 0.0


def test_compute_tdr_perfect_separation():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_scores = [0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95]
    assert compute_tdr(y_true, y_scores, fpr_threshold=0.1) == 1.0


def test_compute_tdr_single_class_returns_zero():
    # Regression test: roc_curve raises ValueError with only one class present,
    # compute_tdr must degrade to 0.0 the same way compute_auc does, rather
    # than crashing a training/evaluation run on a skewed split.
    assert compute_tdr([0, 0, 0], [0.1, 0.4, 0.9], fpr_threshold=0.1) == 0.0
