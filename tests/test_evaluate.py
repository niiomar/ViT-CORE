import numpy as np

from evaluate import summarize_dataset_metrics


def test_summarize_dataset_metrics_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    preds = np.array([0, 0, 1, 1])

    metrics = summarize_dataset_metrics(labels, scores, preds)

    assert metrics["num_samples"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["tdr@0.1"] == 1.0


def test_summarize_dataset_metrics_degrades_on_single_class():
    labels = np.array([0, 0, 0])
    scores = np.array([0.1, 0.4, 0.9])
    preds = np.array([0, 0, 1])

    metrics = summarize_dataset_metrics(labels, scores, preds)

    assert metrics["auc"] == 0.0
    assert metrics["tdr@0.1"] == 0.0
