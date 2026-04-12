import numpy as np

from wsd.evaluation import bootstrap_metric_ci, compute_binary_metrics, tune_threshold


def test_compute_binary_metrics_contains_pr_auc() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred = (y_score >= 0.5).astype(int)
    metrics = compute_binary_metrics(y_true, y_pred, y_score)
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_tune_threshold_returns_reasonable_value() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.9])
    threshold = tune_threshold(y_true, y_score)
    assert 0.4 <= threshold <= 0.6


def test_bootstrap_metric_ci_runs() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_score = np.array([0.2, 0.9, 0.1, 0.8, 0.7, 0.3])
    ci = bootstrap_metric_ci(y_true, y_score, threshold=0.5, metric_name="f1", n_bootstrap=50)
    assert 0.0 <= ci.point_estimate <= 1.0
    assert 0.0 <= ci.ci_low <= 1.0
    assert 0.0 <= ci.ci_high <= 1.0
