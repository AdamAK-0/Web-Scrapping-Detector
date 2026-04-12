"""Research-oriented evaluation utilities.

Includes threshold tuning, bootstrap confidence intervals, and report helpers
for early bot-detection experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(slots=True)
class ConfidenceInterval:
    metric: str
    point_estimate: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, candidate_thresholds: Iterable[float] | None = None) -> float:
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def bootstrap_metric_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
    metric_name: str,
    *,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> ConfidenceInterval:
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    y_pred_arr = (y_score_arr >= threshold).astype(int)

    metric_functions: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
        "accuracy": lambda yt, yp, ys: float(accuracy_score(yt, yp)),
        "precision": lambda yt, yp, ys: float(precision_score(yt, yp, zero_division=0)),
        "recall": lambda yt, yp, ys: float(recall_score(yt, yp, zero_division=0)),
        "f1": lambda yt, yp, ys: float(f1_score(yt, yp, zero_division=0)),
        "roc_auc": _safe_roc_auc,
        "pr_auc": _safe_pr_auc,
    }
    if metric_name not in metric_functions:
        raise ValueError(f"Unsupported metric for CI: {metric_name}")

    point_estimate = metric_functions[metric_name](y_true_arr, y_pred_arr, y_score_arr)
    rng = np.random.default_rng(random_state)
    samples: list[float] = []

    if len(y_true_arr) == 0:
        return ConfidenceInterval(metric=metric_name, point_estimate=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n_bootstrap=0)

    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(y_true_arr), len(y_true_arr))
        yt = y_true_arr[idx]
        ys = y_score_arr[idx]
        yp = (ys >= threshold).astype(int)
        value = metric_functions[metric_name](yt, yp, ys)
        if not np.isnan(value):
            samples.append(float(value))

    if not samples:
        return ConfidenceInterval(metric=metric_name, point_estimate=point_estimate, ci_low=float("nan"), ci_high=float("nan"), n_bootstrap=0)

    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return ConfidenceInterval(metric=metric_name, point_estimate=point_estimate, ci_low=lower, ci_high=upper, n_bootstrap=len(samples))



def attach_metric_confidence_intervals(
    prefix_predictions: pd.DataFrame,
    *,
    threshold: float,
    metric_names: Iterable[str] = ("f1", "roc_auc", "pr_auc"),
    n_bootstrap: int = 500,
    positive_label: str = "bot",
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    working = prefix_predictions.copy()
    working["y_true"] = (working["label"] == positive_label).astype(int)
    for prefix_len, group in working.groupby("prefix_len"):
        y_true = group["y_true"].to_numpy()
        y_score = group["bot_probability"].to_numpy()
        for metric_name in metric_names:
            ci = bootstrap_metric_ci(y_true, y_score, threshold, metric_name, n_bootstrap=n_bootstrap)
            rows.append(
                {
                    "prefix_len": int(prefix_len),
                    "metric": metric_name,
                    "point_estimate": ci.point_estimate,
                    "ci_low": ci.ci_low,
                    "ci_high": ci.ci_high,
                    "n_bootstrap": ci.n_bootstrap,
                }
            )
    return pd.DataFrame(rows)



def _safe_roc_auc(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))



def _safe_pr_auc(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))
