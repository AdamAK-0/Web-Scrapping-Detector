"""Training and evaluation helpers for baseline and online bot detection."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_NEGATIVE_LABEL, DEFAULT_POSITIVE_LABEL, DEFAULT_UNKNOWN_LABEL


@dataclass(slots=True)
class PrefixEvaluation:
    prefix_len: int
    metrics: dict[str, float]


@dataclass(slots=True)
class TrainingArtifacts:
    model_name: str
    feature_columns: list[str]
    model: Pipeline | RandomForestClassifier
    evaluations: list[PrefixEvaluation]
    detection_delay_summary: pd.DataFrame


@dataclass(slots=True)
class ModelBundle:
    model_name: str
    feature_columns: list[str]
    model: Pipeline | RandomForestClassifier
    positive_label: str = DEFAULT_POSITIVE_LABEL
    negative_label: str = DEFAULT_NEGATIVE_LABEL
    threshold: float = 0.5


def build_models(random_state: int = 42) -> dict[str, Pipeline | RandomForestClassifier]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        ),
    }


def train_and_evaluate_by_prefix(
    df: pd.DataFrame,
    prefixes: Iterable[int],
    random_state: int = 42,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
) -> dict[str, TrainingArtifacts]:
    labeled_df = df[df["label"] != DEFAULT_UNKNOWN_LABEL].copy()
    if labeled_df.empty:
        raise ValueError("No labeled rows available for training. Provide human/bot labels first.")

    feature_columns = [col for col in labeled_df.columns if col not in {"session_id", "prefix_len", "label"}]
    all_sessions = labeled_df["session_id"].drop_duplicates().tolist()
    if len(all_sessions) < 4:
        raise ValueError("At least four labeled sessions are needed for a train/test split")

    train_session_ids, test_session_ids = train_test_split(all_sessions, test_size=0.30, random_state=random_state)

    train_df = labeled_df[labeled_df["session_id"].isin(train_session_ids)].copy()
    test_df = labeled_df[labeled_df["session_id"].isin(test_session_ids)].copy()

    models = build_models(random_state=random_state)
    artifacts: dict[str, TrainingArtifacts] = {}

    for model_name, model in models.items():
        full_train = train_df.sort_values(["session_id", "prefix_len"]).drop_duplicates("session_id", keep="last")
        X_train = full_train[feature_columns]
        y_train = (full_train["label"] == positive_label).astype(int)
        model.fit(X_train, y_train)

        evaluations: list[PrefixEvaluation] = []
        per_prefix_predictions: list[pd.DataFrame] = []

        for prefix_len in sorted(prefixes):
            slice_df = test_df[test_df["prefix_len"] == prefix_len].copy()
            if slice_df.empty:
                continue
            X_test = slice_df[feature_columns]
            y_test = (slice_df["label"] == positive_label).astype(int)
            probabilities = model.predict_proba(X_test)[:, 1]
            predicted = (probabilities >= 0.5).astype(int)
            metrics = compute_metrics(y_test.to_numpy(), predicted, probabilities)
            evaluations.append(PrefixEvaluation(prefix_len=prefix_len, metrics=metrics))
            slice_df["bot_probability"] = probabilities
            slice_df["predicted_bot"] = predicted
            per_prefix_predictions.append(slice_df[["session_id", "prefix_len", "label", "bot_probability", "predicted_bot"]])

        detection_delay_summary = summarize_detection_delay(pd.concat(per_prefix_predictions, ignore_index=True), positive_label=positive_label)
        artifacts[model_name] = TrainingArtifacts(
            model_name=model_name,
            feature_columns=feature_columns,
            model=model,
            evaluations=evaluations,
            detection_delay_summary=detection_delay_summary,
        )

    return artifacts


def make_model_bundle(artifact: TrainingArtifacts, *, threshold: float = 0.5) -> ModelBundle:
    return ModelBundle(
        model_name=artifact.model_name,
        feature_columns=artifact.feature_columns,
        model=artifact.model,
        threshold=threshold,
    )


def save_model_bundle(bundle: ModelBundle, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_model_bundle(input_path: str | Path) -> ModelBundle:
    with Path(input_path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("Loaded object is not a ModelBundle")
    return bundle


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def summarize_detection_delay(predictions: pd.DataFrame, positive_label: str = DEFAULT_POSITIVE_LABEL) -> pd.DataFrame:
    rows = []
    predictions = predictions.sort_values(["session_id", "prefix_len"]).copy()

    for session_id, session_df in predictions.groupby("session_id"):
        true_label = str(session_df["label"].iloc[0])
        first_detected_prefix = None
        if true_label == positive_label:
            positive_prefixes = session_df.loc[session_df["predicted_bot"] == 1, "prefix_len"]
            if not positive_prefixes.empty:
                first_detected_prefix = int(positive_prefixes.iloc[0])
        rows.append(
            {
                "session_id": session_id,
                "label": true_label,
                "first_detected_prefix": first_detected_prefix,
                "detected": first_detected_prefix is not None,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    grouped_rows = []
    for label, group in summary.groupby("label"):
        grouped_rows.append(
            {
                "label": label,
                "num_sessions": int(len(group)),
                "detection_rate": float(group["detected"].mean()),
                "mean_first_detected_prefix": float(group["first_detected_prefix"].dropna().mean()) if group["detected"].any() else float("nan"),
            }
        )
    return pd.DataFrame(grouped_rows)
