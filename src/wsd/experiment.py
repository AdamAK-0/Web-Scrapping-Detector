"""Research experiment runner with baselines, leakage audit, and report outputs."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import DEFAULT_BOT_USER_AGENT_PATTERNS, DEFAULT_POSITIVE_LABEL, DEFAULT_UNKNOWN_LABEL
from .evaluation import attach_metric_confidence_intervals, compute_binary_metrics, tune_threshold, tune_thresholds_by_prefix
from .features import extract_features_for_events, extract_prefix_feature_rows, prefix_rows_to_dataframe
from .graph_builder import build_graph_from_csv, infer_category_from_path
from .modeling import build_models, make_model_bundle, save_model_bundle, summarize_detection_delay
from .sessionizer import load_sessions_from_csv, summarize_sessions
from .types import Session

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional at runtime
    plt = None


FEATURE_GROUPS: dict[str, list[str]] = {
    "session": [
        "session_length_so_far",
    ],
    "graph": [
        "unique_nodes",
        "unique_node_ratio",
        "revisit_ratio",
        "repeated_path_ratio",
        "repeated_transition_ratio",
        "backtrack_count",
        "self_loop_ratio",
        "mean_depth",
        "leaf_visit_ratio",
        "out_degree_mean",
        "mean_hop_distance",
        "far_jump_ratio",
        "category_switch_rate",
    ],
    "timing": [
        "mean_delta_t",
        "std_delta_t",
        "burstiness",
        "low_latency_ratio",
        "error_rate",
        "user_agent_switch_rate",
    ],
    "entropy": [
        "transition_entropy",
        "normalized_transition_entropy",
        "category_entropy",
        "normalized_category_entropy",
        "node_entropy",
        "normalized_node_entropy",
        "navigation_entropy_score",
    ],
}

ABLATIONS: dict[str, list[str]] = {
    "all_features": ["session", "graph", "timing", "entropy"],
    "graph_plus_entropy": ["session", "graph", "entropy"],
    "graph_only": ["session", "graph"],
    "entropy_only": ["session", "entropy"],
    "timing_only": ["session", "timing"],
}

STATIC_NUMERIC_FEATURES = [
    "has_suspect_user_agent",
    "ua_length",
    "ua_slash_count",
    "ua_browser_like",
    "first_path_depth",
    "first_status_code",
    "first_body_bytes_sent",
]

STATIC_CATEGORICAL_FEATURES = [
    "ua_family",
    "first_category",
    "first_method",
]

FULL_SESSION_SENTINEL_PREFIX = -1


@dataclass(slots=True)
class ExperimentArtifact:
    artifact_id: str
    model_name: str
    ablation_name: str
    protocol: str
    threshold: float
    threshold_strategy: str
    metrics_by_prefix: pd.DataFrame
    predictions: pd.DataFrame
    confidence_intervals: pd.DataFrame
    detection_delay_summary: pd.DataFrame
    notes: str = ""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run research-grade ablation experiments for the scraper detector")
    parser.add_argument("--data-dir", default="data/nginx_demo_prepared", help="Directory containing graph and request CSVs")
    parser.add_argument("--prefixes", nargs="*", type=int, default=[3, 5, 10, 15, 20], help="Prefix lengths to evaluate")
    parser.add_argument("--output-dir", default=None, help="Output directory for experiment artifacts; defaults to <data-dir>/experiments")
    parser.add_argument("--n-bootstrap", type=int, default=300, help="Bootstrap samples for confidence intervals")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--save-models", action="store_true", help="Save calibrated model bundles for each experiment variant")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph_from_csv(data_dir / "graph_edges.csv", data_dir / "graph_categories.csv")
    sessions = load_sessions_from_csv(data_dir / "requests.csv")
    feature_df = _load_or_build_feature_df(data_dir, sessions=sessions, graph=graph, prefixes=args.prefixes)
    full_session_df = _load_or_build_full_session_feature_df(data_dir, sessions=sessions, graph=graph)
    session_metadata = _build_session_metadata_from_sessions(sessions)
    session_metadata.to_csv(output_dir / "session_metadata.csv", index=False)
    artifacts = run_experiments(
        feature_df,
        prefixes=args.prefixes,
        output_dir=output_dir,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        save_models=args.save_models,
        session_metadata=session_metadata,
        full_session_df=full_session_df,
    )

    leaderboard = build_leaderboard(artifacts)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    split_summary = _read_if_exists(output_dir / "split_summary.csv")
    leakage_audit = _read_if_exists(output_dir / "leakage_audit.csv")
    _write_markdown_summary(output_dir / "summary.md", leaderboard, artifacts, split_summary=split_summary, leakage_audit=leakage_audit)
    _maybe_plot(artifacts, output_dir, feature_df=full_session_df)
    print(f"Experiment outputs written to: {output_dir.resolve()}")
    print(leaderboard.to_string(index=False))



def _load_or_build_feature_df(data_dir: Path, *, sessions: list[Session], graph, prefixes: Iterable[int]) -> pd.DataFrame:
    feature_path = data_dir / "prefix_features.csv"
    if feature_path.exists():
        return pd.read_csv(feature_path)
    feature_rows = extract_prefix_feature_rows(sessions, graph, prefixes=prefixes)
    feature_df = prefix_rows_to_dataframe(feature_rows)
    feature_df.to_csv(feature_path, index=False)
    return feature_df


def _load_or_build_full_session_feature_df(data_dir: Path, *, sessions: list[Session], graph) -> pd.DataFrame:
    feature_path = data_dir / "full_session_features.csv"
    if feature_path.exists():
        return pd.read_csv(feature_path)
    rows: list[dict[str, float | int | str]] = []
    for session in sessions:
        if len(session.events) < 2:
            continue
        features = extract_features_for_events(session.events, graph)
        rows.append(
            {
                "session_id": session.session_id,
                "prefix_len": len(session.events),
                "label": session.label,
                **features,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(feature_path, index=False)
    return df



def run_experiments(
    feature_df: pd.DataFrame,
    *,
    prefixes: Iterable[int],
    output_dir: Path,
    n_bootstrap: int,
    random_state: int,
    save_models: bool,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
    session_metadata: pd.DataFrame | None = None,
    full_session_df: pd.DataFrame | None = None,
) -> list[ExperimentArtifact]:
    prefixes = sorted(set(int(value) for value in prefixes))
    labeled_df = feature_df[feature_df["label"] != DEFAULT_UNKNOWN_LABEL].copy()
    if labeled_df.empty:
        raise ValueError("No labeled rows are available for experiments")

    session_metadata = _ensure_session_metadata(labeled_df, session_metadata=session_metadata)
    full_session_df = _ensure_full_session_df(labeled_df, full_session_df=full_session_df)

    session_ids = session_metadata["session_id"].drop_duplicates().tolist()
    if len(session_ids) < 6:
        raise ValueError("Need at least six sessions for train/validation/test experiments")

    split_ids = _make_standard_split_ids(session_metadata, random_state=random_state)
    _build_split_summary(session_metadata, split_ids).to_csv(output_dir / "split_summary.csv", index=False)
    _build_leakage_audit(session_metadata, split_ids).to_csv(output_dir / "leakage_audit.csv", index=False)

    model_output_dir = output_dir / "models"
    if save_models:
        model_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _run_standard_and_baseline_experiments(
        labeled_df,
        full_session_df,
        session_metadata,
        split_ids=split_ids,
        prefixes=prefixes,
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        save_models=save_models,
        positive_label=positive_label,
    )
    artifacts.extend(
        _run_family_holdout_experiments(
            labeled_df,
            full_session_df,
            session_metadata,
            prefixes=prefixes,
            output_dir=output_dir,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            positive_label=positive_label,
        )
    )
    return artifacts



def build_leaderboard(artifacts: list[ExperimentArtifact]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for artifact in artifacts:
        metrics_df = artifact.metrics_by_prefix.copy()
        if metrics_df.empty:
            continue
        final_row = metrics_df.iloc[-1]
        final_prefix = int(final_row["prefix_len"])
        bot_delay = artifact.detection_delay_summary
        bot_delay_row = bot_delay[bot_delay["label"] == "bot"]
        mean_detection_prefix = float(bot_delay_row["mean_first_detected_prefix"].iloc[0]) if not bot_delay_row.empty else float("nan")
        detection_rate = float(bot_delay_row["detection_rate"].iloc[0]) if not bot_delay_row.empty else float("nan")
        rows.append(
            {
                "protocol": artifact.protocol,
                "model_name": artifact.model_name,
                "ablation_name": artifact.ablation_name,
                "threshold": artifact.threshold,
                "threshold_strategy": artifact.threshold_strategy,
                "final_prefix": final_prefix,
                "decision_point": "session_complete" if final_prefix == FULL_SESSION_SENTINEL_PREFIX else str(final_prefix),
                "accuracy": float(final_row["accuracy"]),
                "precision": float(final_row["precision"]),
                "recall": float(final_row["recall"]),
                "f1": float(final_row["f1"]),
                "roc_auc": float(final_row.get("roc_auc", float("nan"))),
                "pr_auc": float(final_row.get("pr_auc", float("nan"))),
                "bot_detection_rate": detection_rate,
                "mean_first_detected_prefix_bot": mean_detection_prefix,
                "notes": artifact.notes,
            }
        )
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(["f1", "roc_auc", "pr_auc"], ascending=False).reset_index(drop=True)
    return leaderboard



def _feature_columns_for_families(all_columns: list[str], family_names: Iterable[str]) -> list[str]:
    selected: list[str] = []
    for family_name in family_names:
        selected.extend(FEATURE_GROUPS[family_name])
    return [col for col in selected if col in all_columns]



def _full_session_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["session_id", "prefix_len"]).drop_duplicates("session_id", keep="last")



def _predict_across_prefixes(
    model,
    test_df: pd.DataFrame,
    *,
    prefixes: Iterable[int],
    feature_columns: list[str],
    threshold_by_prefix: dict[int, float] | None,
    default_threshold: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for prefix_len in sorted(set(int(x) for x in prefixes)):
        slice_df = test_df[test_df["prefix_len"] == prefix_len].copy()
        if slice_df.empty:
            continue
        scores = model.predict_proba(slice_df[feature_columns])[:, 1]
        threshold = float(threshold_by_prefix.get(prefix_len, default_threshold)) if threshold_by_prefix else float(default_threshold)
        slice_df["bot_probability"] = scores
        slice_df["threshold_used"] = threshold
        slice_df["predicted_bot"] = (scores >= threshold).astype(int)
        rows.append(slice_df[["session_id", "prefix_len", "label", "bot_probability", "threshold_used", "predicted_bot"]])
    if not rows:
        return pd.DataFrame(columns=["session_id", "prefix_len", "label", "bot_probability", "threshold_used", "predicted_bot"])
    return pd.concat(rows, ignore_index=True)



def _metrics_by_prefix(predictions: pd.DataFrame, *, positive_label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for prefix_len, group in predictions.groupby("prefix_len"):
        y_true = (group["label"] == positive_label).astype(int).to_numpy()
        y_score = group["bot_probability"].to_numpy()
        if "predicted_bot" in group.columns:
            y_pred = group["predicted_bot"].to_numpy()
        else:
            threshold = float(group["threshold_used"].iloc[0]) if "threshold_used" in group.columns else 0.5
            y_pred = (y_score >= threshold).astype(int)
        metrics = compute_binary_metrics(y_true, y_pred, y_score)
        row: dict[str, float | int] = {"prefix_len": int(prefix_len), **metrics}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("prefix_len").reset_index(drop=True)


def _run_standard_and_baseline_experiments(
    prefix_df: pd.DataFrame,
    full_session_df: pd.DataFrame,
    session_metadata: pd.DataFrame,
    *,
    split_ids: dict[str, list[str]],
    prefixes: list[int],
    output_dir: Path,
    model_output_dir: Path,
    n_bootstrap: int,
    random_state: int,
    save_models: bool,
    positive_label: str,
) -> list[ExperimentArtifact]:
    train_prefix_df = prefix_df[prefix_df["session_id"].isin(split_ids["train"])].copy()
    val_prefix_df = prefix_df[prefix_df["session_id"].isin(split_ids["val"])].copy()
    test_prefix_df = prefix_df[prefix_df["session_id"].isin(split_ids["test"])].copy()
    train_full_df = full_session_df[full_session_df["session_id"].isin(split_ids["train"])].copy()
    val_full_df = full_session_df[full_session_df["session_id"].isin(split_ids["val"])].copy()
    test_full_df = full_session_df[full_session_df["session_id"].isin(split_ids["test"])].copy()

    artifacts: list[ExperimentArtifact] = []
    models = build_models(random_state=random_state)
    for ablation_name, families in ABLATIONS.items():
        feature_columns = _feature_columns_for_families(train_full_df.columns.tolist(), families)
        if not feature_columns:
            continue
        for model_name, model in models.items():
            artifact, fitted = _fit_prefix_model_artifact(
                model_name=model_name,
                ablation_name=ablation_name,
                protocol="session_split",
                model=model,
                train_full_df=train_full_df,
                val_prefix_df=val_prefix_df,
                val_full_df=val_full_df,
                test_prefix_df=test_prefix_df,
                prefixes=prefixes,
                feature_columns=feature_columns,
                n_bootstrap=n_bootstrap,
                positive_label=positive_label,
                notes="",
            )
            if artifact is None:
                continue
            artifacts.append(artifact)
            _write_artifact_outputs(artifact, output_dir)

            if save_models:
                bundle = make_model_bundle(
                    type("ArtifactShim", (), {"model_name": f"{model_name}_{ablation_name}", "feature_columns": feature_columns, "model": fitted})(),
                    threshold=artifact.threshold,
                )
                save_model_bundle(bundle, model_output_dir / f"{artifact.artifact_id}_bundle.pkl")

            if ablation_name == "all_features":
                full_session_artifact = _build_full_session_baseline_artifact(
                    model_name=model_name,
                    fitted_model=fitted,
                    test_full_df=test_full_df,
                    feature_columns=feature_columns,
                    threshold=artifact.threshold,
                    n_bootstrap=n_bootstrap,
                    positive_label=positive_label,
                )
                artifacts.append(full_session_artifact)
                _write_artifact_outputs(full_session_artifact, output_dir)

    heuristic_artifact = _build_heuristic_artifact(
        val_prefix_df=val_prefix_df,
        test_prefix_df=test_prefix_df,
        session_metadata=session_metadata,
        n_bootstrap=n_bootstrap,
        positive_label=positive_label,
    )
    if heuristic_artifact is not None:
        artifacts.append(heuristic_artifact)
        _write_artifact_outputs(heuristic_artifact, output_dir)

    static_artifact = _build_static_baseline_artifact(
        session_metadata=session_metadata,
        split_ids=split_ids,
        test_prefix_df=test_prefix_df,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        positive_label=positive_label,
    )
    if static_artifact is not None:
        artifacts.append(static_artifact)
        _write_artifact_outputs(static_artifact, output_dir)

    return artifacts


def _run_family_holdout_experiments(
    prefix_df: pd.DataFrame,
    full_session_df: pd.DataFrame,
    session_metadata: pd.DataFrame,
    *,
    prefixes: list[int],
    output_dir: Path,
    n_bootstrap: int,
    random_state: int,
    positive_label: str,
) -> list[ExperimentArtifact]:
    bot_meta = session_metadata[session_metadata["label"] == positive_label].copy()
    family_counts = bot_meta["bot_family"].fillna("").astype(str).value_counts()
    families = [family for family, count in family_counts.items() if family and family != "human" and int(count) >= 3]
    if len(families) < 2:
        return []

    artifacts: list[ExperimentArtifact] = []
    for family in families:
        split_ids = _make_family_holdout_split_ids(session_metadata, held_out_family=family, random_state=random_state)
        if split_ids is None:
            continue
        _build_split_summary(session_metadata, split_ids, protocol="leave_one_bot_family_out", note=family).to_csv(
            output_dir / f"split_summary_holdout_{_slugify(family)}.csv",
            index=False,
        )
        train_full_df = full_session_df[full_session_df["session_id"].isin(split_ids["train"])].copy()
        val_full_df = full_session_df[full_session_df["session_id"].isin(split_ids["val"])].copy()
        val_prefix_df = prefix_df[prefix_df["session_id"].isin(split_ids["val"])].copy()
        test_prefix_df = prefix_df[prefix_df["session_id"].isin(split_ids["test"])].copy()
        feature_columns = _feature_columns_for_families(train_full_df.columns.tolist(), ABLATIONS["all_features"])
        if not feature_columns:
            continue
        for model_name, model in build_models(random_state=random_state).items():
            artifact, _ = _fit_prefix_model_artifact(
                model_name=model_name,
                ablation_name="all_features",
                protocol="leave_one_bot_family_out",
                model=model,
                train_full_df=train_full_df,
                val_prefix_df=val_prefix_df,
                val_full_df=val_full_df,
                test_prefix_df=test_prefix_df,
                prefixes=prefixes,
                feature_columns=feature_columns,
                n_bootstrap=n_bootstrap,
                positive_label=positive_label,
                notes=f"held_out_family={family}",
            )
            if artifact is None:
                continue
            artifacts.append(artifact)
            _write_artifact_outputs(artifact, output_dir)
    return artifacts


def _fit_prefix_model_artifact(
    *,
    model_name: str,
    ablation_name: str,
    protocol: str,
    model,
    train_full_df: pd.DataFrame,
    val_prefix_df: pd.DataFrame,
    val_full_df: pd.DataFrame,
    test_prefix_df: pd.DataFrame,
    prefixes: list[int],
    feature_columns: list[str],
    n_bootstrap: int,
    positive_label: str,
    notes: str,
) -> tuple[ExperimentArtifact | None, object | None]:
    if train_full_df.empty or val_full_df.empty or test_prefix_df.empty:
        return None, None
    if train_full_df["label"].nunique() < 2 or val_full_df["label"].nunique() < 2:
        return None, None

    fitted = clone(model)
    fitted.fit(train_full_df[feature_columns], (train_full_df["label"] == positive_label).astype(int))
    val_full_scores = fitted.predict_proba(val_full_df[feature_columns])[:, 1]
    default_threshold = _safe_tune_threshold((val_full_df["label"] == positive_label).astype(int).to_numpy(), val_full_scores, default_threshold=0.5)

    val_predictions = _predict_across_prefixes(
        fitted,
        val_prefix_df,
        prefixes=prefixes,
        feature_columns=feature_columns,
        threshold_by_prefix=None,
        default_threshold=default_threshold,
    )
    threshold_by_prefix = tune_thresholds_by_prefix(
        val_predictions,
        positive_label=positive_label,
        default_threshold=default_threshold,
    )
    if not threshold_by_prefix:
        threshold_by_prefix = {int(prefix): float(default_threshold) for prefix in prefixes}

    predictions = _predict_across_prefixes(
        fitted,
        test_prefix_df,
        prefixes=prefixes,
        feature_columns=feature_columns,
        threshold_by_prefix=threshold_by_prefix,
        default_threshold=default_threshold,
    )
    if predictions.empty:
        return None, None

    metrics_by_prefix = _metrics_by_prefix(predictions, positive_label=positive_label)
    ci_df = attach_metric_confidence_intervals(
        predictions,
        threshold=threshold_by_prefix,
        n_bootstrap=n_bootstrap,
        positive_label=positive_label,
    )
    detection_delay = summarize_detection_delay(predictions, positive_label=positive_label)
    artifact = ExperimentArtifact(
        artifact_id=f"{_slugify(protocol)}__{_slugify(model_name)}__{_slugify(ablation_name)}{_notes_suffix(notes)}",
        model_name=model_name,
        ablation_name=ablation_name,
        protocol=protocol,
        threshold=float(threshold_by_prefix.get(max(threshold_by_prefix), default_threshold)),
        threshold_strategy="prefix_specific",
        metrics_by_prefix=metrics_by_prefix,
        predictions=predictions,
        confidence_intervals=ci_df,
        detection_delay_summary=detection_delay,
        notes=notes,
    )
    return artifact, fitted


def _build_heuristic_artifact(
    *,
    val_prefix_df: pd.DataFrame,
    test_prefix_df: pd.DataFrame,
    session_metadata: pd.DataFrame,
    n_bootstrap: int,
    positive_label: str,
) -> ExperimentArtifact | None:
    val_predictions = _heuristic_predictions(val_prefix_df, session_metadata)
    if val_predictions.empty:
        return None
    val_full = _full_session_rows(val_predictions)
    default_threshold = _safe_tune_threshold((val_full["label"] == positive_label).astype(int).to_numpy(), val_full["bot_probability"].to_numpy(), default_threshold=0.5)
    threshold_by_prefix = tune_thresholds_by_prefix(
        val_predictions,
        positive_label=positive_label,
        default_threshold=default_threshold,
    )
    predictions = _heuristic_predictions(
        test_prefix_df,
        session_metadata,
        threshold_by_prefix=threshold_by_prefix,
        default_threshold=default_threshold,
    )
    if predictions.empty:
        return None
    metrics_by_prefix = _metrics_by_prefix(predictions, positive_label=positive_label)
    ci_df = attach_metric_confidence_intervals(predictions, threshold=threshold_by_prefix, n_bootstrap=n_bootstrap, positive_label=positive_label)
    detection_delay = summarize_detection_delay(predictions, positive_label=positive_label)
    return ExperimentArtifact(
        artifact_id="session_split__heuristic_rules__heuristic_baseline",
        model_name="heuristic_rules",
        ablation_name="heuristic_baseline",
        protocol="session_split",
        threshold=float(threshold_by_prefix.get(max(threshold_by_prefix), default_threshold)),
        threshold_strategy="prefix_specific",
        metrics_by_prefix=metrics_by_prefix,
        predictions=predictions,
        confidence_intervals=ci_df,
        detection_delay_summary=detection_delay,
        notes="entropy+timing+ua heuristics",
    )


def _heuristic_predictions(
    df: pd.DataFrame,
    session_metadata: pd.DataFrame,
    *,
    threshold_by_prefix: dict[int, float] | None = None,
    default_threshold: float = 0.5,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["session_id", "prefix_len", "label", "bot_probability", "threshold_used", "predicted_bot"])
    join_columns = [column for column in ("session_id", "has_suspect_user_agent", "ua_browser_like") if column in session_metadata.columns]
    working = df.merge(session_metadata[join_columns], on="session_id", how="left")

    score = np.zeros(len(working), dtype=float)
    weight_total = 0.0
    for column, weight, transform in (
        ("navigation_entropy_score", 0.28, lambda s: np.clip(1.0 - s.fillna(0.5).to_numpy(), 0.0, 1.0)),
        ("repeated_transition_ratio", 0.16, lambda s: np.clip(s.fillna(0.0).to_numpy(), 0.0, 1.0)),
        ("low_latency_ratio", 0.17, lambda s: np.clip(s.fillna(0.0).to_numpy(), 0.0, 1.0)),
        ("leaf_visit_ratio", 0.12, lambda s: np.clip(s.fillna(0.0).to_numpy(), 0.0, 1.0)),
        ("far_jump_ratio", 0.08, lambda s: np.clip(s.fillna(0.0).to_numpy(), 0.0, 1.0)),
    ):
        if column in working.columns:
            score += weight * transform(working[column])
            weight_total += weight
    if "has_suspect_user_agent" in working.columns:
        score += 0.12 * working["has_suspect_user_agent"].fillna(0.0).astype(float).to_numpy()
        weight_total += 0.12
    if "ua_browser_like" in working.columns:
        score += 0.07 * (1.0 - working["ua_browser_like"].fillna(0.0).astype(float).to_numpy())
        weight_total += 0.07
    weight_total = max(weight_total, 1.0)

    working["bot_probability"] = np.clip(score / weight_total, 0.0, 1.0)
    working["threshold_used"] = working["prefix_len"].map(lambda prefix: float(threshold_by_prefix.get(int(prefix), default_threshold)) if threshold_by_prefix else float(default_threshold))
    working["predicted_bot"] = (working["bot_probability"] >= working["threshold_used"]).astype(int)
    return working[["session_id", "prefix_len", "label", "bot_probability", "threshold_used", "predicted_bot"]].copy()


def _build_static_baseline_artifact(
    *,
    session_metadata: pd.DataFrame,
    split_ids: dict[str, list[str]],
    test_prefix_df: pd.DataFrame,
    n_bootstrap: int,
    random_state: int,
    positive_label: str,
) -> ExperimentArtifact | None:
    train_meta = session_metadata[session_metadata["session_id"].isin(split_ids["train"])].copy()
    val_meta = session_metadata[session_metadata["session_id"].isin(split_ids["val"])].copy()
    test_meta = session_metadata[session_metadata["session_id"].isin(split_ids["test"])].copy()
    if train_meta.empty or val_meta.empty or test_meta.empty:
        return None
    if train_meta["label"].nunique() < 2 or val_meta["label"].nunique() < 2:
        return None

    X_train, X_val, X_test = _build_static_feature_matrices(train_meta, val_meta, test_meta)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    clf.fit(X_train, (train_meta["label"] == positive_label).astype(int))
    val_scores = clf.predict_proba(X_val)[:, 1]
    threshold = _safe_tune_threshold((val_meta["label"] == positive_label).astype(int).to_numpy(), val_scores, default_threshold=0.5)

    score_map = dict(zip(test_meta["session_id"], clf.predict_proba(X_test)[:, 1], strict=False))
    predictions = _replicate_session_scores_across_prefixes(
        test_prefix_df,
        score_map=score_map,
        threshold_by_prefix={int(prefix): float(threshold) for prefix in sorted(test_prefix_df["prefix_len"].unique())},
        default_threshold=threshold,
    )
    if predictions.empty:
        return None
    metrics_by_prefix = _metrics_by_prefix(predictions, positive_label=positive_label)
    ci_df = attach_metric_confidence_intervals(predictions, threshold=threshold, n_bootstrap=n_bootstrap, positive_label=positive_label)
    detection_delay = summarize_detection_delay(predictions, positive_label=positive_label)
    return ExperimentArtifact(
        artifact_id="session_split__static_logistic_regression__static_request_baseline",
        model_name="static_logistic_regression",
        ablation_name="static_request_baseline",
        protocol="session_split",
        threshold=float(threshold),
        threshold_strategy="global",
        metrics_by_prefix=metrics_by_prefix,
        predictions=predictions,
        confidence_intervals=ci_df,
        detection_delay_summary=detection_delay,
        notes="first-request and header-like features only",
    )


def _replicate_session_scores_across_prefixes(
    prefix_df: pd.DataFrame,
    *,
    score_map: dict[str, float],
    threshold_by_prefix: dict[int, float] | None,
    default_threshold: float,
) -> pd.DataFrame:
    working = prefix_df[["session_id", "prefix_len", "label"]].copy()
    working["bot_probability"] = working["session_id"].map(score_map)
    working = working.dropna(subset=["bot_probability"]).copy()
    working["threshold_used"] = working["prefix_len"].map(lambda prefix: float(threshold_by_prefix.get(int(prefix), default_threshold)) if threshold_by_prefix else float(default_threshold))
    working["predicted_bot"] = (working["bot_probability"] >= working["threshold_used"]).astype(int)
    return working


def _build_full_session_baseline_artifact(
    *,
    model_name: str,
    fitted_model,
    test_full_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
    n_bootstrap: int,
    positive_label: str,
) -> ExperimentArtifact:
    predictions = test_full_df[["session_id", "prefix_len", "label"]].copy()
    predictions["bot_probability"] = fitted_model.predict_proba(test_full_df[feature_columns])[:, 1]
    predictions["threshold_used"] = float(threshold)
    predictions["predicted_bot"] = (predictions["bot_probability"] >= threshold).astype(int)
    metrics = compute_binary_metrics(
        (predictions["label"] == positive_label).astype(int).to_numpy(),
        predictions["predicted_bot"].to_numpy(),
        predictions["bot_probability"].to_numpy(),
    )
    metrics_by_prefix = pd.DataFrame([{"prefix_len": FULL_SESSION_SENTINEL_PREFIX, **metrics}])
    ci_predictions = predictions.copy()
    ci_predictions["prefix_len"] = FULL_SESSION_SENTINEL_PREFIX
    ci_df = attach_metric_confidence_intervals(ci_predictions, threshold=threshold, n_bootstrap=n_bootstrap, positive_label=positive_label)
    detection_delay = summarize_detection_delay(predictions, positive_label=positive_label)
    return ExperimentArtifact(
        artifact_id=f"full_session_only__{_slugify(model_name)}__full_session_baseline",
        model_name=model_name,
        ablation_name="full_session_baseline",
        protocol="full_session_only",
        threshold=float(threshold),
        threshold_strategy="global",
        metrics_by_prefix=metrics_by_prefix,
        predictions=predictions,
        confidence_intervals=ci_df,
        detection_delay_summary=detection_delay,
        notes="decision allowed only after session completion",
    )


def _ensure_session_metadata(feature_df: pd.DataFrame, *, session_metadata: pd.DataFrame | None) -> pd.DataFrame:
    if session_metadata is not None and not session_metadata.empty:
        working = session_metadata.copy()
    else:
        working = _full_session_rows(feature_df)[["session_id", "label", "prefix_len"]].rename(columns={"prefix_len": "num_events"}).copy()
        for column, value in {
            "client_key": "",
            "ip": "",
            "user_agent": "",
            "ua_family": "unknown",
            "bot_family": "generic_bot",
            "has_suspect_user_agent": 0.0,
            "ua_length": 0.0,
            "ua_slash_count": 0.0,
            "ua_browser_like": 0.0,
            "first_path_depth": 0.0,
            "first_category": "unknown",
            "first_method": "GET",
            "first_status_code": 0.0,
            "first_body_bytes_sent": 0.0,
            "path_signature": None,
            "prefix3_signature": None,
            "prefix5_signature": None,
        }.items():
            working[column] = value
        working["path_signature"] = working["session_id"].astype(str)
        working["prefix3_signature"] = working["session_id"].astype(str)
        working["prefix5_signature"] = working["session_id"].astype(str)
    return working.drop_duplicates("session_id").reset_index(drop=True)


def _ensure_full_session_df(feature_df: pd.DataFrame, *, full_session_df: pd.DataFrame | None) -> pd.DataFrame:
    if full_session_df is not None and not full_session_df.empty:
        return full_session_df.copy()
    return _full_session_rows(feature_df)


def _build_session_metadata_from_sessions(sessions: list[Session]) -> pd.DataFrame:
    summary_df = summarize_sessions(sessions)
    summary_lookup = summary_df.set_index("session_id").to_dict(orient="index") if not summary_df.empty else {}
    rows: list[dict[str, object]] = []
    for session in sessions:
        if not session.events:
            continue
        first_event = session.events[0]
        paths = [event.path for event in session.events]
        categories = [event.page_category or infer_category_from_path(event.path) for event in session.events]
        user_agent = first_event.user_agent or ""
        bot_family = str(first_event.extra.get("bot_family", "") or "").strip() or _infer_bot_family(user_agent=user_agent, label=session.label)
        row = {
            "session_id": session.session_id,
            "label": session.label,
            "client_key": str(first_event.extra.get("client_key", "") or ""),
            "ip": str(first_event.extra.get("ip", "") or ""),
            "user_agent": user_agent,
            "ua_family": _infer_ua_family(user_agent),
            "bot_family": bot_family,
            "has_suspect_user_agent": float(_looks_like_bot_user_agent(user_agent)),
            "ua_length": float(len(user_agent)),
            "ua_slash_count": float(user_agent.count("/")),
            "ua_browser_like": float(_looks_browser_like(user_agent)),
            "first_path_depth": float(_path_depth(first_event.path)),
            "first_category": categories[0] if categories else "unknown",
            "first_method": str(first_event.extra.get("method", "") or "GET").upper(),
            "first_status_code": float(first_event.status_code or 0),
            "first_body_bytes_sent": _safe_float(first_event.extra.get("body_bytes_sent")),
            "path_signature": _sequence_signature(paths),
            "prefix3_signature": _sequence_signature(paths[:3]),
            "prefix5_signature": _sequence_signature(paths[:5]),
        }
        row.update(summary_lookup.get(session.session_id, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_standard_split_ids(session_metadata: pd.DataFrame, *, random_state: int) -> dict[str, list[str]]:
    if "path_signature" not in session_metadata.columns:
        session_ids = session_metadata["session_id"].tolist()
        labels = session_metadata["label"].tolist()
        stratify = labels if _can_stratify(labels) else None
        train_ids, temp_ids = train_test_split(session_ids, test_size=0.40, random_state=random_state, stratify=stratify)
        temp_meta = session_metadata[session_metadata["session_id"].isin(temp_ids)].copy()
        temp_labels = temp_meta["label"].tolist()
        temp_stratify = temp_labels if _can_stratify(temp_labels) else None
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=random_state, stratify=temp_stratify)
        return {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}

    rng = np.random.default_rng(random_state)
    splits = {"train": [], "val": [], "test": []}
    targets = {"train": 0.60, "val": 0.20, "test": 0.20}
    for label, label_meta in session_metadata.groupby("label"):
        groups: list[list[str]] = []
        for _, group in label_meta.groupby("path_signature", dropna=False):
            ids = group["session_id"].tolist()
            groups.append(ids if ids else [])
        rng.shuffle(groups)
        counts = {"train": 0, "val": 0, "test": 0}
        total = sum(len(group) for group in groups)
        desired = {name: total * ratio for name, ratio in targets.items()}
        for group_ids in sorted(groups, key=len, reverse=True):
            if not group_ids:
                continue
            split_name = max(desired, key=lambda name: (desired[name] - counts[name], -counts[name]))
            splits[split_name].extend(group_ids)
            counts[split_name] += len(group_ids)
    return {name: sorted(set(ids)) for name, ids in splits.items()}


def _make_family_holdout_split_ids(
    session_metadata: pd.DataFrame,
    *,
    held_out_family: str,
    random_state: int,
) -> dict[str, list[str]] | None:
    human_ids = session_metadata[session_metadata["label"] == "human"]["session_id"].tolist()
    held_out_bot_ids = session_metadata[(session_metadata["label"] == "bot") & (session_metadata["bot_family"] == held_out_family)]["session_id"].tolist()
    remaining_bot_ids = session_metadata[(session_metadata["label"] == "bot") & (session_metadata["bot_family"] != held_out_family)]["session_id"].tolist()
    if len(human_ids) < 6 or len(held_out_bot_ids) < 3 or len(remaining_bot_ids) < 4:
        return None
    human_train, human_temp = train_test_split(human_ids, test_size=0.40, random_state=random_state)
    human_val, human_test = train_test_split(human_temp, test_size=0.50, random_state=random_state)
    bot_train, bot_val = train_test_split(remaining_bot_ids, test_size=0.25, random_state=random_state)
    return {
        "train": sorted(list(human_train) + list(bot_train)),
        "val": sorted(list(human_val) + list(bot_val)),
        "test": sorted(list(human_test) + list(held_out_bot_ids)),
    }


def _build_static_feature_matrices(train_meta: pd.DataFrame, val_meta: pd.DataFrame, test_meta: pd.DataFrame):
    train_frame = _build_static_feature_frame(train_meta)
    val_frame = _build_static_feature_frame(val_meta)
    test_frame = _build_static_feature_frame(test_meta)
    dummy_columns = [column for column in STATIC_CATEGORICAL_FEATURES if column in train_frame.columns]
    train_encoded = pd.get_dummies(train_frame, columns=dummy_columns, dtype=float)
    val_encoded = pd.get_dummies(val_frame, columns=dummy_columns, dtype=float).reindex(columns=train_encoded.columns, fill_value=0.0)
    test_encoded = pd.get_dummies(test_frame, columns=dummy_columns, dtype=float).reindex(columns=train_encoded.columns, fill_value=0.0)
    return train_encoded.astype(float), val_encoded.astype(float), test_encoded.astype(float)


def _build_static_feature_frame(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=metadata.index)
    for column in STATIC_NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(metadata[column], errors="coerce").fillna(0.0) if column in metadata.columns else 0.0
    for column in STATIC_CATEGORICAL_FEATURES:
        frame[column] = metadata[column].fillna("unknown").astype(str) if column in metadata.columns else "unknown"
    return frame


def _build_split_summary(
    session_metadata: pd.DataFrame,
    split_ids: dict[str, list[str]],
    *,
    protocol: str = "session_split",
    note: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, ids in split_ids.items():
        subset = session_metadata[session_metadata["session_id"].isin(ids)].copy()
        rows.append(
            {
                "protocol": protocol,
                "split": split_name,
                "note": note,
                "num_sessions": int(len(subset)),
                "num_humans": int((subset["label"] == "human").sum()),
                "num_bots": int((subset["label"] == "bot").sum()),
                "bot_family_counts": "; ".join(f"{family}:{count}" for family, count in subset[subset["label"] == "bot"]["bot_family"].value_counts().items()),
            }
        )
    return pd.DataFrame(rows)


def _build_leakage_audit(session_metadata: pd.DataFrame, split_ids: dict[str, list[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
        left = session_metadata[session_metadata["session_id"].isin(split_ids[left_name])].copy()
        right = session_metadata[session_metadata["session_id"].isin(split_ids[right_name])].copy()
        for column in ("session_id", "client_key", "path_signature", "prefix3_signature", "prefix5_signature"):
            overlap = sorted(_normalize_overlap_values(left.get(column)) & _normalize_overlap_values(right.get(column)))
            rows.append(
                {
                    "split_pair": f"{left_name}_vs_{right_name}",
                    "key": column,
                    "shared_count": int(len(overlap)),
                    "examples": "; ".join(overlap[:5]),
                }
            )
    return pd.DataFrame(rows)


def _write_artifact_outputs(artifact: ExperimentArtifact, output_dir: Path) -> None:
    artifact.metrics_by_prefix.to_csv(output_dir / f"metrics_{artifact.artifact_id}.csv", index=False)
    artifact.predictions.to_csv(output_dir / f"predictions_{artifact.artifact_id}.csv", index=False)
    artifact.confidence_intervals.to_csv(output_dir / f"confidence_intervals_{artifact.artifact_id}.csv", index=False)
    artifact.detection_delay_summary.to_csv(output_dir / f"detection_delay_{artifact.artifact_id}.csv", index=False)



def _write_markdown_summary(
    path: Path,
    leaderboard: pd.DataFrame,
    artifacts: list[ExperimentArtifact],
    *,
    split_summary: pd.DataFrame | None = None,
    leakage_audit: pd.DataFrame | None = None,
) -> None:
    lines: list[str] = [
        "# Experiment Summary",
        "",
        "This report was generated automatically from the research experiment runner.",
        "",
        "The current pipeline now checks the proposal's key requirements:",
        "",
        "- early detection from session prefixes",
        "- graph, timing, and entropy-aware features",
        "- heuristic, static, and full-session baselines",
        "- split leakage auditing and family holdout evaluation",
        "",
    ]
    if leaderboard.empty:
        lines.append("No experiment results were produced.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    top = leaderboard.iloc[0]
    lines.extend(
        [
            "## Best configuration",
            "",
            f"- Protocol: **{top['protocol']}**",
            f"- Model: **{top['model_name']}**",
            f"- Feature set: **{top['ablation_name']}**",
            f"- Decision point: **{top['decision_point']}**",
            f"- Threshold strategy: **{top['threshold_strategy']}**",
            f"- Final F1: **{top['f1']:.3f}**",
            f"- Final ROC-AUC: **{top['roc_auc']:.3f}**",
            f"- Mean first detection prefix for bots: **{top['mean_first_detected_prefix_bot']:.2f}**",
            "",
            "## Leaderboard",
            "",
            _df_to_markdown(leaderboard),
            "",
        ]
    )

    if split_summary is not None and not split_summary.empty:
        lines.extend(
            [
                "## Split summary",
                "",
                _df_to_markdown(split_summary),
                "",
            ]
        )

    if leakage_audit is not None and not leakage_audit.empty:
        lines.extend(
            [
                "## Leakage audit",
                "",
                _df_to_markdown(leakage_audit),
                "",
            ]
        )

    baseline_rows = leaderboard[leaderboard["ablation_name"].isin({"heuristic_baseline", "static_request_baseline", "full_session_baseline"})]
    if not baseline_rows.empty:
        lines.extend(
            [
                "## Baseline comparison",
                "",
                _df_to_markdown(baseline_rows),
                "",
            ]
        )

    holdout_rows = leaderboard[leaderboard["protocol"] == "leave_one_bot_family_out"]
    if not holdout_rows.empty:
        lines.extend(
            [
                "## Leave-one-bot-family-out",
                "",
                _df_to_markdown(holdout_rows),
                "",
            ]
        )

    for artifact in artifacts:
        if (
            artifact.protocol == top["protocol"]
            and artifact.model_name == top["model_name"]
            and artifact.ablation_name == top["ablation_name"]
            and artifact.notes == top["notes"]
        ):
            lines.extend(
                [
                    "## Confidence intervals for the best configuration",
                    "",
                    _df_to_markdown(artifact.confidence_intervals),
                    "",
                    "## Detection delay summary",
                    "",
                    _df_to_markdown(artifact.detection_delay_summary),
                    "",
                ]
            )
            break

    path.write_text("\n".join(lines), encoding="utf-8")



def _maybe_plot(artifacts: list[ExperimentArtifact], output_dir: Path, *, feature_df: pd.DataFrame | None = None) -> None:
    if plt is None or not artifacts:
        return

    leaderboard = build_leaderboard(artifacts)
    if leaderboard.empty:
        return

    standard_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.protocol == "session_split" and artifact.metrics_by_prefix["prefix_len"].min() >= 0
    ]
    if standard_artifacts:
        standard_leaderboard = build_leaderboard(standard_artifacts)
        top_configs = standard_leaderboard.head(3)[["model_name", "ablation_name", "notes"]].to_records(index=False)
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        for model_name, ablation_name, notes in top_configs:
            artifact = next(
                item
                for item in standard_artifacts
                if item.model_name == model_name and item.ablation_name == ablation_name and item.notes == notes
            )
            ax.plot(artifact.metrics_by_prefix["prefix_len"], artifact.metrics_by_prefix["f1"], marker="o", label=f"{model_name} | {ablation_name}")
        ax.set_xlabel("Prefix length")
        ax.set_ylabel("F1 score")
        ax.set_title("Early-detection F1 by prefix length")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "plot_f1_by_prefix.png", dpi=150)
        plt.close(fig)

    # Plot 2: leaderboard bar chart.
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    leaderboard_view = leaderboard.head(12)
    x_labels = [f"{row.model_name}\n{row.ablation_name}\n{row.protocol}" for row in leaderboard_view.itertuples(index=False)]
    ax.bar(range(len(leaderboard_view)), leaderboard_view["f1"].tolist())
    ax.set_xticks(range(len(leaderboard_view)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel("Final F1")
    ax.set_title("Experiment leaderboard")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_leaderboard_f1.png", dpi=150)
    plt.close(fig)

    if feature_df is not None and not feature_df.empty and "navigation_entropy_score" in feature_df.columns:
        final_rows = feature_df.sort_values(["session_id", "prefix_len"]).drop_duplicates("session_id", keep="last")
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(1, 1, 1)
        grouped = [final_rows.loc[final_rows["label"] == label, "navigation_entropy_score"].to_numpy() for label in ["human", "bot"]]
        ax.boxplot(grouped, tick_labels=["human", "bot"])
        ax.set_ylabel("Navigation entropy score")
        ax.set_title("Entropy distribution by class")
        fig.tight_layout()
        fig.savefig(output_dir / "plot_entropy_distribution.png", dpi=150)
        plt.close(fig)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *rows])


def _safe_tune_threshold(y_true: np.ndarray, y_score: np.ndarray, *, default_threshold: float) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float(default_threshold)
    return float(tune_threshold(y_true, y_score))


def _can_stratify(labels: list[str]) -> bool:
    series = pd.Series(labels)
    return not series.empty and int(series.value_counts().min()) >= 2


def _looks_like_bot_user_agent(user_agent: str) -> bool:
    lowered = user_agent.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in DEFAULT_BOT_USER_AGENT_PATTERNS)


def _looks_browser_like(user_agent: str) -> bool:
    lowered = user_agent.lower()
    return any(token in lowered for token in ("mozilla/", "chrome/", "firefox/", "safari/", "edg/"))


def _infer_ua_family(user_agent: str) -> str:
    lowered = user_agent.lower()
    if "python-requests" in lowered:
        return "python_requests"
    if "researchscraper" in lowered:
        return "research_scraper"
    if "researchstealth" in lowered or "stealth-browser" in lowered:
        return "stealth_browser"
    if _looks_browser_like(user_agent):
        return "browser_like"
    if not user_agent:
        return "missing"
    return "other"


def _infer_bot_family(*, user_agent: str, label: str) -> str:
    if label != DEFAULT_POSITIVE_LABEL:
        return "human"
    lowered = user_agent.lower()
    for family_name, pattern in (
        ("bfs", r"bfs"),
        ("dfs", r"dfs"),
        ("linear", r"linear"),
        ("product_focus", r"productfocus|products"),
        ("article_focus", r"articlefocus|articles"),
        ("stealth_revisit", r"researchstealth|stealth-browser|revisit"),
        ("generic_requests", r"python-requests|curl/|wget/|urllib|httpx|aiohttp"),
    ):
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return family_name
    return "generic_bot"


def _path_depth(path: str) -> int:
    return len([part for part in str(path).split("/") if part])


def _sequence_signature(paths: list[str]) -> str:
    if not paths:
        return ""
    return sha1(" > ".join(paths).encode("utf-8")).hexdigest()[:12]


def _normalize_overlap_values(series: pd.Series | None) -> set[str]:
    if series is None:
        return set()
    return {
        str(value).strip()
        for value in series.dropna().astype(str).tolist()
        if str(value).strip() and str(value).strip().lower() != "nan"
    }


def _safe_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "item"


def _notes_suffix(notes: str) -> str:
    return f"__{_slugify(notes)}" if notes else ""


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
