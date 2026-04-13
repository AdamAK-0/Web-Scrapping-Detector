"""Research experiment runner with baselines, audits, and thesis-focused reports."""

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
        "branching_concentration",
        "coverage_concentration",
        "transition_concentration",
        "category_concentration",
        "revisit_growth",
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
        "category_transition_entropy",
        "normalized_category_transition_entropy",
        "local_branching_entropy",
        "node_entropy_delta",
        "transition_entropy_delta",
        "category_entropy_delta",
        "category_transition_entropy_delta",
        "normalized_node_entropy_delta",
        "normalized_transition_entropy_delta",
        "normalized_category_entropy_delta",
        "normalized_category_transition_entropy_delta",
        "entropy_slope",
        "transition_entropy_slope",
        "category_transition_entropy_slope",
        "navigation_entropy_score",
        "navigation_entropy_score_v2",
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
DEFAULT_PROTOCOLS = [
    "session_split",
    "hard_prefix_session_split",
    "leave_one_bot_family_out",
    "time_split",
    "leave_one_human_user_out",
]
DEFAULT_HARD_PREFIXES = [3, 5, 10]
ENTROPY_VARIANT_COLUMNS = [
    "node_entropy",
    "normalized_node_entropy",
    "transition_entropy",
    "normalized_transition_entropy",
    "category_entropy",
    "normalized_category_entropy",
    "category_transition_entropy",
    "normalized_category_transition_entropy",
    "local_branching_entropy",
    "navigation_entropy_score",
    "navigation_entropy_score_v2",
    "normalized_node_entropy_delta",
    "normalized_transition_entropy_delta",
    "normalized_category_transition_entropy_delta",
    "entropy_slope",
    "transition_entropy_slope",
    "category_transition_entropy_slope",
]


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
    parser.add_argument("--model-set", nargs="*", default=None, help="Optional subset of model names to run")
    parser.add_argument("--protocols", nargs="*", default=DEFAULT_PROTOCOLS, help="Protocols to enable")
    parser.add_argument("--entropy-variants", nargs="*", default=ENTROPY_VARIANT_COLUMNS, help="Entropy features to compare in the entropy audit")
    parser.add_argument("--hard-prefixes", nargs="*", type=int, default=DEFAULT_HARD_PREFIXES, help="Short prefixes to emphasize in hard-prefix reporting")
    parser.add_argument("--group-key", default="path_signature", help="Session metadata key used for grouped splitting")
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
    summary_df = _read_if_exists(data_dir / "session_summary.csv")
    if summary_df is not None and not summary_df.empty:
        session_metadata = _merge_session_metadata_summary(session_metadata, summary_df)
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
        selected_models=args.model_set,
        protocols=args.protocols,
        entropy_variants=args.entropy_variants,
        hard_prefixes=args.hard_prefixes,
        group_key=args.group_key,
    )

    leaderboard = build_leaderboard(artifacts)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    split_summary = _read_if_exists(output_dir / "split_summary.csv")
    leakage_audit = _read_if_exists(output_dir / "leakage_audit.csv")
    shortcut_audit = _read_if_exists(output_dir / "shortcut_audit.csv")
    entropy_comparison = _read_if_exists(output_dir / "entropy_variant_comparison.csv")
    _write_markdown_summary(
        output_dir / "summary.md",
        leaderboard,
        artifacts,
        split_summary=split_summary,
        leakage_audit=leakage_audit,
        shortcut_audit=shortcut_audit,
        entropy_comparison=entropy_comparison,
    )
    _maybe_plot(artifacts, output_dir, feature_df=full_session_df, hard_prefixes=args.hard_prefixes)
    print(f"Experiment outputs written to: {output_dir.resolve()}")
    print(leaderboard.to_string(index=False))



def _load_or_build_feature_df(data_dir: Path, *, sessions: list[Session], graph, prefixes: Iterable[int]) -> pd.DataFrame:
    feature_path = data_dir / "prefix_features.csv"
    if feature_path.exists():
        feature_df = pd.read_csv(feature_path)
        if {"navigation_entropy_score_v2", "local_branching_entropy"}.issubset(feature_df.columns):
            return feature_df
    feature_rows = extract_prefix_feature_rows(sessions, graph, prefixes=prefixes)
    feature_df = prefix_rows_to_dataframe(feature_rows)
    feature_df.to_csv(feature_path, index=False)
    return feature_df


def _load_or_build_full_session_feature_df(data_dir: Path, *, sessions: list[Session], graph) -> pd.DataFrame:
    feature_path = data_dir / "full_session_features.csv"
    if feature_path.exists():
        feature_df = pd.read_csv(feature_path)
        if {"navigation_entropy_score_v2", "local_branching_entropy"}.issubset(feature_df.columns):
            return feature_df
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
    selected_models: Iterable[str] | None = None,
    protocols: Iterable[str] | None = None,
    entropy_variants: Iterable[str] | None = None,
    hard_prefixes: Iterable[int] | None = None,
    group_key: str = "path_signature",
) -> list[ExperimentArtifact]:
    prefixes = sorted(set(int(value) for value in prefixes))
    hard_prefixes = sorted(set(int(value) for value in (hard_prefixes or DEFAULT_HARD_PREFIXES) if int(value) in prefixes))
    labeled_df = feature_df[feature_df["label"] != DEFAULT_UNKNOWN_LABEL].copy()
    if labeled_df.empty:
        raise ValueError("No labeled rows are available for experiments")

    session_metadata = _ensure_session_metadata(labeled_df, session_metadata=session_metadata)
    full_session_df = _ensure_full_session_df(labeled_df, full_session_df=full_session_df)
    protocols = list(protocols or DEFAULT_PROTOCOLS)
    entropy_variants = [column for column in (entropy_variants or ENTROPY_VARIANT_COLUMNS) if column in labeled_df.columns or column in full_session_df.columns]

    session_ids = session_metadata["session_id"].drop_duplicates().tolist()
    if len(session_ids) < 6:
        raise ValueError("Need at least six sessions for train/validation/test experiments")
    split_summaries: list[pd.DataFrame] = []
    leakage_frames: list[pd.DataFrame] = []
    shortcut_audit = _build_shortcut_audit(session_metadata, full_session_df=full_session_df)
    shortcut_audit.to_csv(output_dir / "shortcut_audit.csv", index=False)
    shortcut_red_flags = _build_shortcut_red_flags(session_metadata, full_session_df=full_session_df, positive_label=positive_label)
    shortcut_red_flags.to_csv(output_dir / "shortcut_red_flags.csv", index=False)
    entropy_comparison = _build_entropy_variant_comparison(
        labeled_df,
        hard_prefixes=hard_prefixes or prefixes[:3],
        entropy_variants=entropy_variants,
        positive_label=positive_label,
    )
    entropy_comparison.to_csv(output_dir / "entropy_variant_comparison.csv", index=False)

    model_output_dir = output_dir / "models"
    if save_models:
        model_output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[ExperimentArtifact] = []
    if "session_split" in protocols or "hard_prefix_session_split" in protocols:
        split_ids = _make_standard_split_ids(session_metadata, random_state=random_state, group_key=group_key)
        split_summaries.append(_build_split_summary(session_metadata, split_ids, protocol="session_split"))
        leakage_frames.append(_build_leakage_audit(session_metadata, split_ids, protocol="session_split", group_key=group_key))
        if "session_split" in protocols:
            artifacts.extend(
                _run_experiment_suite_for_split(
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
                    protocol="session_split",
                    selected_models=selected_models,
                    selected_ablations=list(ABLATIONS),
                )
            )
        if "hard_prefix_session_split" in protocols and hard_prefixes:
            artifacts.extend(
                _run_experiment_suite_for_split(
                    labeled_df,
                    full_session_df,
                    session_metadata,
                    split_ids=split_ids,
                    prefixes=hard_prefixes,
                    output_dir=output_dir,
                    model_output_dir=model_output_dir,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                    save_models=False,
                    positive_label=positive_label,
                    protocol="hard_prefix_session_split",
                    selected_models=selected_models,
                    selected_ablations=list(ABLATIONS),
                )
            )
    if "time_split" in protocols:
        time_split_ids = _make_time_split_ids(session_metadata)
        if time_split_ids is not None:
            split_summaries.append(_build_split_summary(session_metadata, time_split_ids, protocol="time_split"))
            leakage_frames.append(_build_leakage_audit(session_metadata, time_split_ids, protocol="time_split", group_key=group_key))
            artifacts.extend(
                _run_experiment_suite_for_split(
                    labeled_df,
                    full_session_df,
                    session_metadata,
                    split_ids=time_split_ids,
                    prefixes=prefixes,
                    output_dir=output_dir,
                    model_output_dir=model_output_dir,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state,
                    save_models=False,
                    positive_label=positive_label,
                    protocol="time_split",
                    selected_models=selected_models,
                    selected_ablations=["all_features", "graph_plus_entropy", "entropy_only", "timing_only"],
                )
            )
    if "leave_one_bot_family_out" in protocols:
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
                selected_models=selected_models,
                split_summaries=split_summaries,
                leakage_frames=leakage_frames,
                group_key=group_key,
            )
        )
    if "leave_one_human_user_out" in protocols:
        artifacts.extend(
            _run_human_holdout_experiments(
                labeled_df,
                full_session_df,
                session_metadata,
                prefixes=prefixes,
                output_dir=output_dir,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                positive_label=positive_label,
                selected_models=selected_models,
                split_summaries=split_summaries,
                leakage_frames=leakage_frames,
                group_key=group_key,
            )
        )

    if split_summaries:
        pd.concat(split_summaries, ignore_index=True).to_csv(output_dir / "split_summary.csv", index=False)
    if leakage_frames:
        pd.concat(leakage_frames, ignore_index=True).to_csv(output_dir / "leakage_audit.csv", index=False)
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


def _run_experiment_suite_for_split(
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
    protocol: str,
    selected_models: Iterable[str] | None,
    selected_ablations: Iterable[str],
    notes: str = "",
) -> list[ExperimentArtifact]:
    train_prefix_df = prefix_df[prefix_df["session_id"].isin(split_ids["train"])].copy()
    val_prefix_df = prefix_df[prefix_df["session_id"].isin(split_ids["val"])].copy()
    test_prefix_df = prefix_df[prefix_df["session_id"].isin(split_ids["test"])].copy()
    train_full_df = full_session_df[full_session_df["session_id"].isin(split_ids["train"])].copy()
    val_full_df = full_session_df[full_session_df["session_id"].isin(split_ids["val"])].copy()
    test_full_df = full_session_df[full_session_df["session_id"].isin(split_ids["test"])].copy()

    artifacts: list[ExperimentArtifact] = []
    models = build_models(random_state=random_state, selected_models=selected_models)
    for ablation_name in selected_ablations:
        families = ABLATIONS[ablation_name]
        feature_columns = _feature_columns_for_families(train_full_df.columns.tolist(), families)
        if not feature_columns:
            continue
        for model_name, model in models.items():
            artifact, fitted = _fit_prefix_model_artifact(
                model_name=model_name,
                ablation_name=ablation_name,
                protocol=protocol,
                model=model,
                train_full_df=train_full_df,
                val_prefix_df=val_prefix_df,
                val_full_df=val_full_df,
                test_prefix_df=test_prefix_df,
                prefixes=prefixes,
                feature_columns=feature_columns,
                n_bootstrap=n_bootstrap,
                positive_label=positive_label,
                notes=notes,
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
                    protocol=protocol,
                    fitted_model=fitted,
                    test_full_df=test_full_df,
                    feature_columns=feature_columns,
                    threshold=artifact.threshold,
                    n_bootstrap=n_bootstrap,
                    positive_label=positive_label,
                    notes=notes,
                )
                artifacts.append(full_session_artifact)
                _write_artifact_outputs(full_session_artifact, output_dir)

    heuristic_artifact = _build_heuristic_artifact(
        val_prefix_df=val_prefix_df,
        test_prefix_df=test_prefix_df,
        session_metadata=session_metadata,
        n_bootstrap=n_bootstrap,
        positive_label=positive_label,
        protocol=protocol,
        notes=notes,
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
        protocol=protocol,
        notes=notes,
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
    selected_models: Iterable[str] | None,
    split_summaries: list[pd.DataFrame],
    leakage_frames: list[pd.DataFrame],
    group_key: str,
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
        split_summaries.append(_build_split_summary(session_metadata, split_ids, protocol="leave_one_bot_family_out", note=family))
        leakage_frames.append(
            _build_leakage_audit(session_metadata, split_ids, protocol="leave_one_bot_family_out", note=family, group_key=group_key)
        )
        artifacts.extend(
            _run_experiment_suite_for_split(
                prefix_df,
                full_session_df,
                session_metadata,
                split_ids=split_ids,
                prefixes=prefixes,
                output_dir=output_dir,
                model_output_dir=output_dir / "models",
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                save_models=False,
                positive_label=positive_label,
                protocol="leave_one_bot_family_out",
                selected_models=selected_models,
                selected_ablations=["all_features", "graph_plus_entropy", "entropy_only", "timing_only"],
                notes=f"held_out_family={family}",
            )
        )
    return artifacts


def _run_human_holdout_experiments(
    prefix_df: pd.DataFrame,
    full_session_df: pd.DataFrame,
    session_metadata: pd.DataFrame,
    *,
    prefixes: list[int],
    output_dir: Path,
    n_bootstrap: int,
    random_state: int,
    positive_label: str,
    selected_models: Iterable[str] | None,
    split_summaries: list[pd.DataFrame],
    leakage_frames: list[pd.DataFrame],
    group_key: str,
) -> list[ExperimentArtifact]:
    human_meta = session_metadata[(session_metadata["label"] != positive_label) & session_metadata["participant_id"].fillna("").astype(str).str.strip().ne("")]
    participant_counts = human_meta["participant_id"].astype(str).value_counts()
    participants = [participant for participant, count in participant_counts.items() if int(count) >= 2]
    if len(participants) < 2:
        return []

    artifacts: list[ExperimentArtifact] = []
    for participant_id in participants:
        split_ids = _make_human_holdout_split_ids(session_metadata, held_out_participant=participant_id, random_state=random_state)
        if split_ids is None:
            continue
        split_summaries.append(_build_split_summary(session_metadata, split_ids, protocol="leave_one_human_user_out", note=participant_id))
        leakage_frames.append(
            _build_leakage_audit(session_metadata, split_ids, protocol="leave_one_human_user_out", note=participant_id, group_key=group_key)
        )
        artifacts.extend(
            _run_experiment_suite_for_split(
                prefix_df,
                full_session_df,
                session_metadata,
                split_ids=split_ids,
                prefixes=prefixes,
                output_dir=output_dir,
                model_output_dir=output_dir / "models",
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                save_models=False,
                positive_label=positive_label,
                protocol="leave_one_human_user_out",
                selected_models=selected_models,
                selected_ablations=["all_features", "graph_plus_entropy", "entropy_only", "timing_only"],
                notes=f"held_out_participant={participant_id}",
            )
        )
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
    protocol: str,
    notes: str,
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
        artifact_id=f"{_slugify(protocol)}__heuristic_rules__heuristic_baseline{_notes_suffix(notes)}",
        model_name="heuristic_rules",
        ablation_name="heuristic_baseline",
        protocol=protocol,
        threshold=float(threshold_by_prefix.get(max(threshold_by_prefix), default_threshold)),
        threshold_strategy="prefix_specific",
        metrics_by_prefix=metrics_by_prefix,
        predictions=predictions,
        confidence_intervals=ci_df,
        detection_delay_summary=detection_delay,
        notes="entropy+timing+ua heuristics" if not notes else f"{notes}; entropy+timing+ua heuristics",
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
    protocol: str,
    notes: str,
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
        artifact_id=f"{_slugify(protocol)}__static_logistic_regression__static_request_baseline{_notes_suffix(notes)}",
        model_name="static_logistic_regression",
        ablation_name="static_request_baseline",
        protocol=protocol,
        threshold=float(threshold),
        threshold_strategy="global",
        metrics_by_prefix=metrics_by_prefix,
        predictions=predictions,
        confidence_intervals=ci_df,
        detection_delay_summary=detection_delay,
        notes="first-request and header-like features only" if not notes else f"{notes}; first-request and header-like features only",
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
    protocol: str,
    fitted_model,
    test_full_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
    n_bootstrap: int,
    positive_label: str,
    notes: str,
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
        artifact_id=f"{_slugify(protocol)}__{_slugify(model_name)}__full_session_baseline{_notes_suffix(notes)}",
        model_name=model_name,
        ablation_name="full_session_baseline",
        protocol=f"{protocol}_full_session_only",
        threshold=float(threshold),
        threshold_strategy="global",
        metrics_by_prefix=metrics_by_prefix,
        predictions=predictions,
        confidence_intervals=ci_df,
        detection_delay_summary=detection_delay,
        notes="decision allowed only after session completion" if not notes else f"{notes}; decision allowed only after session completion",
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
            "start_timestamp": 0.0,
            "end_timestamp": 0.0,
            "duration_seconds": 0.0,
            "first_path": "",
            "participant_id": "",
            "traffic_family": "",
            "collection_method": "",
            "automation_stack": "",
            "notes": "",
            "referrer_present_ratio": 0.0,
            "num_asset_requests": 0.0,
            "asset_request_ratio": 0.0,
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
        automation_stack = str(first_event.extra.get("automation_stack", "") or "").strip()
        traffic_family = str(first_event.extra.get("traffic_family", "") or "").strip()
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
            "participant_id": str(first_event.extra.get("participant_id", "") or ""),
            "traffic_family": traffic_family or bot_family,
            "collection_method": str(first_event.extra.get("collection_method", "") or ""),
            "automation_stack": automation_stack,
            "notes": str(first_event.extra.get("notes", "") or ""),
        }
        row.update(summary_lookup.get(session.session_id, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_session_metadata_summary(session_metadata: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    if session_metadata.empty or summary_df.empty:
        return session_metadata
    summary = summary_df.copy()
    overlap_columns = [column for column in summary.columns if column != "session_id" and column not in session_metadata.columns]
    if not overlap_columns:
        merge_columns = [column for column in summary.columns if column != "session_id"]
        merged = session_metadata.merge(summary[["session_id", *merge_columns]], on="session_id", how="left", suffixes=("", "_summary"))
        for column in merge_columns:
            summary_column = f"{column}_summary"
            if summary_column in merged.columns:
                if column in merged.columns:
                    merged[column] = merged[column].fillna(merged.pop(summary_column))
                else:
                    merged[column] = merged.pop(summary_column)
        return merged
    return session_metadata.merge(summary[["session_id", *overlap_columns]], on="session_id", how="left")


def _make_standard_split_ids(session_metadata: pd.DataFrame, *, random_state: int, group_key: str = "path_signature") -> dict[str, list[str]]:
    if group_key not in session_metadata.columns:
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
        for _, group in label_meta.groupby(group_key, dropna=False):
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


def _make_time_split_ids(session_metadata: pd.DataFrame) -> dict[str, list[str]] | None:
    if "start_timestamp" not in session_metadata.columns:
        return None
    ordered = session_metadata.sort_values(["start_timestamp", "session_id"]).reset_index(drop=True)
    if len(ordered) < 6:
        return None
    train_end = max(1, int(len(ordered) * 0.60))
    val_end = max(train_end + 1, int(len(ordered) * 0.80))
    split_ids = {
        "train": ordered.iloc[:train_end]["session_id"].tolist(),
        "val": ordered.iloc[train_end:val_end]["session_id"].tolist(),
        "test": ordered.iloc[val_end:]["session_id"].tolist(),
    }
    if any(not ids for ids in split_ids.values()):
        return None
    if any(session_metadata[session_metadata["session_id"].isin(ids)]["label"].nunique() < 2 for ids in split_ids.values()):
        return None
    return split_ids


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


def _make_human_holdout_split_ids(
    session_metadata: pd.DataFrame,
    *,
    held_out_participant: str,
    random_state: int,
) -> dict[str, list[str]] | None:
    human_meta = session_metadata[session_metadata["label"] == "human"].copy()
    held_out_human_ids = human_meta[human_meta["participant_id"] == held_out_participant]["session_id"].tolist()
    remaining_human_ids = human_meta[human_meta["participant_id"] != held_out_participant]["session_id"].tolist()
    bot_ids = session_metadata[session_metadata["label"] == "bot"]["session_id"].tolist()
    if len(held_out_human_ids) < 2 or len(remaining_human_ids) < 4 or len(bot_ids) < 6:
        return None
    human_train, human_val = train_test_split(remaining_human_ids, test_size=0.25, random_state=random_state)
    bot_train, bot_temp = train_test_split(bot_ids, test_size=0.40, random_state=random_state)
    bot_val, bot_test = train_test_split(bot_temp, test_size=0.50, random_state=random_state)
    return {
        "train": sorted(list(human_train) + list(bot_train)),
        "val": sorted(list(human_val) + list(bot_val)),
        "test": sorted(list(held_out_human_ids) + list(bot_test)),
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
                "num_participants": int(subset["participant_id"].fillna("").astype(str).str.strip().replace("", np.nan).dropna().nunique())
                if "participant_id" in subset.columns
                else 0,
                "bot_family_counts": "; ".join(f"{family}:{count}" for family, count in subset[subset["label"] == "bot"]["bot_family"].value_counts().items()),
            }
        )
    return pd.DataFrame(rows)


def _build_leakage_audit(
    session_metadata: pd.DataFrame,
    split_ids: dict[str, list[str]],
    *,
    protocol: str = "session_split",
    note: str = "",
    group_key: str = "path_signature",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
        left = session_metadata[session_metadata["session_id"].isin(split_ids[left_name])].copy()
        right = session_metadata[session_metadata["session_id"].isin(split_ids[right_name])].copy()
        for column in ("session_id", "client_key", "path_signature", "prefix3_signature", "prefix5_signature"):
            overlap = sorted(_normalize_overlap_values(left.get(column)) & _normalize_overlap_values(right.get(column)))
            rows.append(
                {
                    "protocol": protocol,
                    "note": note,
                    "split_pair": f"{left_name}_vs_{right_name}",
                    "key": column,
                    "shared_count": int(len(overlap)),
                    "examples": "; ".join(overlap[:5]),
                }
            )
        if group_key in left.columns and group_key in right.columns:
            grouped_overlap = sorted(_normalize_overlap_values(left.get(group_key)) & _normalize_overlap_values(right.get(group_key)))
            rows.append(
                {
                    "protocol": protocol,
                    "note": note,
                    "split_pair": f"{left_name}_vs_{right_name}",
                    "key": f"group_key:{group_key}",
                    "shared_count": int(len(grouped_overlap)),
                    "examples": "; ".join(grouped_overlap[:5]),
                }
            )
    return pd.DataFrame(rows)


def _build_shortcut_audit(session_metadata: pd.DataFrame, *, full_session_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    merge_columns = ["session_id"]
    merge_columns.extend(
        [
            column
            for column in ("navigation_entropy_score", "navigation_entropy_score_v2")
            if column in full_session_df.columns and column not in merge_columns
        ]
    )
    enriched = session_metadata.merge(
        full_session_df[merge_columns],
        on="session_id",
        how="left",
    )

    for label, group in enriched.groupby("label"):
        rows.extend(
            [
                {"section": "class_summary", "label": label, "metric": "num_sessions", "value": float(len(group))},
                {"section": "class_summary", "label": label, "metric": "mean_num_events", "value": float(group["num_events"].mean()) if "num_events" in group.columns else float("nan")},
                {"section": "class_summary", "label": label, "metric": "mean_unique_paths", "value": float(group["unique_paths"].mean()) if "unique_paths" in group.columns else float("nan")},
                {
                    "section": "class_summary",
                    "label": label,
                    "metric": "mean_referrer_present_ratio",
                    "value": float(group["referrer_present_ratio"].mean()) if "referrer_present_ratio" in group.columns else float("nan"),
                },
                {
                    "section": "class_summary",
                    "label": label,
                    "metric": "mean_asset_request_ratio",
                    "value": float(group["asset_request_ratio"].mean()) if "asset_request_ratio" in group.columns else float("nan"),
                },
                {
                    "section": "class_summary",
                    "label": label,
                    "metric": "share_suspect_user_agent",
                    "value": float(group["has_suspect_user_agent"].mean()) if "has_suspect_user_agent" in group.columns else float("nan"),
                },
                {
                    "section": "class_summary",
                    "label": label,
                    "metric": "mean_navigation_entropy_score_v2",
                    "value": float(group["navigation_entropy_score_v2"].mean()) if "navigation_entropy_score_v2" in group.columns else float("nan"),
                },
            ]
        )
        for column in ("ua_family", "traffic_family", "first_path"):
            if column not in group.columns:
                continue
            top_values = group[column].fillna("missing").astype(str).value_counts().head(5)
            for value, count in top_values.items():
                rows.append(
                    {
                        "section": f"top_{column}",
                        "label": label,
                        "metric": value,
                        "value": float(count),
                    }
                )
    return pd.DataFrame(rows)


def _build_shortcut_red_flags(
    session_metadata: pd.DataFrame,
    *,
    full_session_df: pd.DataFrame,
    positive_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if session_metadata.empty:
        return pd.DataFrame(rows)
    enriched = session_metadata.copy()
    if not full_session_df.empty:
        merge_columns = ["session_id"]
        merge_columns.extend(
            [column for column in ("navigation_entropy_score", "navigation_entropy_score_v2") if column in full_session_df.columns]
        )
        enriched = enriched.merge(full_session_df[merge_columns], on="session_id", how="left")

    y_true = (enriched["label"] == positive_label).astype(int).to_numpy()
    numeric_candidates = [
        "has_suspect_user_agent",
        "ua_browser_like",
        "referrer_present_ratio",
        "asset_request_ratio",
        "num_asset_requests",
        "num_page_requests",
        "num_events",
        "unique_paths",
        "mean_delta_t",
        "first_path_depth",
    ]
    for column in numeric_candidates:
        if column not in enriched.columns:
            continue
        values = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0).to_numpy()
        if len(np.unique(values)) < 2 or len(np.unique(y_true)) < 2:
            continue
        auc = _safe_shortcut_auc(y_true, values)
        human_values = values[y_true == 0]
        bot_values = values[y_true == 1]
        rows.append(
            {
                "cue_type": "numeric",
                "cue_name": column,
                "support": int(len(values)),
                "signal_strength": float(auc),
                "human_mean": float(np.mean(human_values)) if len(human_values) else float("nan"),
                "bot_mean": float(np.mean(bot_values)) if len(bot_values) else float("nan"),
                "flag_level": _shortcut_flag_level(auc),
                "details": "single-feature separability",
            }
        )

    categorical_candidates = ["ua_family", "first_path", "first_category", "collection_method", "automation_stack", "traffic_family"]
    for column in categorical_candidates:
        if column not in enriched.columns:
            continue
        grouped = enriched.groupby(column, dropna=False)
        for value, group in grouped:
            support = int(len(group))
            if support < 3:
                continue
            bot_rate = float((group["label"] == positive_label).mean())
            purity = max(bot_rate, 1.0 - bot_rate)
            rows.append(
                {
                    "cue_type": "categorical",
                    "cue_name": f"{column}={value}",
                    "support": support,
                    "signal_strength": purity,
                    "human_mean": float("nan"),
                    "bot_mean": float("nan"),
                    "flag_level": _shortcut_flag_level(purity),
                    "details": f"class_purity={purity:.3f}; bot_rate={bot_rate:.3f}",
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["flag_level", "signal_strength", "support"], ascending=[True, False, False]).reset_index(drop=True)
    return result


def _build_entropy_variant_comparison(
    feature_df: pd.DataFrame,
    *,
    hard_prefixes: Iterable[int],
    entropy_variants: Iterable[str],
    positive_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if feature_df.empty:
        return pd.DataFrame(rows)
    working = feature_df[feature_df["label"] != DEFAULT_UNKNOWN_LABEL].copy()
    if working.empty:
        return pd.DataFrame(rows)
    hard_prefixes = sorted(set(int(prefix) for prefix in hard_prefixes))
    candidate_columns = [column for column in entropy_variants if column in working.columns]
    for column in candidate_columns:
        prefix_scores: list[float] = []
        for prefix_len in hard_prefixes:
            subset = working[working["prefix_len"] == prefix_len].copy()
            if subset.empty:
                continue
            y_true = (subset["label"] == positive_label).astype(int).to_numpy()
            values = pd.to_numeric(subset[column], errors="coerce").fillna(0.0).to_numpy()
            if len(np.unique(y_true)) < 2:
                roc_auc = float("nan")
            else:
                try:
                    from sklearn.metrics import roc_auc_score

                    roc_auc = float(roc_auc_score(y_true, values))
                except ValueError:
                    roc_auc = float("nan")
            human_values = values[y_true == 0]
            bot_values = values[y_true == 1]
            rows.append(
                {
                    "feature": column,
                    "prefix_len": prefix_len,
                    "roc_auc": roc_auc,
                    "human_mean": float(np.mean(human_values)) if len(human_values) else float("nan"),
                    "bot_mean": float(np.mean(bot_values)) if len(bot_values) else float("nan"),
                    "absolute_gap": float(abs(np.mean(bot_values) - np.mean(human_values))) if len(human_values) and len(bot_values) else float("nan"),
                }
            )
            if not np.isnan(roc_auc):
                prefix_scores.append(roc_auc)
        if prefix_scores:
            rows.append(
                {
                    "feature": column,
                    "prefix_len": "mean",
                    "roc_auc": float(np.mean(prefix_scores)),
                    "human_mean": float("nan"),
                    "bot_mean": float("nan"),
                    "absolute_gap": float("nan"),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        sort_key = result["prefix_len"].map(lambda value: 999 if value == "mean" else int(value))
        result = result.assign(_sort_key=sort_key).sort_values(["_sort_key", "roc_auc", "absolute_gap"], ascending=[True, False, False]).drop(columns="_sort_key")
    return result


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
    shortcut_audit: pd.DataFrame | None = None,
    entropy_comparison: pd.DataFrame | None = None,
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
        "- split leakage auditing, shortcut audits, and holdout evaluation",
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

    if shortcut_audit is not None and not shortcut_audit.empty:
        lines.extend(
            [
                "## Shortcut audit",
                "",
                _df_to_markdown(shortcut_audit),
                "",
            ]
        )

    shortcut_red_flags = _read_if_exists(path.parent / "shortcut_red_flags.csv")
    if shortcut_red_flags is not None and not shortcut_red_flags.empty:
        lines.extend(
            [
                "## Shortcut red flags",
                "",
                _df_to_markdown(shortcut_red_flags.head(25)),
                "",
            ]
        )

    if entropy_comparison is not None and not entropy_comparison.empty:
        lines.extend(
            [
                "## Entropy variant comparison",
                "",
                _df_to_markdown(entropy_comparison),
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
        easy_rows = baseline_rows[(baseline_rows["f1"] >= 0.999) & (baseline_rows["recall"] >= 0.999)]
        if not easy_rows.empty:
            lines.extend(
                [
                    "## Dataset warning",
                    "",
                    "Some baseline runs are still effectively perfect. Treat those results as evidence that the dataset may still be easier than the final thesis setting.",
                    "",
                ]
            )
    if shortcut_red_flags is not None and not shortcut_red_flags.empty:
        high_flags = shortcut_red_flags[shortcut_red_flags["flag_level"] == "high"]
        if not high_flags.empty:
            lines.extend(
                [
                    "## Leakage warning",
                    "",
                    "Some individual shortcut cues show very strong class separability on their own. Inspect `shortcut_red_flags.csv` before making strong claims about graph or entropy novelty.",
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



def _maybe_plot(
    artifacts: list[ExperimentArtifact],
    output_dir: Path,
    *,
    feature_df: pd.DataFrame | None = None,
    hard_prefixes: Iterable[int] | None = None,
) -> None:
    if plt is None or not artifacts:
        return

    leaderboard = build_leaderboard(artifacts)
    if leaderboard.empty:
        return
    hard_prefixes = sorted(set(int(prefix) for prefix in (hard_prefixes or DEFAULT_HARD_PREFIXES)))

    standard_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.protocol == "session_split" and not artifact.metrics_by_prefix.empty and artifact.metrics_by_prefix["prefix_len"].min() >= 0
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

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        for model_name, ablation_name, notes in top_configs:
            artifact = next(
                item
                for item in standard_artifacts
                if item.model_name == model_name and item.ablation_name == ablation_name and item.notes == notes
            )
            ax.plot(artifact.metrics_by_prefix["prefix_len"], artifact.metrics_by_prefix["recall"], marker="o", label=f"{model_name} | {ablation_name}")
        ax.set_xlabel("Prefix length")
        ax.set_ylabel("Recall")
        ax.set_title("Early-detection recall by prefix length")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "plot_recall_by_prefix.png", dpi=150)
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

    if standard_artifacts:
        best = next(
            artifact
            for artifact in standard_artifacts
            if artifact.model_name == standard_leaderboard.iloc[0]["model_name"]
            and artifact.ablation_name == standard_leaderboard.iloc[0]["ablation_name"]
            and artifact.notes == standard_leaderboard.iloc[0]["notes"]
        )
        for prefix_len in hard_prefixes[:2]:
            subset = best.predictions[best.predictions["prefix_len"] == prefix_len].copy()
            if subset.empty:
                continue
            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(1, 1, 1)
            human_scores = subset.loc[subset["label"] == "human", "bot_probability"].to_numpy()
            bot_scores = subset.loc[subset["label"] == "bot", "bot_probability"].to_numpy()
            bins = np.linspace(0.0, 1.0, 16)
            if len(human_scores):
                ax.hist(human_scores, bins=bins, alpha=0.6, label="human")
            if len(bot_scores):
                ax.hist(bot_scores, bins=bins, alpha=0.6, label="bot")
            ax.set_xlabel("Bot probability")
            ax.set_ylabel("Count")
            ax.set_title(f"Score distribution at prefix {prefix_len}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"plot_score_hist_prefix_{prefix_len}.png", dpi=150)
            plt.close(fig)

        detection_rows = leaderboard[
            leaderboard["mean_first_detected_prefix_bot"].notna()
            & leaderboard["protocol"].isin(["session_split", "hard_prefix_session_split", "time_split"])
        ].head(10)
        if not detection_rows.empty:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 1, 1)
            labels = [f"{row.model_name}\n{row.ablation_name}\n{row.protocol}" for row in detection_rows.itertuples(index=False)]
            ax.bar(range(len(detection_rows)), detection_rows["mean_first_detected_prefix_bot"].tolist())
            ax.set_xticks(range(len(detection_rows)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Mean first detected prefix (bots)")
            ax.set_title("Detection speed comparison")
            fig.tight_layout()
            fig.savefig(output_dir / "plot_mean_first_detected_prefix.png", dpi=150)
            plt.close(fig)

        trajectory_rows = []
        human_subset = best.predictions[best.predictions["label"] == "human"].copy()
        bot_subset = best.predictions[best.predictions["label"] == "bot"].copy()
        if not human_subset.empty:
            human_session = str(human_subset.groupby("session_id")["bot_probability"].mean().idxmin())
            trajectory = human_subset[human_subset["session_id"] == human_session].copy()
            trajectory["series_label"] = f"human:{human_session}"
            trajectory_rows.append(trajectory)
        if not bot_subset.empty:
            bot_means = bot_subset.groupby("session_id")["bot_probability"].mean().sort_values()
            simple_bot_session = str(bot_means.index[-1])
            trajectory = bot_subset[bot_subset["session_id"] == simple_bot_session].copy()
            trajectory["series_label"] = f"bot_high:{simple_bot_session}"
            trajectory_rows.append(trajectory)
            if len(bot_means) >= 2:
                harder_bot_session = str(bot_means.index[len(bot_means) // 2])
                trajectory = bot_subset[bot_subset["session_id"] == harder_bot_session].copy()
                trajectory["series_label"] = f"bot_mid:{harder_bot_session}"
                trajectory_rows.append(trajectory)
        if trajectory_rows:
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(1, 1, 1)
            combined = pd.concat(trajectory_rows, ignore_index=True)
            for session_id, group in combined.groupby("series_label"):
                ax.plot(group["prefix_len"], group["bot_probability"], marker="o", label=session_id)
            ax.set_xlabel("Prefix length")
            ax.set_ylabel("Bot probability")
            ax.set_title("Example score trajectories")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "plot_score_trajectories.png", dpi=150)
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


def _safe_shortcut_auc(y_true: np.ndarray, values: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, values))
    except ValueError:
        return float("nan")
    if np.isnan(auc):
        return auc
    return max(auc, 1.0 - auc)


def _shortcut_flag_level(value: float) -> str:
    if np.isnan(value):
        return "low"
    if value >= 0.95:
        return "high"
    if value >= 0.85:
        return "medium"
    return "low"


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
    if "playwright" in lowered:
        return "playwright"
    if "selenium" in lowered or "webdriver" in lowered:
        return "selenium"
    if "researchscraper" in lowered:
        return "research_scraper"
    if "researchstealth" in lowered or "stealth-browser" in lowered or "hybridcrawler" in lowered or "browsernoise" in lowered:
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
        ("browser_hybrid", r"hybridcrawler|browser_hybrid"),
        ("browser_noise", r"browsernoise|browser_noise"),
        ("playwright_browser", r"playwright"),
        ("selenium_browser", r"selenium|webdriver"),
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
