"""Research experiment runner with ablations, threshold tuning, and report outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from .config import DEFAULT_POSITIVE_LABEL, DEFAULT_UNKNOWN_LABEL
from .evaluation import attach_metric_confidence_intervals, compute_binary_metrics, tune_threshold
from .features import extract_prefix_feature_rows, prefix_rows_to_dataframe
from .graph_builder import build_graph_from_csv
from .modeling import build_models, make_model_bundle, save_model_bundle, summarize_detection_delay
from .sessionizer import load_sessions_from_csv

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


@dataclass(slots=True)
class ExperimentArtifact:
    model_name: str
    ablation_name: str
    threshold: float
    metrics_by_prefix: pd.DataFrame
    predictions: pd.DataFrame
    confidence_intervals: pd.DataFrame
    detection_delay_summary: pd.DataFrame



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

    feature_df = _load_or_build_feature_df(data_dir, prefixes=args.prefixes)
    artifacts = run_experiments(
        feature_df,
        prefixes=args.prefixes,
        output_dir=output_dir,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        save_models=args.save_models,
    )

    leaderboard = build_leaderboard(artifacts)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    _write_markdown_summary(output_dir / "summary.md", leaderboard, artifacts)
    _maybe_plot(artifacts, output_dir)
    print(f"Experiment outputs written to: {output_dir.resolve()}")
    print(leaderboard.to_string(index=False))



def _load_or_build_feature_df(data_dir: Path, prefixes: Iterable[int]) -> pd.DataFrame:
    feature_path = data_dir / "prefix_features.csv"
    if feature_path.exists():
        return pd.read_csv(feature_path)
    graph = build_graph_from_csv(data_dir / "graph_edges.csv", data_dir / "graph_categories.csv")
    sessions = load_sessions_from_csv(data_dir / "requests.csv")
    feature_rows = extract_prefix_feature_rows(sessions, graph, prefixes=prefixes)
    feature_df = prefix_rows_to_dataframe(feature_rows)
    feature_df.to_csv(feature_path, index=False)
    return feature_df



def run_experiments(
    feature_df: pd.DataFrame,
    *,
    prefixes: Iterable[int],
    output_dir: Path,
    n_bootstrap: int,
    random_state: int,
    save_models: bool,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
) -> list[ExperimentArtifact]:
    labeled_df = feature_df[feature_df["label"] != DEFAULT_UNKNOWN_LABEL].copy()
    if labeled_df.empty:
        raise ValueError("No labeled rows are available for experiments")

    session_ids = labeled_df["session_id"].drop_duplicates().tolist()
    if len(session_ids) < 6:
        raise ValueError("Need at least six sessions for train/validation/test experiments")

    train_ids, temp_ids = train_test_split(session_ids, test_size=0.40, random_state=random_state)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=random_state)

    train_df = labeled_df[labeled_df["session_id"].isin(train_ids)].copy()
    val_df = labeled_df[labeled_df["session_id"].isin(val_ids)].copy()
    test_df = labeled_df[labeled_df["session_id"].isin(test_ids)].copy()

    models = build_models(random_state=random_state)
    artifacts: list[ExperimentArtifact] = []
    model_output_dir = output_dir / "models"
    if save_models:
        model_output_dir.mkdir(parents=True, exist_ok=True)

    for ablation_name, families in ABLATIONS.items():
        feature_columns = _feature_columns_for_families(train_df.columns.tolist(), families)
        if not feature_columns:
            continue
        train_full = _full_session_rows(train_df)
        val_full = _full_session_rows(val_df)

        X_train = train_full[feature_columns]
        y_train = (train_full["label"] == positive_label).astype(int)
        X_val = val_full[feature_columns]
        y_val = (val_full["label"] == positive_label).astype(int)

        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            val_scores = fitted.predict_proba(X_val)[:, 1]
            threshold = tune_threshold(y_val.to_numpy(), val_scores)

            predictions = _predict_across_prefixes(
                fitted,
                test_df,
                prefixes=prefixes,
                feature_columns=feature_columns,
                threshold=threshold,
                positive_label=positive_label,
            )
            metrics_by_prefix = _metrics_by_prefix(predictions, threshold=threshold, positive_label=positive_label)
            metrics_by_prefix.to_csv(output_dir / f"metrics_{model_name}_{ablation_name}.csv", index=False)
            predictions.to_csv(output_dir / f"predictions_{model_name}_{ablation_name}.csv", index=False)
            ci_df = attach_metric_confidence_intervals(
                predictions,
                threshold=threshold,
                n_bootstrap=n_bootstrap,
                positive_label=positive_label,
            )
            ci_df.to_csv(output_dir / f"confidence_intervals_{model_name}_{ablation_name}.csv", index=False)
            detection_delay = summarize_detection_delay(predictions, positive_label=positive_label)
            detection_delay.to_csv(output_dir / f"detection_delay_{model_name}_{ablation_name}.csv", index=False)

            artifact = ExperimentArtifact(
                model_name=model_name,
                ablation_name=ablation_name,
                threshold=threshold,
                metrics_by_prefix=metrics_by_prefix,
                predictions=predictions,
                confidence_intervals=ci_df,
                detection_delay_summary=detection_delay,
            )
            artifacts.append(artifact)

            if save_models:
                bundle = make_model_bundle(
                    type("ArtifactShim", (), {"model_name": f"{model_name}_{ablation_name}", "feature_columns": feature_columns, "model": fitted})(),
                    threshold=threshold,
                )
                save_model_bundle(bundle, model_output_dir / f"{model_name}_{ablation_name}_bundle.pkl")

    return artifacts



def build_leaderboard(artifacts: list[ExperimentArtifact]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for artifact in artifacts:
        metrics_df = artifact.metrics_by_prefix.copy()
        if metrics_df.empty:
            continue
        final_prefix = int(metrics_df["prefix_len"].max())
        final_row = metrics_df[metrics_df["prefix_len"] == final_prefix].iloc[0]
        bot_delay = artifact.detection_delay_summary
        bot_delay_row = bot_delay[bot_delay["label"] == "bot"]
        mean_detection_prefix = float(bot_delay_row["mean_first_detected_prefix"].iloc[0]) if not bot_delay_row.empty else float("nan")
        detection_rate = float(bot_delay_row["detection_rate"].iloc[0]) if not bot_delay_row.empty else float("nan")
        rows.append(
            {
                "model_name": artifact.model_name,
                "ablation_name": artifact.ablation_name,
                "threshold": artifact.threshold,
                "final_prefix": final_prefix,
                "accuracy": float(final_row["accuracy"]),
                "precision": float(final_row["precision"]),
                "recall": float(final_row["recall"]),
                "f1": float(final_row["f1"]),
                "roc_auc": float(final_row.get("roc_auc", float("nan"))),
                "pr_auc": float(final_row.get("pr_auc", float("nan"))),
                "bot_detection_rate": detection_rate,
                "mean_first_detected_prefix_bot": mean_detection_prefix,
            }
        )
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(["f1", "roc_auc", "pr_auc"], ascending=False)
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
    threshold: float,
    positive_label: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for prefix_len in sorted(set(int(x) for x in prefixes)):
        slice_df = test_df[test_df["prefix_len"] == prefix_len].copy()
        if slice_df.empty:
            continue
        scores = model.predict_proba(slice_df[feature_columns])[:, 1]
        slice_df["bot_probability"] = scores
        slice_df["predicted_bot"] = (scores >= threshold).astype(int)
        rows.append(slice_df[["session_id", "prefix_len", "label", "bot_probability", "predicted_bot"]])
    if not rows:
        return pd.DataFrame(columns=["session_id", "prefix_len", "label", "bot_probability", "predicted_bot"])
    return pd.concat(rows, ignore_index=True)



def _metrics_by_prefix(predictions: pd.DataFrame, *, threshold: float, positive_label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for prefix_len, group in predictions.groupby("prefix_len"):
        y_true = (group["label"] == positive_label).astype(int).to_numpy()
        y_score = group["bot_probability"].to_numpy()
        y_pred = (y_score >= threshold).astype(int)
        metrics = compute_binary_metrics(y_true, y_pred, y_score)
        row: dict[str, float | int] = {"prefix_len": int(prefix_len), **metrics}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("prefix_len")



def _write_markdown_summary(path: Path, leaderboard: pd.DataFrame, artifacts: list[ExperimentArtifact]) -> None:
    lines: list[str] = [
        "# Experiment Summary",
        "",
        "This report was generated automatically from the research experiment runner.",
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
            f"- Model: **{top['model_name']}**",
            f"- Feature set: **{top['ablation_name']}**",
            f"- Tuned threshold: **{top['threshold']:.2f}**",
            f"- Final-prefix F1: **{top['f1']:.3f}**",
            f"- Final-prefix ROC-AUC: **{top['roc_auc']:.3f}**",
            f"- Mean first detection prefix for bots: **{top['mean_first_detected_prefix_bot']:.2f}**",
            "",
            "## Leaderboard",
            "",
            _df_to_markdown(leaderboard),
            "",
        ]
    )

    # Add short CI section for the top artifact.
    for artifact in artifacts:
        if artifact.model_name == top["model_name"] and artifact.ablation_name == top["ablation_name"]:
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



def _maybe_plot(artifacts: list[ExperimentArtifact], output_dir: Path) -> None:
    if plt is None or not artifacts:
        return

    leaderboard = build_leaderboard(artifacts)
    if leaderboard.empty:
        return

    # Plot 1: F1 by prefix for the top 3 experiment configurations.
    top_configs = leaderboard.head(3)[["model_name", "ablation_name"]].to_records(index=False)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    for model_name, ablation_name in top_configs:
        artifact = next(a for a in artifacts if a.model_name == model_name and a.ablation_name == ablation_name)
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
    x_labels = [f"{row.model_name}\n{row.ablation_name}" for row in leaderboard.itertuples(index=False)]
    ax.bar(range(len(leaderboard)), leaderboard["f1"].tolist())
    ax.set_xticks(range(len(leaderboard)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel("Final-prefix F1")
    ax.set_title("Experiment leaderboard")
    fig.tight_layout()
    fig.savefig(output_dir / "plot_leaderboard_f1.png", dpi=150)
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


if __name__ == "__main__":
    main()
