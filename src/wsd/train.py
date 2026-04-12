"""CLI entrypoint for generating data, training models, and exporting results."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_PREFIXES
from .features import extract_prefix_feature_rows, prefix_rows_to_dataframe
from .graph_builder import build_graph_from_csv
from .modeling import make_model_bundle, save_model_bundle, train_and_evaluate_by_prefix
from .sessionizer import load_sessions_from_csv
from .synthetic_data import generate_synthetic_dataset, save_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the web scraping detector prototype")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate a fresh synthetic dataset before training")
    parser.add_argument("--output-dir", default="data/synthetic_run", help="Output directory used when generating synthetic data")
    parser.add_argument("--data-dir", default="data/synthetic_run", help="Directory containing graph_edges.csv, graph_categories.csv, and requests.csv")
    parser.add_argument("--prefixes", nargs="*", type=int, default=DEFAULT_PREFIXES, help="Prefix lengths used for online evaluation")
    parser.add_argument("--save-model-dir", default=None, help="Optional directory where trained .pkl bundles will be stored")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.generate_synthetic:
        dataset = generate_synthetic_dataset()
        output_dir = Path(args.output_dir)
        save_synthetic_dataset(dataset, output_dir)
        data_dir = output_dir
        print(f"Synthetic dataset written to: {output_dir.resolve()}")

    graph = build_graph_from_csv(data_dir / "graph_edges.csv", data_dir / "graph_categories.csv")
    sessions = load_sessions_from_csv(data_dir / "requests.csv")
    feature_rows = extract_prefix_feature_rows(sessions, graph, prefixes=args.prefixes)
    feature_df = prefix_rows_to_dataframe(feature_rows)
    feature_df.to_csv(data_dir / "prefix_features.csv", index=False)

    artifacts = train_and_evaluate_by_prefix(feature_df, prefixes=args.prefixes)

    model_dir = Path(args.save_model_dir) if args.save_model_dir else data_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for model_name, artifact in artifacts.items():
        eval_df = _evaluations_to_dataframe(artifact.evaluations)
        eval_df.to_csv(data_dir / f"metrics_{model_name}.csv", index=False)
        artifact.detection_delay_summary.to_csv(data_dir / f"detection_delay_{model_name}.csv", index=False)
        save_model_bundle(make_model_bundle(artifact), model_dir / f"{model_name}_bundle.pkl")
        print(f"\n=== {model_name} ===")
        print(eval_df.to_string(index=False))
        print("\nDetection delay summary:")
        print(artifact.detection_delay_summary.to_string(index=False))
        print(f"Saved model bundle: {(model_dir / f'{model_name}_bundle.pkl').resolve()}")


def _evaluations_to_dataframe(evaluations):
    import pandas as pd

    rows = []
    for item in evaluations:
        row = {"prefix_len": item.prefix_len}
        row.update(item.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
