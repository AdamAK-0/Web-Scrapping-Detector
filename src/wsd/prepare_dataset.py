"""CLI for building a research dataset from raw web-server logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .graph_builder import build_graph_from_sessions, save_graph_to_csv
from .labeling import apply_session_labels
from .log_parsers import read_raw_logs
from .sessionizer import build_sessions_from_dataframe, sessions_to_dataframe, summarize_sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a trainable navigation dataset from raw logs")
    parser.add_argument("--input-path", required=True, help="Raw input log file (CSV, JSONL, or Nginx combined log)")
    parser.add_argument("--format", default="auto", choices=["auto", "csv", "jsonl", "nginx_combined"], help="Raw log format")
    parser.add_argument("--output-dir", default="data/prepared_run", help="Output directory for normalized requests, graph files, and summaries")
    parser.add_argument("--manual-labels", default=None, help="Optional CSV with client_key,label or session_id,label")
    parser.add_argument("--session-timeout", type=float, default=30 * 60, help="Inactivity timeout used to split sessions, in seconds")
    parser.add_argument("--min-session-length", type=int, default=2, help="Minimum number of page requests kept per session")
    parser.add_argument("--keep-assets", action="store_true", help="Keep asset requests instead of filtering to page-like paths")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_raw_logs(args.input_path, log_format=args.format)
    raw_df.to_csv(output_dir / "normalized_requests.csv", index=False)

    all_request_sessions = build_sessions_from_dataframe(
        raw_df,
        session_timeout_seconds=args.session_timeout,
        drop_asset_requests=False,
        min_session_length=1,
    )
    sessions = build_sessions_from_dataframe(
        raw_df,
        session_timeout_seconds=args.session_timeout,
        drop_asset_requests=not args.keep_assets,
        min_session_length=args.min_session_length,
    )
    summary_df = apply_session_labels(sessions, manual_labels_path=args.manual_labels)
    if all_request_sessions:
        all_request_summary = summarize_sessions(all_request_sessions)
        enrich_columns = [
            "session_id",
            "start_timestamp",
            "end_timestamp",
            "first_path",
            "referrer_present_ratio",
            "num_page_requests",
            "num_asset_requests",
            "asset_request_ratio",
            "has_asset_requests",
        ]
        extra_summary = all_request_summary[[column for column in enrich_columns if column in all_request_summary.columns]].copy()
        if not extra_summary.empty:
            summary_df = summary_df.merge(extra_summary, on="session_id", how="left", suffixes=("", "_all_requests"))
    session_df = sessions_to_dataframe(sessions)
    session_df.to_csv(output_dir / "requests.csv", index=False)
    summary_df.to_csv(output_dir / "session_summary.csv", index=False)

    graph = build_graph_from_sessions(sessions)
    save_graph_to_csv(graph, output_dir / "graph_edges.csv", output_dir / "graph_categories.csv")

    labeled_counts = summary_df["proposed_label"].value_counts(dropna=False).to_dict() if not summary_df.empty else {}
    print(f"Prepared dataset written to: {output_dir.resolve()}")
    print(f"Sessions: {len(sessions)}")
    print(f"Proposed label counts: {labeled_counts}")


if __name__ == "__main__":
    main()
