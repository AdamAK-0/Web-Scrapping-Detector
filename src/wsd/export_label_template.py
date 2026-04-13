"""Export a session annotation template from a prepared dataset summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = [
    "session_id",
    "client_key",
    "label",
    "participant_id",
    "traffic_family",
    "collection_method",
    "automation_stack",
    "notes",
    "num_events",
    "duration_seconds",
    "first_path",
    "start_timestamp",
    "user_agent",
    "proposed_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a manual session-annotation template from session_summary.csv")
    parser.add_argument("--session-summary", required=True, help="Path to session_summary.csv from wsd.prepare_dataset")
    parser.add_argument("--output-path", required=True, help="Where to write the annotation template CSV")
    parser.add_argument("--only-unknown", action="store_true", help="Only export sessions whose proposed_label is unknown")
    parser.add_argument("--keep-existing-labels", action="store_true", help="Keep existing label values instead of blanking them for annotation")
    return parser.parse_args()


def build_annotation_template(
    session_summary: pd.DataFrame,
    *,
    only_unknown: bool = False,
    keep_existing_labels: bool = False,
) -> pd.DataFrame:
    working = session_summary.copy()
    if only_unknown and "proposed_label" in working.columns:
        working = working[working["proposed_label"].fillna("").astype(str).str.lower() == "unknown"].copy()
    if "label" not in working.columns:
        working["label"] = ""
    if not keep_existing_labels:
        working["label"] = ""
    for column in DEFAULT_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    return working[DEFAULT_COLUMNS].sort_values(["participant_id", "session_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    session_summary = pd.read_csv(args.session_summary)
    template = build_annotation_template(
        session_summary,
        only_unknown=args.only_unknown,
        keep_existing_labels=args.keep_existing_labels,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    print(f"Annotation template written to: {output_path.resolve()}")
    print(f"Sessions exported: {len(template)}")


if __name__ == "__main__":
    main()
