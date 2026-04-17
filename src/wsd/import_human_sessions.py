"""Import archived real-human Nginx logs into the live lab dataset."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .lab_traffic import LABEL_COLUMNS, merge_labels
from .log_parsers import NginxCombinedRegex, read_raw_logs
from .sessionizer import build_sessions_from_dataframe, sessions_to_dataframe, summarize_sessions
from .types import Session

NONHUMAN_PREVIEW_AGENT_PATTERNS = [
    r"bot",
    r"crawler",
    r"spider",
    r"slurp",
    r"google-read-aloud",
    r"whatsapp",
    r"facebookexternalhit",
    r"discordbot",
    r"slackbot",
    r"telegrambot",
    r"linkedinbot",
    r"twitterbot",
]


@dataclass(frozen=True)
class ImportResult:
    archive_dir: Path
    raw_log_copy: Path
    manifest_path: Path
    requests_path: Path
    labels_snapshot_path: Path
    live_log_path: Path
    labels_path: Path
    num_sessions: int
    num_requests: int
    num_filtered_requests: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive and import real human browsing sessions from an Nginx log")
    parser.add_argument(
        "--input-log",
        required=True,
        nargs="+",
        help="Collected human-only log file(s), for example logs.txt or archived raw_human_sessions.log files",
    )
    parser.add_argument("--format", default="auto", choices=["auto", "csv", "jsonl", "nginx_combined"], help="Input log format")
    parser.add_argument("--archive-dir", default="data/human_session_archives", help="Directory where raw copies and manifests are stored")
    parser.add_argument("--live-log", default="data/live_logs/access.log", help="Live access log path used by prepare_live_dataset.bat")
    parser.add_argument("--labels-path", default="data/live_labels/manual_labels.csv", help="Manual label CSV to update")
    parser.add_argument("--session-timeout", type=float, default=30 * 60, help="Seconds of inactivity before a new session")
    parser.add_argument("--min-session-length", type=int, default=2, help="Minimum page requests required for an imported session")
    parser.add_argument("--participant-prefix", default="human_manual", help="Prefix used for generated participant IDs")
    parser.add_argument("--traffic-family", default="human_navigation", help="Metadata traffic_family value")
    parser.add_argument("--collection-method", default="manual_browser", help="Metadata collection_method value")
    parser.add_argument("--automation-stack", default="browser", help="Metadata automation_stack value")
    parser.add_argument("--notes", default="imported real human browsing log", help="Metadata notes value")
    parser.add_argument(
        "--keep-preview-agents",
        action="store_true",
        help="Keep link-preview/crawler user agents such as WhatsApp or Google Read Aloud in the imported human set",
    )
    parser.add_argument(
        "--append-live-log",
        action="store_true",
        help="Append the source log to the live log instead of replacing it. Use after reset unless you know session IDs cannot shift.",
    )
    return parser.parse_args()


def import_human_log(
    *,
    input_log: str | Path,
    log_format: str = "auto",
    archive_dir: str | Path = "data/human_session_archives",
    live_log: str | Path = "data/live_logs/access.log",
    labels_path: str | Path = "data/live_labels/manual_labels.csv",
    session_timeout: float = 30 * 60,
    min_session_length: int = 2,
    participant_prefix: str = "human_manual",
    traffic_family: str = "human_navigation",
    collection_method: str = "manual_browser",
    automation_stack: str = "browser",
    notes: str = "imported real human browsing log",
    exclude_preview_agents: bool = True,
    append_live_log: bool = False,
) -> ImportResult:
    source_path = Path(input_log)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    archive_root = Path(archive_dir)
    import_id = datetime.now(UTC).strftime("human_%Y%m%d_%H%M%S")
    run_archive_dir = archive_root / import_id
    run_archive_dir.mkdir(parents=True, exist_ok=True)

    raw_log_copy = run_archive_dir / _archive_log_name(source_path)
    shutil.copy2(source_path, raw_log_copy)

    all_raw_df = read_raw_logs(source_path, log_format=log_format)
    raw_df = _filter_preview_agents(all_raw_df) if exclude_preview_agents else all_raw_df.copy()
    filtered_requests = len(all_raw_df) - len(raw_df)
    sessions = build_sessions_from_dataframe(
        raw_df,
        session_timeout_seconds=session_timeout,
        min_session_length=min_session_length,
    )
    _attach_human_metadata(
        sessions,
        participant_prefix=participant_prefix,
        traffic_family=traffic_family,
        collection_method=collection_method,
        automation_stack=automation_stack,
        notes=notes,
    )

    all_normalized_path = run_archive_dir / "normalized_requests_all.csv"
    all_raw_df.to_csv(all_normalized_path, index=False)

    normalized_path = run_archive_dir / "normalized_requests.csv"
    raw_df.to_csv(normalized_path, index=False)

    requests_path = run_archive_dir / "identified_requests.csv"
    sessions_to_dataframe(sessions).to_csv(requests_path, index=False)

    manifest = build_session_manifest(sessions)
    manifest_path = run_archive_dir / "identified_human_sessions.csv"
    manifest.to_csv(manifest_path, index=False)

    label_rows = [
        {
            "client_key": "",
            "session_id": str(row["session_id"]),
            "label": "human",
            "participant_id": str(row["participant_id"]),
            "traffic_family": traffic_family,
            "collection_method": collection_method,
            "automation_stack": automation_stack,
            "notes": notes,
        }
        for row in manifest.to_dict(orient="records")
    ]

    label_file = Path(labels_path)
    merge_labels(label_file, label_rows)
    labels_snapshot_path = run_archive_dir / "manual_labels_imported.csv"
    _read_label_rows(label_file).to_csv(labels_snapshot_path, index=False)

    live_log_path = Path(live_log)
    live_log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_live_log(
        source_path=source_path,
        live_log_path=live_log_path,
        append_live_log=append_live_log,
        exclude_preview_agents=exclude_preview_agents,
    )

    summary_path = run_archive_dir / "import_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"input_log={source_path.resolve()}",
                f"raw_log_copy={raw_log_copy.resolve()}",
                f"live_log={live_log_path.resolve()}",
                f"labels_path={label_file.resolve()}",
                f"sessions={len(sessions)}",
                f"normalized_requests={len(raw_df)}",
                f"filtered_preview_requests={filtered_requests}",
                f"mode={'append' if append_live_log else 'replace'}",
                "assumption=source log contains real human browsing sessions only",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return ImportResult(
        archive_dir=run_archive_dir,
        raw_log_copy=raw_log_copy,
        manifest_path=manifest_path,
        requests_path=requests_path,
        labels_snapshot_path=labels_snapshot_path,
        live_log_path=live_log_path,
        labels_path=label_file,
        num_sessions=len(sessions),
        num_requests=len(raw_df),
        num_filtered_requests=filtered_requests,
    )


def import_human_logs(
    *,
    input_logs: list[str | Path],
    log_format: str = "auto",
    archive_dir: str | Path = "data/human_session_archives",
    live_log: str | Path = "data/live_logs/access.log",
    labels_path: str | Path = "data/live_labels/manual_labels.csv",
    session_timeout: float = 30 * 60,
    min_session_length: int = 2,
    participant_prefix: str = "human_manual",
    traffic_family: str = "human_navigation",
    collection_method: str = "manual_browser",
    automation_stack: str = "browser",
    notes: str = "imported real human browsing log",
    exclude_preview_agents: bool = True,
    append_live_log: bool = False,
) -> ImportResult:
    source_paths = [Path(path) for path in input_logs]
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    if len(source_paths) == 1:
        return import_human_log(
            input_log=source_paths[0],
            log_format=log_format,
            archive_dir=archive_dir,
            live_log=live_log,
            labels_path=labels_path,
            session_timeout=session_timeout,
            min_session_length=min_session_length,
            participant_prefix=participant_prefix,
            traffic_family=traffic_family,
            collection_method=collection_method,
            automation_stack=automation_stack,
            notes=notes,
            exclude_preview_agents=exclude_preview_agents,
            append_live_log=append_live_log,
        )

    archive_root = Path(archive_dir)
    archive_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wsd_human_import_") as temp_dir:
        combined_log = Path(temp_dir) / "combined_human_sessions.log"
        _write_combined_unique_log(source_paths, combined_log)
        result = import_human_log(
            input_log=combined_log,
            log_format=log_format,
            archive_dir=archive_dir,
            live_log=live_log,
            labels_path=labels_path,
            session_timeout=session_timeout,
            min_session_length=min_session_length,
            participant_prefix=participant_prefix,
            traffic_family=traffic_family,
            collection_method=collection_method,
            automation_stack=automation_stack,
            notes=notes,
            exclude_preview_agents=exclude_preview_agents,
            append_live_log=append_live_log,
        )
        source_manifest = result.archive_dir / "source_logs.txt"
        source_manifest.write_text(
            "\n".join(str(path.resolve()) for path in source_paths) + "\n",
            encoding="utf-8",
        )
        return result


def build_session_manifest(sessions: list[Session]) -> pd.DataFrame:
    summary = summarize_sessions(sessions)
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "label",
                "participant_id",
                "traffic_family",
                "collection_method",
                "automation_stack",
                "notes",
                "client_key",
                "ip",
                "num_events",
                "duration_seconds",
                "first_path",
                "start_time_utc",
                "end_time_utc",
                "path_signature",
                "paths",
                "user_agent",
            ]
        )

    paths_by_session = {session.session_id: session.paths() for session in sessions}
    manifest = summary.copy()
    manifest["label"] = "human"
    manifest["start_time_utc"] = manifest["start_timestamp"].map(_timestamp_to_utc)
    manifest["end_time_utc"] = manifest["end_timestamp"].map(_timestamp_to_utc)
    manifest["path_signature"] = manifest["session_id"].map(lambda sid: " > ".join(paths_by_session.get(sid, [])))
    manifest["paths"] = manifest["session_id"].map(lambda sid: " | ".join(paths_by_session.get(sid, [])))

    preferred_columns = [
        "session_id",
        "label",
        "participant_id",
        "traffic_family",
        "collection_method",
        "automation_stack",
        "notes",
        "client_key",
        "ip",
        "num_events",
        "duration_seconds",
        "unique_paths",
        "first_path",
        "start_time_utc",
        "end_time_utc",
        "path_signature",
        "paths",
        "user_agent",
    ]
    manifest = manifest.sort_values(["start_timestamp", "session_id"]).reset_index(drop=True)
    return manifest[[column for column in preferred_columns if column in manifest.columns]].copy()


def _attach_human_metadata(
    sessions: list[Session],
    *,
    participant_prefix: str,
    traffic_family: str,
    collection_method: str,
    automation_stack: str,
    notes: str,
) -> None:
    for index, session in enumerate(sorted(sessions, key=lambda item: (item.events[0].timestamp if item.events else 0.0, item.session_id)), start=1):
        participant_id = f"{participant_prefix}_{index:03d}"
        session.label = "human"
        for event in session.events:
            event.label = "human"
            event.extra["participant_id"] = participant_id
            event.extra["traffic_family"] = traffic_family
            event.extra["collection_method"] = collection_method
            event.extra["automation_stack"] = automation_stack
            event.extra["notes"] = notes


def _read_label_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.read_csv(path)


def _write_combined_unique_log(source_paths: list[Path], output_path: Path) -> None:
    seen: set[str] = set()
    lines: list[str] = []
    for source_path in source_paths:
        with source_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\r\n")
                if not line or line in seen:
                    continue
                seen.add(line)
                lines.append(line)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _filter_preview_agents(df: pd.DataFrame) -> pd.DataFrame:
    if "user_agent" not in df.columns or df.empty:
        return df.copy()
    mask = df["user_agent"].map(_is_nonhuman_preview_agent)
    return df[~mask].copy().reset_index(drop=True)


def _write_live_log(
    *,
    source_path: Path,
    live_log_path: Path,
    append_live_log: bool,
    exclude_preview_agents: bool,
) -> None:
    raw_text = source_path.read_text(encoding="utf-8", errors="ignore")
    if exclude_preview_agents:
        lines = [line for line in raw_text.splitlines() if not _nginx_line_has_preview_agent(line)]
        raw_text = "\n".join(lines)
        if raw_text:
            raw_text += "\n"

    if append_live_log and live_log_path.exists() and not _same_path(source_path, live_log_path):
        with live_log_path.open("a", encoding="utf-8", newline="") as destination:
            if live_log_path.stat().st_size > 0:
                destination.write("\n")
            destination.write(raw_text)
        return

    live_log_path.write_text(raw_text, encoding="utf-8")


def _nginx_line_has_preview_agent(line: str) -> bool:
    match = NginxCombinedRegex.match(line.strip())
    if not match:
        return False
    return _is_nonhuman_preview_agent(match.groupdict().get("user_agent"))


def _is_nonhuman_preview_agent(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).lower()
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in NONHUMAN_PREVIEW_AGENT_PATTERNS)


def _archive_log_name(source_path: Path) -> str:
    suffix = source_path.suffix or ".log"
    return f"raw_human_sessions{suffix}"


def _timestamp_to_utc(value: object) -> str:
    timestamp = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(timestamp):
        return ""
    return datetime.fromtimestamp(float(timestamp), tz=UTC).isoformat()


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def main() -> None:
    args = parse_args()
    result = import_human_logs(
        input_logs=args.input_log,
        log_format=args.format,
        archive_dir=args.archive_dir,
        live_log=args.live_log,
        labels_path=args.labels_path,
        session_timeout=args.session_timeout,
        min_session_length=args.min_session_length,
        participant_prefix=args.participant_prefix,
        traffic_family=args.traffic_family,
        collection_method=args.collection_method,
        automation_stack=args.automation_stack,
        notes=args.notes,
        exclude_preview_agents=not args.keep_preview_agents,
        append_live_log=args.append_live_log,
    )
    print(f"Imported human sessions: {result.num_sessions}")
    print(f"Normalized requests: {result.num_requests}")
    print(f"Filtered preview/crawler requests: {result.num_filtered_requests}")
    print(f"Archived raw log: {result.raw_log_copy.resolve()}")
    print(f"Session manifest: {result.manifest_path.resolve()}")
    print(f"Live log ready for prepare_live_dataset: {result.live_log_path.resolve()}")
    print(f"Updated labels: {result.labels_path.resolve()}")


if __name__ == "__main__":
    main()
