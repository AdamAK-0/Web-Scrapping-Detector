"""Weak-labeling and manual-label utilities for research dataset preparation."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import DEFAULT_BOT_USER_AGENT_PATTERNS, DEFAULT_NEGATIVE_LABEL, DEFAULT_POSITIVE_LABEL, DEFAULT_UNKNOWN_LABEL
from .sessionizer import summarize_sessions
from .types import Session

OPTIONAL_MANUAL_METADATA_COLUMNS = [
    "participant_id",
    "traffic_family",
    "collection_method",
    "automation_stack",
    "notes",
]


def apply_session_labels(
    sessions: list[Session],
    *,
    manual_labels_path: str | Path | None = None,
    bot_user_agent_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Return a session summary with proposed labels and mutate session labels.

    Label precedence:
    1. manual labels
    2. strong bot user-agent match
    3. weak heuristic score
    4. unknown
    """
    bot_user_agent_patterns = bot_user_agent_patterns or DEFAULT_BOT_USER_AGENT_PATTERNS
    manual_df = load_manual_labels(manual_labels_path) if manual_labels_path else pd.DataFrame()

    summary = summarize_sessions(sessions)
    if summary.empty:
        return summary

    summary = _merge_manual_metadata(summary, manual_df)
    summary["suspicious_user_agent"] = summary["user_agent"].map(lambda ua: _matches_any(ua, bot_user_agent_patterns))
    summary["weak_bot_score"] = summary.apply(_weak_bot_score, axis=1)
    summary["proposed_label"] = summary.apply(_decide_label, axis=1)

    summary_lookup = summary.set_index("session_id").to_dict(orient="index")
    for session in sessions:
        row = summary_lookup.get(session.session_id, {})
        session.label = str(row.get("proposed_label", DEFAULT_UNKNOWN_LABEL))
        for event in session.events:
            event.label = session.label
            for column in OPTIONAL_MANUAL_METADATA_COLUMNS:
                value = row.get(column)
                if value is None or pd.isna(value) or not str(value).strip():
                    continue
                event.extra[column] = str(value).strip()
            if row.get("traffic_family") and not event.extra.get("traffic_family"):
                event.extra["traffic_family"] = row["traffic_family"]
    return summary


def load_manual_labels(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    label_df = pd.read_csv(path)
    if not {"label"}.issubset(label_df.columns):
        raise ValueError("Manual labels CSV must contain a 'label' column and a 'client_key' or 'session_id' column")
    if "client_key" not in label_df.columns and "session_id" not in label_df.columns:
        raise ValueError("Manual labels CSV must contain either 'client_key' or 'session_id'")
    working = label_df.copy()
    working["label"] = working["label"].astype(str).str.strip().str.lower()
    for column in ("client_key", "session_id", *OPTIONAL_MANUAL_METADATA_COLUMNS):
        if column not in working.columns:
            working[column] = pd.NA
    return working[["client_key", "session_id", "label", *OPTIONAL_MANUAL_METADATA_COLUMNS]].copy()


def _matches_any(value: object, patterns: list[str]) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).lower()
    return any(re.search(pattern, text, flags=re.IGNORECASE) is not None for pattern in patterns)


def _weak_bot_score(row: pd.Series) -> float:
    score = 0.0
    if bool(row.get("suspicious_user_agent", False)):
        score += 0.55
    if float(row.get("mean_delta_t", 0.0)) <= 1.0:
        score += 0.20
    if int(row.get("num_events", 0)) >= 20:
        score += 0.10
    if float(row.get("unique_path_ratio", 0.0)) >= 0.90:
        score += 0.10
    if int(row.get("page_categories", 0)) <= 2 and int(row.get("num_events", 0)) >= 10:
        score += 0.05
    return min(score, 1.0)


def _decide_label(row: pd.Series) -> str:
    manual_label = row.get("manual_label")
    if isinstance(manual_label, str) and manual_label.strip():
        return manual_label.strip().lower()
    if bool(row.get("suspicious_user_agent", False)) and float(row.get("weak_bot_score", 0.0)) >= 0.55:
        return DEFAULT_POSITIVE_LABEL
    if float(row.get("weak_bot_score", 0.0)) >= 0.75:
        return DEFAULT_POSITIVE_LABEL
    if float(row.get("weak_bot_score", 0.0)) <= 0.15 and int(row.get("num_events", 0)) >= 3:
        return DEFAULT_NEGATIVE_LABEL
    return DEFAULT_UNKNOWN_LABEL


def _merge_manual_metadata(summary: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    working = summary.copy()
    working["manual_label"] = pd.NA
    for column in OPTIONAL_MANUAL_METADATA_COLUMNS:
        working[column] = working.get(column, pd.Series(index=working.index, dtype="object"))
        working[column] = working[column].replace("", pd.NA)

    if manual_df.empty:
        return working

    client_manual = manual_df[manual_df["client_key"].notna()].copy()
    if not client_manual.empty:
        client_columns = ["client_key", "label", *OPTIONAL_MANUAL_METADATA_COLUMNS]
        client_manual = client_manual[client_columns].rename(columns={"label": "manual_label"})
        working = working.merge(client_manual, on="client_key", how="left", suffixes=("", "_manual_client"))
        working["manual_label"] = working["manual_label"].fillna(working.pop("manual_label_manual_client"))
        for column in OPTIONAL_MANUAL_METADATA_COLUMNS:
            manual_column = f"{column}_manual_client"
            if manual_column in working.columns:
                working[column] = working[column].fillna(working.pop(manual_column))

    session_manual = manual_df[manual_df["session_id"].notna()].copy()
    if not session_manual.empty:
        session_columns = ["session_id", "label", *OPTIONAL_MANUAL_METADATA_COLUMNS]
        session_manual = session_manual[session_columns].rename(columns={"label": "manual_label"})
        working = working.merge(session_manual, on="session_id", how="left", suffixes=("", "_manual_session"))
        working["manual_label"] = working["manual_label"].fillna(working.pop("manual_label_manual_session"))
        for column in OPTIONAL_MANUAL_METADATA_COLUMNS:
            manual_column = f"{column}_manual_session"
            if manual_column in working.columns:
                working[column] = working[column].fillna(working.pop(manual_column))

    return working
