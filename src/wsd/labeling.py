"""Weak-labeling and manual-label utilities for research dataset preparation."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import DEFAULT_BOT_USER_AGENT_PATTERNS, DEFAULT_NEGATIVE_LABEL, DEFAULT_POSITIVE_LABEL, DEFAULT_UNKNOWN_LABEL
from .sessionizer import summarize_sessions
from .types import Session


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
    manual = load_manual_labels(manual_labels_path) if manual_labels_path else {}

    summary = summarize_sessions(sessions)
    if summary.empty:
        return summary

    summary["manual_label"] = summary["client_key"].map(manual).fillna(summary["session_id"].map(manual))
    summary["suspicious_user_agent"] = summary["user_agent"].map(lambda ua: _matches_any(ua, bot_user_agent_patterns))
    summary["weak_bot_score"] = summary.apply(_weak_bot_score, axis=1)
    summary["proposed_label"] = summary.apply(_decide_label, axis=1)

    proposed_by_session = dict(zip(summary["session_id"], summary["proposed_label"], strict=False))
    for session in sessions:
        session.label = proposed_by_session.get(session.session_id, DEFAULT_UNKNOWN_LABEL)
        for event in session.events:
            event.label = session.label
    return summary


def load_manual_labels(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    label_df = pd.read_csv(path)
    if not {"label"}.issubset(label_df.columns):
        raise ValueError("Manual labels CSV must contain a 'label' column and a 'client_key' or 'session_id' column")
    key_column = "client_key" if "client_key" in label_df.columns else "session_id" if "session_id" in label_df.columns else None
    if key_column is None:
        raise ValueError("Manual labels CSV must contain either 'client_key' or 'session_id'")
    return {str(row[key_column]): str(row["label"]).strip().lower() for row in label_df.to_dict(orient="records")}


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
