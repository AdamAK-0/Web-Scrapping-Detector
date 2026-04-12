"""Utilities for turning request logs into ordered sessions."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from .config import (
    ALLOWED_LABELS,
    DEFAULT_NEGATIVE_LABEL,
    DEFAULT_PAGE_LIKE_EXTENSIONS_TO_EXCLUDE,
    DEFAULT_SESSION_TIMEOUT_SECONDS,
    DEFAULT_UNKNOWN_LABEL,
)
from .graph_builder import infer_category_from_path
from .types import RequestEvent, Session


REQUIRED_COLUMNS = {"session_id", "timestamp", "path", "delta_t", "label"}


def load_sessions_from_csv(csv_path: str | Path) -> list[Session]:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Request log missing required columns: {sorted(missing)}")

    grouped: dict[str, list[RequestEvent]] = defaultdict(list)
    labels_by_session: dict[str, str] = {}

    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

    for row in df.to_dict(orient="records"):
        label = str(row["label"]).lower().strip()
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported label '{label}'. Allowed labels: {sorted(ALLOWED_LABELS)}")

        session_id = str(row["session_id"])
        if session_id in labels_by_session and labels_by_session[session_id] != label:
            raise ValueError(f"Session {session_id} contains conflicting labels")
        labels_by_session[session_id] = label

        extra = {
            k: v
            for k, v in row.items()
            if k not in REQUIRED_COLUMNS | {"page_category", "referrer", "user_agent", "status_code", "client_key", "ip", "method"}
        }
        grouped[session_id].append(
            RequestEvent(
                session_id=session_id,
                timestamp=float(row["timestamp"]),
                path=str(row["path"]),
                delta_t=float(row["delta_t"]),
                label=label,
                page_category=_optional_str(row.get("page_category")),
                referrer=_optional_str(row.get("referrer")),
                user_agent=_optional_str(row.get("user_agent")),
                status_code=_optional_int(row.get("status_code")),
                extra=extra,
            )
        )

    sessions = [Session(session_id=sid, label=labels_by_session[sid], events=events) for sid, events in grouped.items()]
    return sorted(sessions, key=lambda session: session.session_id)


def save_sessions_to_csv(sessions: list[Session], csv_path: str | Path) -> None:
    rows = []
    for session in sessions:
        for event in session.events:
            row = {
                "session_id": session.session_id,
                "timestamp": event.timestamp,
                "path": event.path,
                "delta_t": event.delta_t,
                "label": session.label,
                "page_category": event.page_category,
                "referrer": event.referrer,
                "user_agent": event.user_agent,
                "status_code": event.status_code,
            }
            row.update(event.extra)
            rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def build_sessions_from_dataframe(
    df: pd.DataFrame,
    *,
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS,
    default_label: str = DEFAULT_UNKNOWN_LABEL,
    drop_asset_requests: bool = True,
    min_session_length: int = 2,
) -> list[Session]:
    if df.empty:
        return []

    working = df.copy()
    working["path"] = working["path"].map(normalize_path)
    working = working[working["path"].notna()].copy()
    working["path"] = working["path"].astype(str)
    if drop_asset_requests:
        working = working[working["path"].map(is_page_like_path)].copy()
    if working.empty:
        return []

    if "client_key" not in working.columns:
        working["client_key"] = working.apply(_derive_client_key, axis=1)

    working["timestamp"] = pd.to_numeric(working["timestamp"], errors="coerce")
    working = working[working["timestamp"].notna()].copy()
    working = working.sort_values(["client_key", "timestamp", "path"]).reset_index(drop=True)

    sessions: list[Session] = []
    for client_key, group in working.groupby("client_key", sort=True):
        session_index = 0
        buffer: list[dict[str, object]] = []
        last_timestamp: float | None = None

        for row in group.to_dict(orient="records"):
            timestamp = float(row["timestamp"])
            if last_timestamp is not None and timestamp - last_timestamp > session_timeout_seconds:
                maybe_session = _build_session_from_rows(client_key, session_index, buffer, default_label=default_label)
                if maybe_session is not None and len(maybe_session.events) >= min_session_length:
                    sessions.append(maybe_session)
                session_index += 1
                buffer = []
            buffer.append(row)
            last_timestamp = timestamp

        maybe_session = _build_session_from_rows(client_key, session_index, buffer, default_label=default_label)
        if maybe_session is not None and len(maybe_session.events) >= min_session_length:
            sessions.append(maybe_session)

    return sessions


def sessions_to_dataframe(sessions: list[Session]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session in sessions:
        for event in session.events:
            row = {
                "session_id": session.session_id,
                "timestamp": event.timestamp,
                "path": event.path,
                "delta_t": event.delta_t,
                "label": session.label,
                "page_category": event.page_category,
                "referrer": event.referrer,
                "user_agent": event.user_agent,
                "status_code": event.status_code,
            }
            row.update(event.extra)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_sessions(sessions: list[Session]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session in sessions:
        events = session.events
        timestamps = [e.timestamp for e in events]
        deltas = [e.delta_t for e in events[1:]]
        unique_paths = len({e.path for e in events})
        client_key = str(events[0].extra.get("client_key", "")) if events else ""
        ip = str(events[0].extra.get("ip", "")) if events else ""
        rows.append(
            {
                "session_id": session.session_id,
                "label": session.label,
                "client_key": client_key,
                "ip": ip,
                "num_events": len(events),
                "duration_seconds": max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0,
                "unique_paths": unique_paths,
                "unique_path_ratio": unique_paths / len(events) if events else 0.0,
                "mean_delta_t": sum(deltas) / len(deltas) if deltas else 0.0,
                "page_categories": len({e.page_category or infer_category_from_path(e.path) for e in events}),
                "user_agent": events[0].user_agent if events else None,
            }
        )
    return pd.DataFrame(rows)


def normalize_path(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        path = parsed.path or "/"
    else:
        path = raw.split("?")[0].split("#")[0]
    if not path.startswith("/"):
        path = "/" + path
    return path


def is_page_like_path(path: str) -> bool:
    lowered = path.lower()
    if lowered in {"/favicon.ico"}:
        return False
    if "." not in lowered.rsplit("/", 1)[-1]:
        return True
    for ext in DEFAULT_PAGE_LIKE_EXTENSIONS_TO_EXCLUDE:
        if lowered.endswith(ext):
            return False
    return True


def _build_session_from_rows(client_key: str, session_index: int, rows: list[dict[str, object]], default_label: str) -> Session | None:
    if not rows:
        return None
    session_id = _make_session_id(client_key, session_index)
    labels = {str(row.get("label", default_label)).lower().strip() or default_label for row in rows}
    labels = {label for label in labels if label in ALLOWED_LABELS}
    label = labels.pop() if len(labels) == 1 else (default_label if default_label in ALLOWED_LABELS else DEFAULT_NEGATIVE_LABEL)

    events: list[RequestEvent] = []
    last_timestamp: float | None = None
    last_path: str | None = None
    for row in rows:
        timestamp = float(row["timestamp"])
        path = str(row["path"])
        delta_t = 0.0 if last_timestamp is None else max(0.0, timestamp - last_timestamp)
        page_category = _optional_str(row.get("page_category")) or infer_category_from_path(path)
        event = RequestEvent(
            session_id=session_id,
            timestamp=timestamp,
            path=path,
            delta_t=delta_t,
            label=label,
            page_category=page_category,
            referrer=_optional_str(row.get("referrer")) or last_path,
            user_agent=_optional_str(row.get("user_agent")),
            status_code=_optional_int(row.get("status_code")),
            extra={
                "client_key": client_key,
                "ip": _optional_str(row.get("ip")),
                "method": _optional_str(row.get("method")),
                **{k: v for k, v in row.items() if k not in {"timestamp", "path", "page_category", "referrer", "user_agent", "status_code", "label", "client_key", "ip", "method"}},
            },
        )
        events.append(event)
        last_timestamp = timestamp
        last_path = path

    return Session(session_id=session_id, label=label, events=events)


def _derive_client_key(row: pd.Series) -> str:
    for column in ("cookie_id", "client_id"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return f"{column}:{str(value).strip()}"
    ip = str(row.get("ip", "")).strip()
    user_agent = str(row.get("user_agent", "")).strip()
    return f"ip_ua:{ip}|{user_agent}"


def _make_session_id(client_key: str, session_index: int) -> str:
    digest = sha1(client_key.encode("utf-8")).hexdigest()[:12]
    return f"sess_{digest}_{session_index:03d}"


def _optional_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
