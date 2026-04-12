"""Raw log ingestion for CSV, JSONL, and Nginx combined logs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd

from .sessionizer import normalize_path

NginxCombinedRegex = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
    r'"(?P<method>\S+) (?P<path>\S+)(?: (?P<protocol>[^"]+))?" '
    r'(?P<status_code>\d{3}) (?P<body_bytes_sent>\S+) '
    r'"(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)"$'
)


CSV_COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "ts", "@timestamp", "date", "datetime"],
    "path": ["path", "uri", "url", "request_uri", "request_path", "cs-uri-stem"],
    "method": ["method", "request_method", "http_method", "cs-method"],
    "status_code": ["status_code", "status", "sc-status"],
    "user_agent": ["user_agent", "http_user_agent", "agent", "cs-user-agent"],
    "referrer": ["referrer", "referer", "http_referer", "cs-referer"],
    "ip": ["ip", "client_ip", "remote_addr", "x_forwarded_for", "c-ip"],
    "cookie_id": ["cookie_id", "session_cookie", "cookie", "sessionid"],
    "client_id": ["client_id", "visitor_id", "device_id"],
    "label": ["label"],
}


def read_raw_logs(input_path: str | Path, log_format: str = "auto") -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if log_format == "auto":
        log_format = infer_log_format(path)

    if log_format == "csv":
        df = pd.read_csv(path)
        return normalize_request_dataframe(df)
    if log_format == "jsonl":
        return normalize_request_dataframe(_read_jsonl(path))
    if log_format == "nginx_combined":
        return _read_nginx_combined(path)
    raise ValueError(f"Unsupported log format: {log_format}")


def infer_log_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith("{"):
                return "jsonl"
            if NginxCombinedRegex.match(text):
                return "nginx_combined"
            break
    return "csv"


def normalize_request_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame()
    for canonical, aliases in CSV_COLUMN_ALIASES.items():
        source = _first_existing_column(df.columns, aliases)
        if source is not None:
            normalized[canonical] = df[source]

    if "timestamp" not in normalized.columns:
        raise ValueError("Input logs must contain a timestamp column")
    if "path" not in normalized.columns:
        if "request_line" in df.columns:
            normalized["path"] = df["request_line"].map(_path_from_request_line)
        else:
            raise ValueError("Input logs must contain a path/url/request_uri column")

    normalized["timestamp"] = _coerce_timestamp_series(normalized["timestamp"])
    normalized["path"] = normalized["path"].map(normalize_path)
    if "referrer" in normalized.columns:
        normalized["referrer"] = normalized["referrer"].map(_normalize_referrer)
    if "status_code" in normalized.columns:
        normalized["status_code"] = pd.to_numeric(normalized["status_code"], errors="coerce").astype("Int64")
    if "method" not in normalized.columns:
        normalized["method"] = "GET"
    if "ip" not in normalized.columns:
        normalized["ip"] = None
    if "user_agent" not in normalized.columns:
        normalized["user_agent"] = None
    if "label" not in normalized.columns:
        normalized["label"] = None

    normalized = normalized[normalized["timestamp"].notna() & normalized["path"].notna()].copy()
    normalized["timestamp"] = normalized["timestamp"].astype(float)
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    return normalized


def _read_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return pd.DataFrame(records)


def _read_nginx_combined(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            match = NginxCombinedRegex.match(text)
            if not match:
                raise ValueError(f"Could not parse Nginx combined log line {line_number}: {text}")
            data = match.groupdict()
            rows.append(
                {
                    "ip": data["ip"],
                    "timestamp": _parse_nginx_timestamp(data["timestamp"]),
                    "method": data["method"],
                    "path": normalize_path(data["path"]),
                    "status_code": int(data["status_code"]),
                    "body_bytes_sent": None if data["body_bytes_sent"] == "-" else int(data["body_bytes_sent"]),
                    "referrer": _normalize_referrer(data["referrer"]),
                    "user_agent": None if data["user_agent"] == "-" else data["user_agent"],
                    "label": None,
                }
            )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _parse_nginx_timestamp(text: str) -> float:
    timestamp = pd.to_datetime(text, format="%d/%b/%Y:%H:%M:%S %z", utc=True)
    return float(timestamp.timestamp())


def _normalize_referrer(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urlparse(text)
        return normalize_path(parsed.path)
    return normalize_path(text)


def _path_from_request_line(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    parts = str(value).split()
    if len(parts) >= 2:
        return normalize_path(parts[1])
    return None


def _coerce_timestamp_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.map(lambda x: float(x.timestamp()) if not pd.isna(x) else None)


def _first_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match is not None:
            return match
    return None
