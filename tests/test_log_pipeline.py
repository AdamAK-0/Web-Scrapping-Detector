from pathlib import Path

import pandas as pd

from wsd.labeling import apply_session_labels
from wsd.log_parsers import read_raw_logs
from wsd.sessionizer import build_sessions_from_dataframe


def test_nginx_combined_parse_and_sessionize(tmp_path: Path) -> None:
    log_text = "\n".join(
        [
            '203.0.113.10 - - [12/Apr/2026:10:00:00 +0000] "GET / HTTP/1.1" 200 512 "-" "Mozilla/5.0"',
            '203.0.113.10 - - [12/Apr/2026:10:00:05 +0000] "GET /blog HTTP/1.1" 200 512 "/" "Mozilla/5.0"',
            '203.0.113.10 - - [12/Apr/2026:10:00:11 +0000] "GET /about HTTP/1.1" 200 512 "/blog" "Mozilla/5.0"',
            '203.0.113.99 - - [12/Apr/2026:10:00:00 +0000] "GET / HTTP/1.1" 200 512 "-" "python-requests/2.31"',
            '203.0.113.99 - - [12/Apr/2026:10:00:01 +0000] "GET /blog/post-1 HTTP/1.1" 200 512 "/" "python-requests/2.31"',
        ]
    )
    path = tmp_path / "access.log"
    path.write_text(log_text + "\n", encoding="utf-8")

    df = read_raw_logs(path, log_format="nginx_combined")
    sessions = build_sessions_from_dataframe(df, min_session_length=2)
    summary = apply_session_labels(sessions)

    assert len(sessions) == 2
    assert set(summary["proposed_label"]) == {"human", "bot"}


def test_manual_label_metadata_flows_into_summary_and_events(tmp_path: Path) -> None:
    log_text = "\n".join(
        [
            '203.0.113.10 - - [12/Apr/2026:10:00:00 +0000] "GET / HTTP/1.1" 200 512 "-" "Mozilla/5.0 SampleUA"',
            '203.0.113.10 - - [12/Apr/2026:10:00:05 +0000] "GET /products HTTP/1.1" 200 512 "/" "Mozilla/5.0 SampleUA"',
        ]
    )
    path = tmp_path / "access.log"
    path.write_text(log_text + "\n", encoding="utf-8")
    manual_labels = tmp_path / "manual_labels.csv"
    pd.DataFrame(
        [
            {
                "client_key": "ip_ua:203.0.113.10|Mozilla/5.0 SampleUA",
                "label": "human",
                "participant_id": "participant_01",
                "collection_method": "manual_browser",
            }
        ]
    ).to_csv(manual_labels, index=False)

    df = read_raw_logs(path, log_format="nginx_combined")
    sessions = build_sessions_from_dataframe(df, min_session_length=2)
    summary = apply_session_labels(sessions, manual_labels_path=manual_labels)

    assert summary["participant_id"].iloc[0] == "participant_01"
    assert sessions[0].events[0].extra["participant_id"] == "participant_01"
