from pathlib import Path

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
