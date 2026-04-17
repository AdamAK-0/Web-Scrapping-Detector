from pathlib import Path

import pandas as pd

from wsd.import_human_sessions import import_human_log, import_human_logs


def test_import_human_log_archives_sessions_and_labels(tmp_path: Path) -> None:
    source_log = tmp_path / "logs.txt"
    source_log.write_text(
        "\n".join(
            [
                '203.0.113.10 - - [12/Apr/2026:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 512 "-" "Mozilla/5.0 TestHuman"',
                '203.0.113.10 - - [12/Apr/2026:10:00:08 +0000] "GET /products.html HTTP/1.1" 200 512 "/index.html" "Mozilla/5.0 TestHuman"',
                '203.0.113.10 - - [12/Apr/2026:10:00:12 +0000] "GET /assets/site.css HTTP/1.1" 200 32 "/products.html" "Mozilla/5.0 TestHuman"',
                '203.0.113.10 - - [12/Apr/2026:11:00:00 +0000] "GET /articles.html HTTP/1.1" 200 512 "-" "Mozilla/5.0 TestHuman"',
                '203.0.113.10 - - [12/Apr/2026:11:00:05 +0000] "GET /faq.html HTTP/1.1" 200 512 "/articles.html" "Mozilla/5.0 TestHuman"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_human_log(
        input_log=source_log,
        archive_dir=tmp_path / "archives",
        live_log=tmp_path / "live" / "access.log",
        labels_path=tmp_path / "labels" / "manual_labels.csv",
        participant_prefix="p",
    )

    assert result.num_sessions == 2
    assert result.raw_log_copy.exists()
    assert result.live_log_path.read_text(encoding="utf-8") == source_log.read_text(encoding="utf-8")

    manifest = pd.read_csv(result.manifest_path)
    assert manifest["participant_id"].tolist() == ["p_001", "p_002"]
    assert set(manifest["label"]) == {"human"}
    assert "path_signature" in manifest.columns

    labels = pd.read_csv(result.labels_path)
    assert set(labels["session_id"]) == set(manifest["session_id"])
    assert set(labels["label"]) == {"human"}
    assert set(labels["collection_method"]) == {"manual_browser"}


def test_import_human_log_filters_link_preview_agents_from_working_log(tmp_path: Path) -> None:
    source_log = tmp_path / "logs.txt"
    source_log.write_text(
        "\n".join(
            [
                '203.0.113.10 - - [12/Apr/2026:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 512 "-" "Mozilla/5.0 TestHuman"',
                '203.0.113.10 - - [12/Apr/2026:10:00:05 +0000] "GET /products.html HTTP/1.1" 200 512 "/index.html" "Mozilla/5.0 TestHuman"',
                '203.0.113.55 - - [12/Apr/2026:10:01:00 +0000] "GET /index.html HTTP/1.1" 200 512 "-" "WhatsApp/2.23.20.0"',
                '203.0.113.55 - - [12/Apr/2026:10:01:02 +0000] "GET /products.html HTTP/1.1" 200 512 "/index.html" "WhatsApp/2.23.20.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = import_human_log(
        input_log=source_log,
        archive_dir=tmp_path / "archives",
        live_log=tmp_path / "live" / "access.log",
        labels_path=tmp_path / "labels" / "manual_labels.csv",
    )

    assert result.num_filtered_requests == 2
    assert "WhatsApp" in result.raw_log_copy.read_text(encoding="utf-8")
    assert "WhatsApp" not in result.live_log_path.read_text(encoding="utf-8")
    assert len(pd.read_csv(result.manifest_path)) == 1


def test_import_human_logs_combines_archives_without_duplicate_lines(tmp_path: Path) -> None:
    first_log = tmp_path / "first.log"
    first_log.write_text(
        "\n".join(
            [
                '203.0.113.10 - - [12/Apr/2026:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 512 "-" "Mozilla/5.0 HumanA"',
                '203.0.113.10 - - [12/Apr/2026:10:00:05 +0000] "GET /products.html HTTP/1.1" 200 512 "/index.html" "Mozilla/5.0 HumanA"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    second_log = tmp_path / "second.log"
    second_log.write_text(
        first_log.read_text(encoding="utf-8")
        + '203.0.113.11 - - [12/Apr/2026:10:02:00 +0000] "GET /articles.html HTTP/1.1" 200 512 "-" "Mozilla/5.0 HumanB"\n'
        + '203.0.113.11 - - [12/Apr/2026:10:02:05 +0000] "GET /faq.html HTTP/1.1" 200 512 "/articles.html" "Mozilla/5.0 HumanB"\n',
        encoding="utf-8",
    )

    result = import_human_logs(
        input_logs=[first_log, second_log],
        archive_dir=tmp_path / "archives",
        live_log=tmp_path / "live" / "access.log",
        labels_path=tmp_path / "labels" / "manual_labels.csv",
    )

    manifest = pd.read_csv(result.manifest_path)
    assert result.num_requests == 4
    assert len(manifest) == 2
    assert (result.archive_dir / "source_logs.txt").exists()
