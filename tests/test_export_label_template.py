import pandas as pd

from wsd.export_label_template import build_annotation_template


def test_build_annotation_template_exports_expected_columns() -> None:
    summary = pd.DataFrame(
        [
            {
                "session_id": "sess_001",
                "client_key": "ip_ua:1|ua",
                "proposed_label": "unknown",
                "participant_id": "",
                "num_events": 7,
                "duration_seconds": 42.0,
                "first_path": "/index.html",
                "user_agent": "Mozilla/5.0",
            }
        ]
    )
    template = build_annotation_template(summary, only_unknown=True, keep_existing_labels=False)
    assert "participant_id" in template.columns
    assert "traffic_family" in template.columns
    assert template.loc[0, "label"] == ""
    assert template.loc[0, "session_id"] == "sess_001"
