from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from generic_models.generic_admin_panel import GenericAdminState, build_sessions_from_log_frame, read_generic_log_frame, score_generic_sessions
from generic_models.site_catalog import build_all_generic_sites, get_websites, root_path


class ConstantModel:
    def predict_proba(self, X):  # noqa: N803 - sklearn-style API
        return np.column_stack([np.full(len(X), 0.18), np.full(len(X), 0.82)])


def test_generic_sites_have_distinct_shapes_and_ports() -> None:
    specs = get_websites()
    assert len(specs) >= 4
    assert len({spec.port for spec in specs.values()}) == len(specs)
    assert len({spec.shape for spec in specs.values()}) == len(specs)

    sites = build_all_generic_sites()
    for site_id, site in sites.items():
        assert root_path(site_id) in site.graph
        assert site.graph.number_of_nodes() >= 10
        assert site.graph.number_of_edges() >= 10
        assert site.reachable_nodes


def test_generic_live_log_sessionizes_and_scores(tmp_path: Path) -> None:
    specs = get_websites()
    site_id = "atlas_shop"
    spec = specs[site_id]
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    now = time.time()
    events = [
        {"timestamp": now + index, "site_id": site_id, "ip": "127.0.0.1", "user_agent": "GenericWSDTestBot/bfs", "path": path, "status_code": 200}
        for index, path in enumerate([root_path(site_id), "/catalog", "/catalog/water", "/p/aqua"])
    ]
    with (log_dir / f"{site_id}.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")

    state = GenericAdminState(model_dir=tmp_path / "models", log_dir=log_dir)
    state.active_bundle = {
        "model": ConstantModel(),
        "feature_columns": ["prefix_len", "coverage_ratio", "path_entropy", "revisit_rate", "graph_distance_mean"],
        "threshold": 0.5,
    }
    state.active_model_name = "constant"
    state.threshold = 0.5

    frame = read_generic_log_frame(log_dir)
    sessions = build_sessions_from_log_frame(frame, state=state)
    rows = score_generic_sessions(sessions, state=state)

    assert len(sessions) == 1
    assert sessions[0].paths[0] == root_path(site_id)
    assert rows[0]["site_name"] == spec.name
    assert rows[0]["predicted_label"] == "bot"
    assert rows[0]["bot_probability_pct"] == 82
