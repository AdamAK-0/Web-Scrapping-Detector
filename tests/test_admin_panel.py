from pathlib import Path

import pandas as pd

from wsd.admin_panel import AdminState, create_app, discover_model_bundles, reset_live_view, score_live_log
from wsd.graph_builder import build_graph_from_edge_list, save_graph_to_csv
from wsd.modeling import ModelBundle, build_models, save_model_bundle


def test_admin_panel_discovers_models_and_scores_live_log(tmp_path: Path) -> None:
    graph = build_graph_from_edge_list([
        ("/index.html", "/products.html"),
        ("/products.html", "/cart.html"),
    ])
    graph_dir = tmp_path / "prepared"
    model_dir = graph_dir / "models"
    graph_dir.mkdir()
    model_dir.mkdir()
    save_graph_to_csv(graph, graph_dir / "graph_edges.csv", graph_dir / "graph_categories.csv")

    feature_columns = ["session_length_so_far", "mean_delta_t"]
    model = build_models(random_state=1, selected_models=["logistic_regression"])["logistic_regression"]
    model.fit(
        pd.DataFrame([
            {"session_length_so_far": 3.0, "mean_delta_t": 0.2},
            {"session_length_so_far": 3.0, "mean_delta_t": 8.0},
        ]),
        [1, 0],
    )
    save_model_bundle(
        ModelBundle(model_name="logistic_regression", feature_columns=feature_columns, model=model, threshold=0.5),
        model_dir / "logistic_regression_bundle.pkl",
    )

    log_path = tmp_path / "access.log"
    log_path.write_text(
        "\n".join(
            [
                '203.0.113.10 - - [12/Apr/2026:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 512 "-" "Mozilla/5.0"',
                '203.0.113.10 - - [12/Apr/2026:10:00:01 +0000] "GET /products.html HTTP/1.1" 200 512 "/index.html" "Mozilla/5.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = AdminState(
        model_dir=model_dir,
        graph_dir=graph_dir,
        log_path=log_path,
        labels_path=tmp_path / "manual_labels.csv",
        base_url="http://127.0.0.1:8039",
    )
    app = create_app(state)

    assert discover_model_bundles(model_dir)[0]["model_name"] == "logistic_regression"
    assert app.state.admin_state.active_model_name == "logistic_regression"

    status = score_live_log(state)
    assert status["ok"] is True
    assert status["summary"]["total_sessions"] == 1
    assert status["sessions"][0]["ready"] is True
    assert status["sessions"][0]["bot_probability"] is not None


def test_admin_panel_treats_empty_live_log_as_clean_state(tmp_path: Path) -> None:
    graph = build_graph_from_edge_list([("/index.html", "/products.html")])
    graph_dir = tmp_path / "prepared"
    model_dir = graph_dir / "models"
    graph_dir.mkdir()
    model_dir.mkdir()
    save_graph_to_csv(graph, graph_dir / "graph_edges.csv", graph_dir / "graph_categories.csv")

    feature_columns = ["session_length_so_far", "mean_delta_t"]
    model = build_models(random_state=1, selected_models=["logistic_regression"])["logistic_regression"]
    model.fit(
        pd.DataFrame([
            {"session_length_so_far": 3.0, "mean_delta_t": 0.2},
            {"session_length_so_far": 3.0, "mean_delta_t": 8.0},
        ]),
        [1, 0],
    )
    save_model_bundle(
        ModelBundle(model_name="logistic_regression", feature_columns=feature_columns, model=model, threshold=0.5),
        model_dir / "logistic_regression_bundle.pkl",
    )

    log_path = tmp_path / "access.log"
    log_path.write_text("", encoding="utf-8")
    state = AdminState(
        model_dir=model_dir,
        graph_dir=graph_dir,
        log_path=log_path,
        labels_path=tmp_path / "manual_labels.csv",
        base_url="http://127.0.0.1:8039",
    )

    status = score_live_log(state)
    assert status["ok"] is True
    assert status["summary"]["total_sessions"] == 0
    assert status["sessions"] == []


def test_admin_panel_reset_live_view_clears_logs_labels_and_bot_runs(tmp_path: Path) -> None:
    log_path = tmp_path / "live" / "access.log"
    error_log = tmp_path / "live" / "error.log"
    labels_path = tmp_path / "labels" / "manual_labels.csv"
    bot_run_dir = tmp_path / "admin_bot_runs"
    log_path.parent.mkdir()
    labels_path.parent.mkdir()
    bot_run_dir.mkdir()
    log_path.write_text("old access\n", encoding="utf-8")
    error_log.write_text("old error\n", encoding="utf-8")
    labels_path.write_text("session_id,label\ns1,bot\n", encoding="utf-8")
    (bot_run_dir / "old.log").write_text("old run\n", encoding="utf-8")

    state = AdminState(
        model_dir=tmp_path / "models",
        graph_dir=tmp_path / "prepared",
        log_path=log_path,
        labels_path=labels_path,
        base_url="http://127.0.0.1:8039",
        bot_run_dir=bot_run_dir,
    )

    result = reset_live_view(state)

    assert result["warnings"] == []
    assert log_path.read_text(encoding="utf-8") == ""
    assert error_log.read_text(encoding="utf-8") == ""
    assert not labels_path.exists()
    assert not bot_run_dir.exists()
    assert state.ignore_log_before_timestamp > 0
