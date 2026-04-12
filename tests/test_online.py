from pathlib import Path

from wsd.graph_builder import build_graph_from_edge_list
from wsd.modeling import ModelBundle, build_models
from wsd.online import OnlineDetector
import pandas as pd


def test_online_detector_returns_score_after_two_events() -> None:
    graph = build_graph_from_edge_list([
        ("/", "/blog"),
        ("/blog", "/blog/post-1"),
        ("/blog/post-1", "/support/faq-1"),
    ])

    feature_columns = [
        "session_length_so_far",
        "unique_nodes",
        "unique_node_ratio",
        "revisit_ratio",
        "repeated_path_ratio",
        "repeated_transition_ratio",
        "backtrack_count",
        "self_loop_ratio",
        "mean_depth",
        "leaf_visit_ratio",
        "out_degree_mean",
        "mean_hop_distance",
        "far_jump_ratio",
        "mean_delta_t",
        "std_delta_t",
        "burstiness",
        "low_latency_ratio",
        "error_rate",
        "user_agent_switch_rate",
        "category_switch_rate",
        "transition_entropy",
        "normalized_transition_entropy",
        "category_entropy",
        "normalized_category_entropy",
        "node_entropy",
        "normalized_node_entropy",
        "navigation_entropy_score",
    ]

    model = build_models(random_state=1)["logistic_regression"]
    X = pd.DataFrame([
        {col: 0.2 for col in feature_columns},
        {col: 0.8 for col in feature_columns},
    ])
    y = [1, 0]
    model.fit(X, y)
    bundle = ModelBundle(model_name="logreg", feature_columns=feature_columns, model=model)
    detector = OnlineDetector(bundle, graph)

    first = detector.observe(session_id="s1", path="/", timestamp=1.0, user_agent="python-requests/2.31")
    second = detector.observe(session_id="s1", path="/blog", timestamp=2.0, user_agent="python-requests/2.31")

    assert first["ready"] is False
    assert second["ready"] is True
    assert isinstance(second["bot_probability"], float)
