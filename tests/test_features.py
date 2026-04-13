from wsd.features import extract_features_for_prefix
from wsd.synthetic_data import generate_demo_graph
from wsd.types import RequestEvent, Session


def test_extract_features_contains_navigation_entropy_score() -> None:
    graph = generate_demo_graph()
    session = Session(
        session_id="s1",
        label="human",
        events=[
            RequestEvent("s1", 1.0, "/", 1.0, "human", page_category="home"),
            RequestEvent("s1", 3.0, "/products", 2.0, "human", page_category="products"),
            RequestEvent("s1", 6.0, "/products/category-1", 3.0, "human", page_category="products"),
            RequestEvent("s1", 9.0, "/products/category-1/item-1", 3.0, "human", page_category="products"),
        ],
    )
    features = extract_features_for_prefix(session, graph, prefix_len=4)
    assert "navigation_entropy_score" in features
    assert "navigation_entropy_score_v2" in features
    assert "local_branching_entropy" in features
    assert "category_transition_entropy" in features
    assert "entropy_slope" in features
    assert 0.0 <= features["navigation_entropy_score"] <= 1.0
    assert 0.0 <= features["navigation_entropy_score_v2"] <= 1.0
    assert features["unique_nodes"] == 4.0
