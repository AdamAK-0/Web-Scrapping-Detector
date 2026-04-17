from pathlib import Path

from generic_models.train_generic_models import build_prefix_feature_frame, generate_multisite_dataset, train_generic_models


def test_generic_pipeline_smoke(tmp_path: Path) -> None:
    import random

    sessions, _site_summary, sites = generate_multisite_dataset(num_sites=8, sessions_per_site=10, rng=random.Random(7))
    features = build_prefix_feature_frame(sessions, sites=sites, prefixes=[3, 5])

    leaderboard, prefix_metrics = train_generic_models(
        features,
        model_dir=tmp_path / "models",
        selected_models=["logistic_regression"],
        random_state=7,
    )

    assert not features.empty
    assert "coverage_ratio" in features.columns
    assert "path_entropy" in features.columns
    assert "entry_pagerank" in features.columns
    assert "bridge_visit_ratio" in features.columns
    assert leaderboard["model_name"].tolist() == ["logistic_regression"]
    assert set(prefix_metrics["prefix_len"]) == {3, 5}
    assert (tmp_path / "models" / "logistic_regression_generic_bundle.pkl").exists()
