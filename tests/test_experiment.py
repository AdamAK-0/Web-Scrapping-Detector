from pathlib import Path

import pandas as pd

from wsd.experiment import build_leaderboard, run_experiments
from wsd.features import extract_prefix_feature_rows, prefix_rows_to_dataframe
from wsd.graph_builder import build_graph_from_csv
from wsd.sessionizer import load_sessions_from_csv


def test_experiment_runner_produces_leaderboard(tmp_path: Path) -> None:
    data_dir = Path('/mnt/data/web_scraping_detector_project/data/nginx_demo_prepared')
    graph = build_graph_from_csv(data_dir / 'graph_edges.csv', data_dir / 'graph_categories.csv')
    sessions = load_sessions_from_csv(data_dir / 'requests.csv')
    feature_rows = extract_prefix_feature_rows(sessions, graph, prefixes=[3, 5, 10])
    feature_df = prefix_rows_to_dataframe(feature_rows)

    artifacts = run_experiments(
        feature_df,
        prefixes=[3, 5, 10],
        output_dir=tmp_path,
        n_bootstrap=20,
        random_state=42,
        save_models=False,
    )
    leaderboard = build_leaderboard(artifacts)
    assert isinstance(leaderboard, pd.DataFrame)
    assert not leaderboard.empty
    assert {'model_name', 'ablation_name', 'f1'}.issubset(leaderboard.columns)
