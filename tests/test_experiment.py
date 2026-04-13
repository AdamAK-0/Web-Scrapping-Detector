from pathlib import Path

import pandas as pd

from wsd.experiment import _build_session_metadata_from_sessions, build_leaderboard, run_experiments
from wsd.features import extract_features_for_events, extract_prefix_feature_rows, prefix_rows_to_dataframe
from wsd.graph_builder import build_graph_from_csv
from wsd.sessionizer import load_sessions_from_csv


def test_experiment_runner_produces_leaderboard(tmp_path: Path) -> None:
    data_dir = Path(__file__).resolve().parents[1] / 'data' / 'nginx_demo_prepared'
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


def test_experiment_runner_supports_harder_protocols_and_outputs(tmp_path: Path) -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "nginx_demo_prepared"
    graph = build_graph_from_csv(data_dir / "graph_edges.csv", data_dir / "graph_categories.csv")
    all_sessions = load_sessions_from_csv(data_dir / "requests.csv")
    human_sessions = [session for session in all_sessions if session.label == "human"][:8]
    bot_sessions = [session for session in all_sessions if session.label == "bot"][:8]
    sessions = human_sessions + bot_sessions
    feature_rows = extract_prefix_feature_rows(sessions, graph, prefixes=[3, 5, 10])
    feature_df = prefix_rows_to_dataframe(feature_rows)

    full_rows = []
    for session in sessions:
        if len(session.events) < 2:
            continue
        full_rows.append(
            {
                "session_id": session.session_id,
                "prefix_len": len(session.events),
                "label": session.label,
                **extract_features_for_events(session.events, graph),
            }
        )
    full_session_df = pd.DataFrame(full_rows)
    session_metadata = _build_session_metadata_from_sessions(sessions)
    human_ids = session_metadata[session_metadata["label"] == "human"]["session_id"].tolist()
    bot_ids = session_metadata[session_metadata["label"] == "bot"]["session_id"].tolist()
    base_participants = ["human_a", "human_b", "human_c", "human_d"]
    participant_ids = [base_participants[index % len(base_participants)] for index in range(len(human_ids))]
    session_metadata.loc[session_metadata["label"] == "human", "participant_id"] = participant_ids
    interleaved_ids = []
    for pair_index in range(max(len(human_ids), len(bot_ids))):
        if pair_index < len(human_ids):
            interleaved_ids.append(human_ids[pair_index])
        if pair_index < len(bot_ids):
            interleaved_ids.append(bot_ids[pair_index])
    timestamp_map = {session_id: float(index * 10) for index, session_id in enumerate(interleaved_ids)}
    session_metadata["start_timestamp"] = session_metadata["session_id"].map(timestamp_map).fillna(0.0)
    session_metadata["end_timestamp"] = session_metadata["start_timestamp"] + 5.0

    artifacts = run_experiments(
        feature_df,
        prefixes=[3, 5, 10],
        output_dir=tmp_path,
        n_bootstrap=2,
        random_state=42,
        save_models=False,
        session_metadata=session_metadata,
        full_session_df=full_session_df,
        selected_models=["logistic_regression"],
        protocols=["session_split", "hard_prefix_session_split", "time_split", "leave_one_human_user_out"],
        hard_prefixes=[3, 5],
    )

    leaderboard = build_leaderboard(artifacts)
    assert not leaderboard.empty
    assert {"session_split", "hard_prefix_session_split", "time_split", "leave_one_human_user_out"}.issubset(set(leaderboard["protocol"]))
    assert (tmp_path / "shortcut_audit.csv").exists()
    assert (tmp_path / "shortcut_red_flags.csv").exists()
    assert (tmp_path / "entropy_variant_comparison.csv").exists()
