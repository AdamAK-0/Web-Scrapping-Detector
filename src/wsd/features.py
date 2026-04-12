"""Incremental graph- and timing-aware feature extraction for navigation sessions."""

from __future__ import annotations

from collections import Counter
from statistics import mean, pstdev
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from .entropy import normalized_entropy, shannon_entropy
from .graph_builder import infer_category_from_path, shortest_path_length_or_fallback
from .types import PrefixFeatureRow, RequestEvent, Session


def extract_prefix_feature_rows(
    sessions: list[Session],
    graph: nx.DiGraph,
    prefixes: Iterable[int],
) -> list[PrefixFeatureRow]:
    rows: list[PrefixFeatureRow] = []
    prefixes = sorted(set(int(p) for p in prefixes if int(p) >= 2))
    for session in sessions:
        for prefix_len in prefixes:
            if len(session.events) < prefix_len:
                continue
            features = extract_features_for_prefix(session, graph, prefix_len)
            rows.append(
                PrefixFeatureRow(
                    session_id=session.session_id,
                    prefix_len=prefix_len,
                    label=session.label,
                    features=features,
                )
            )
    return rows


def prefix_rows_to_dataframe(rows: list[PrefixFeatureRow]) -> pd.DataFrame:
    records = []
    for row in rows:
        record = {
            "session_id": row.session_id,
            "prefix_len": row.prefix_len,
            "label": row.label,
        }
        record.update(row.features)
        records.append(record)
    return pd.DataFrame(records)


def extract_features_for_prefix(session: Session, graph: nx.DiGraph, prefix_len: int) -> dict[str, float]:
    return extract_features_for_events(session.events[:prefix_len], graph)


def extract_features_for_events(events: list[RequestEvent], graph: nx.DiGraph) -> dict[str, float]:
    if len(events) < 2:
        raise ValueError("At least two events are required to extract navigation features")

    paths = [event.path for event in events]
    deltas = [event.delta_t for event in events]
    categories = [
        event.page_category or graph.nodes.get(path, {}).get("category") or infer_category_from_path(path)
        for path, event in zip(paths, events, strict=False)
    ]
    transitions = list(zip(paths[:-1], paths[1:], strict=False))

    unique_nodes = len(set(paths))
    revisit_ratio = 1.0 - (unique_nodes / len(paths))
    repeated_transition_ratio = _repeated_transition_ratio(transitions)
    repeated_path_ratio = _repeated_path_ratio(paths)
    backtrack_count = sum(1 for a, b in zip(paths[:-2], paths[2:], strict=False) if a == b)
    self_loop_ratio = sum(1 for src, dst in transitions if src == dst) / max(1, len(transitions))

    depths = [graph.nodes.get(path, {}).get("depth", -1) for path in paths]
    mean_depth = _safe_mean([d for d in depths if d >= 0])
    leaf_visit_ratio = _leaf_visit_ratio(paths, graph)
    out_degree_mean = _safe_mean([graph.out_degree(path) if path in graph else 0 for path in paths])

    hop_distances = [shortest_path_length_or_fallback(graph, src, dst, fallback=999) for src, dst in transitions]
    mean_hop_distance = _safe_mean(hop_distances)
    far_jump_ratio = sum(1 for d in hop_distances if d >= 3) / max(1, len(hop_distances))

    timing_deltas = deltas[1:] if len(deltas) > 1 else deltas
    mean_delta_t = _safe_mean(timing_deltas)
    std_delta_t = _safe_std(timing_deltas)
    burstiness = _burstiness(timing_deltas)
    low_latency_ratio = sum(1 for d in timing_deltas if d <= 1.0) / max(1, len(timing_deltas))

    status_codes = [event.status_code for event in events if event.status_code is not None]
    error_rate = sum(1 for status in status_codes if int(status) >= 400) / max(1, len(status_codes)) if status_codes else 0.0

    user_agents = [event.user_agent for event in events if event.user_agent]
    user_agent_switch_rate = sum(1 for a, b in zip(user_agents[:-1], user_agents[1:], strict=False) if a != b) / max(1, len(user_agents) - 1) if len(user_agents) >= 2 else 0.0

    category_switch_rate = sum(1 for a, b in zip(categories[:-1], categories[1:], strict=False) if a != b) / max(1, len(categories) - 1)

    transition_entropy = shannon_entropy(transitions)
    normalized_transition_entropy = normalized_entropy(transitions)
    category_entropy = shannon_entropy(categories)
    normalized_category_entropy = normalized_entropy(categories)
    node_entropy = shannon_entropy(paths)
    normalized_node_entropy = normalized_entropy(paths)

    return {
        "session_length_so_far": float(len(paths)),
        "unique_nodes": float(unique_nodes),
        "unique_node_ratio": unique_nodes / len(paths),
        "revisit_ratio": revisit_ratio,
        "repeated_path_ratio": repeated_path_ratio,
        "repeated_transition_ratio": repeated_transition_ratio,
        "backtrack_count": float(backtrack_count),
        "self_loop_ratio": self_loop_ratio,
        "mean_depth": mean_depth,
        "leaf_visit_ratio": leaf_visit_ratio,
        "out_degree_mean": out_degree_mean,
        "mean_hop_distance": mean_hop_distance,
        "far_jump_ratio": far_jump_ratio,
        "mean_delta_t": mean_delta_t,
        "std_delta_t": std_delta_t,
        "burstiness": burstiness,
        "low_latency_ratio": low_latency_ratio,
        "error_rate": error_rate,
        "user_agent_switch_rate": user_agent_switch_rate,
        "category_switch_rate": category_switch_rate,
        "transition_entropy": transition_entropy,
        "normalized_transition_entropy": normalized_transition_entropy,
        "category_entropy": category_entropy,
        "normalized_category_entropy": normalized_category_entropy,
        "node_entropy": node_entropy,
        "normalized_node_entropy": normalized_node_entropy,
        "navigation_entropy_score": _navigation_entropy_score(
            normalized_transition_entropy=normalized_transition_entropy,
            normalized_category_entropy=normalized_category_entropy,
            revisit_ratio=revisit_ratio,
            far_jump_ratio=far_jump_ratio,
            category_switch_rate=category_switch_rate,
            burstiness=burstiness,
        ),
    }


def _repeated_transition_ratio(transitions: list[tuple[str, str]]) -> float:
    if not transitions:
        return 0.0
    counts = Counter(transitions)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(transitions)


def _repeated_path_ratio(paths: list[str]) -> float:
    if not paths:
        return 0.0
    counts = Counter(paths)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(paths)


def _leaf_visit_ratio(paths: list[str], graph: nx.DiGraph) -> float:
    if not paths:
        return 0.0
    leaf_visits = 0
    for path in paths:
        if path not in graph:
            continue
        if graph.out_degree(path) == 0:
            leaf_visits += 1
    return leaf_visits / len(paths)


def _safe_mean(values: list[float | int]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_std(values: list[float | int]) -> float:
    return float(pstdev(values)) if len(values) >= 2 else 0.0


def _burstiness(values: list[float | int]) -> float:
    if len(values) < 2:
        return 0.0
    m = _safe_mean(values)
    s = _safe_std(values)
    denominator = s + m
    if denominator == 0:
        return 0.0
    return (s - m) / denominator


def _navigation_entropy_score(
    normalized_transition_entropy: float,
    normalized_category_entropy: float,
    revisit_ratio: float,
    far_jump_ratio: float,
    category_switch_rate: float,
    burstiness: float,
) -> float:
    """Research-prototype score in [0, 1].

    Higher values indicate more exploratory, human-like browsing.
    Lower values indicate more structured, crawler-like traversal.
    """
    score = (
        0.30 * normalized_transition_entropy
        + 0.20 * normalized_category_entropy
        + 0.15 * (1.0 - revisit_ratio)
        + 0.15 * category_switch_rate
        + 0.10 * far_jump_ratio
        + 0.10 * (1.0 - abs(burstiness))
    )
    return float(np.clip(score, 0.0, 1.0))
