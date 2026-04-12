"""Synthetic website and session generator for reproducible experiments."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx
import pandas as pd

from .config import DEFAULT_NEGATIVE_LABEL, DEFAULT_POSITIVE_LABEL, DEFAULT_RANDOM_SEED, ROOT_NODE
from .graph_builder import annotate_graph_metadata, infer_category_from_path
from .types import RequestEvent, Session


@dataclass(slots=True)
class SyntheticDataset:
    graph: nx.DiGraph
    sessions: list[Session]


def generate_demo_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    categories = ["products", "blog", "support", "about"]
    graph.add_node(ROOT_NODE)

    for category in categories:
        category_root = f"/{category}"
        graph.add_edge(ROOT_NODE, category_root)

    for i in range(1, 7):
        graph.add_edge("/products", f"/products/category-{i}")
        for j in range(1, 8):
            product = f"/products/category-{i}/item-{j}"
            graph.add_edge(f"/products/category-{i}", product)
            graph.add_edge(product, f"/support/faq-{(i + j) % 6 + 1}")

    for i in range(1, 10):
        blog = f"/blog/post-{i}"
        graph.add_edge("/blog", blog)
        if i < 9:
            graph.add_edge(blog, f"/blog/post-{i+1}")
        graph.add_edge(blog, f"/products/category-{(i % 6) + 1}")

    for i in range(1, 7):
        faq = f"/support/faq-{i}"
        graph.add_edge("/support", faq)
        graph.add_edge(faq, "/support/contact")

    graph.add_edge("/about", "/about/team")
    graph.add_edge("/about", "/about/careers")
    graph.add_edge("/about/team", "/about/careers")
    graph.add_edge("/about/careers", "/support/contact")

    annotate_graph_metadata(graph, categories={node: infer_category_from_path(node) for node in graph.nodes})
    return graph


def generate_synthetic_dataset(
    num_humans: int = 160,
    num_bots_per_strategy: int = 80,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> SyntheticDataset:
    rng = random.Random(random_seed)
    graph = generate_demo_graph()
    sessions: list[Session] = []

    bot_strategies: list[Callable[[nx.DiGraph, random.Random, str], Session]] = [
        generate_bfs_bot_session,
        generate_dfs_bot_session,
        generate_linear_content_scraper_session,
        generate_randomized_stealth_bot_session,
    ]

    for i in range(num_humans):
        sessions.append(generate_human_session(graph, rng, session_id=f"human_{i:04d}"))

    counter = 0
    for strategy in bot_strategies:
        for _ in range(num_bots_per_strategy):
            sessions.append(strategy(graph, rng, session_id=f"bot_{counter:04d}"))
            counter += 1

    return SyntheticDataset(graph=graph, sessions=sessions)


def save_synthetic_dataset(dataset: SyntheticDataset, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_rows = [{"source": src, "target": dst} for src, dst in dataset.graph.edges]
    pd.DataFrame(edge_rows).to_csv(output_dir / "graph_edges.csv", index=False)

    category_rows = [
        {"path": node, "category": dataset.graph.nodes[node].get("category", infer_category_from_path(node))}
        for node in dataset.graph.nodes
    ]
    pd.DataFrame(category_rows).to_csv(output_dir / "graph_categories.csv", index=False)

    request_rows = []
    for session in dataset.sessions:
        for event in session.events:
            request_rows.append(
                {
                    "session_id": event.session_id,
                    "timestamp": event.timestamp,
                    "path": event.path,
                    "delta_t": event.delta_t,
                    "label": event.label,
                    "page_category": event.page_category,
                    "referrer": event.referrer,
                    "user_agent": event.user_agent,
                    "status_code": event.status_code,
                    **event.extra,
                }
            )
    pd.DataFrame(request_rows).to_csv(output_dir / "requests.csv", index=False)


def export_synthetic_nginx_logs(dataset: SyntheticDataset, output_dir: str | Path) -> None:
    """Export a realistic combined-log demo plus manual labels for dataset preparation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    label_rows: list[dict[str, str]] = []
    for index, session in enumerate(dataset.sessions):
        ip = f"203.0.113.{(index % 200) + 1}"
        user_agent = session.events[0].user_agent or ("Mozilla/5.0" if session.label == DEFAULT_NEGATIVE_LABEL else "python-requests/2.x")
        client_key = f"ip_ua:{ip}|{user_agent}"
        label_rows.append({"client_key": client_key, "label": session.label})
        for event in session.events:
            dt = pd.to_datetime(event.timestamp, unit="s", utc=True)
            timestamp_str = dt.strftime("%d/%b/%Y:%H:%M:%S +0000")
            referrer = event.referrer or "-"
            log_lines.append(
                f'{ip} - - [{timestamp_str}] "GET {event.path} HTTP/1.1" {event.status_code or 200} 512 "{referrer}" "{user_agent}"'
            )

    (output_dir / "access.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    pd.DataFrame(label_rows).drop_duplicates().to_csv(output_dir / "manual_labels.csv", index=False)


def generate_human_session(graph: nx.DiGraph, rng: random.Random, session_id: str) -> Session:
    length = rng.randint(8, 22)
    current = ROOT_NODE
    timestamp = 0.0
    events: list[RequestEvent] = []
    breadcrumb: list[str] = [ROOT_NODE]

    for step in range(length):
        if step == 0:
            next_path = ROOT_NODE
        else:
            neighbors = list(graph.successors(current))
            options = neighbors.copy()
            if len(breadcrumb) >= 2:
                options.append(breadcrumb[-2])
            if rng.random() < 0.18:
                options.extend(rng.sample(list(graph.nodes), k=min(3, graph.number_of_nodes())))
            next_path = rng.choice(options or [ROOT_NODE])

        delta_t = max(0.5, rng.gauss(5.0, 2.0))
        timestamp += delta_t
        events.append(_make_event(graph, session_id, timestamp, delta_t, next_path, DEFAULT_NEGATIVE_LABEL, current))

        if next_path != current:
            breadcrumb.append(next_path)
        current = next_path

    return Session(session_id=session_id, label=DEFAULT_NEGATIVE_LABEL, events=events)


def generate_bfs_bot_session(graph: nx.DiGraph, rng: random.Random, session_id: str) -> Session:
    order = list(nx.bfs_tree(graph, source=ROOT_NODE).nodes)
    return _make_scripted_session(graph, rng, session_id, order[: rng.randint(18, 35)], DEFAULT_POSITIVE_LABEL, mean_delay=0.6, std_delay=0.15)


def generate_dfs_bot_session(graph: nx.DiGraph, rng: random.Random, session_id: str) -> Session:
    order = list(nx.dfs_preorder_nodes(graph, source=ROOT_NODE))
    return _make_scripted_session(graph, rng, session_id, order[: rng.randint(18, 35)], DEFAULT_POSITIVE_LABEL, mean_delay=0.7, std_delay=0.18)


def generate_linear_content_scraper_session(graph: nx.DiGraph, rng: random.Random, session_id: str) -> Session:
    blog_chain = [ROOT_NODE, "/blog"] + [f"/blog/post-{i}" for i in range(1, 10)]
    return _make_scripted_session(graph, rng, session_id, blog_chain[: rng.randint(8, len(blog_chain))], DEFAULT_POSITIVE_LABEL, mean_delay=0.45, std_delay=0.1)


def generate_randomized_stealth_bot_session(graph: nx.DiGraph, rng: random.Random, session_id: str) -> Session:
    current = ROOT_NODE
    timestamp = 0.0
    events: list[RequestEvent] = []
    visited_edges: defaultdict[tuple[str, str], int] = defaultdict(int)
    length = rng.randint(12, 26)

    for step in range(length):
        if step == 0:
            next_path = ROOT_NODE
        else:
            neighbors = list(graph.successors(current))
            if not neighbors:
                current = ROOT_NODE
                neighbors = list(graph.successors(current))
            neighbors = sorted(neighbors, key=lambda path: visited_edges[(current, path)])
            next_path = neighbors[0]
            if rng.random() < 0.12:
                next_path = rng.choice(neighbors)

        delta_t = max(0.3, rng.gauss(1.3, 0.35))
        timestamp += delta_t
        events.append(_make_event(graph, session_id, timestamp, delta_t, next_path, DEFAULT_POSITIVE_LABEL, current))
        if step > 0:
            visited_edges[(current, next_path)] += 1
        current = next_path

    return Session(session_id=session_id, label=DEFAULT_POSITIVE_LABEL, events=events)


def _make_scripted_session(
    graph: nx.DiGraph,
    rng: random.Random,
    session_id: str,
    path_sequence: list[str],
    label: str,
    mean_delay: float,
    std_delay: float,
) -> Session:
    timestamp = 0.0
    events: list[RequestEvent] = []
    current = ROOT_NODE
    for path in path_sequence:
        delta_t = max(0.1, rng.gauss(mean_delay, std_delay))
        timestamp += delta_t
        events.append(_make_event(graph, session_id, timestamp, delta_t, path, label, current))
        current = path
    return Session(session_id=session_id, label=label, events=events)


def _make_event(graph: nx.DiGraph, session_id: str, timestamp: float, delta_t: float, path: str, label: str, referrer: str | None) -> RequestEvent:
    category = graph.nodes.get(path, {}).get("category", infer_category_from_path(path))
    user_agent = "Mozilla/5.0" if label == DEFAULT_NEGATIVE_LABEL else "python-requests/2.x"
    if label == DEFAULT_POSITIVE_LABEL and math.isclose(delta_t, 1.3, rel_tol=0.2):
        user_agent = "stealth-browser/1.0"
    return RequestEvent(
        session_id=session_id,
        timestamp=timestamp,
        path=path,
        delta_t=delta_t,
        label=label,
        page_category=category,
        referrer=referrer,
        user_agent=user_agent,
        status_code=200,
    )
