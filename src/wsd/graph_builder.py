"""Helpers for building and annotating a website traversal graph."""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

from .config import ROOT_NODE
from .types import Session


def build_graph_from_edge_list(edges: Iterable[tuple[str, str]], categories: dict[str, str] | None = None) -> nx.DiGraph:
    graph = nx.DiGraph()
    for src, dst in edges:
        graph.add_edge(src, dst)
    categories = categories or {}
    annotate_graph_metadata(graph, categories=categories)
    return graph


def build_graph_from_csv(edge_csv_path: str | Path, categories_csv_path: str | Path | None = None) -> nx.DiGraph:
    edge_df = pd.read_csv(edge_csv_path)
    required = {"source", "target"}
    missing = required.difference(edge_df.columns)
    if missing:
        raise ValueError(f"Edge CSV missing required columns: {sorted(missing)}")

    categories: dict[str, str] = {}
    if categories_csv_path is not None and Path(categories_csv_path).exists():
        category_df = pd.read_csv(categories_csv_path)
        required_categories = {"path", "category"}
        missing = required_categories.difference(category_df.columns)
        if missing:
            raise ValueError(f"Category CSV missing required columns: {sorted(missing)}")
        categories = dict(zip(category_df["path"], category_df["category"], strict=False))

    graph = build_graph_from_edge_list(
        list(zip(edge_df["source"], edge_df["target"], strict=False)),
        categories=categories,
    )
    if "weight" in edge_df.columns:
        for row in edge_df.to_dict(orient="records"):
            graph[row["source"]][row["target"]]["weight"] = int(row["weight"])
    return graph


def build_graph_from_sessions(sessions: list[Session], *, use_referrers: bool = True) -> nx.DiGraph:
    edge_counter: Counter[tuple[str, str]] = Counter()
    categories: dict[str, str] = {}

    for session in sessions:
        paths = [event.path for event in session.events]
        for path in paths:
            categories[path] = infer_category_from_path(path)
        for src, dst in zip(paths[:-1], paths[1:], strict=False):
            edge_counter[(src, dst)] += 1
        if use_referrers:
            for event in session.events:
                if event.referrer and event.referrer.startswith("/") and event.referrer != event.path:
                    edge_counter[(event.referrer, event.path)] += 1

    graph = build_graph_from_edge_list(edge_counter.keys(), categories=categories)
    for (src, dst), weight in edge_counter.items():
        graph[src][dst]["weight"] = int(weight)
    return graph


def save_graph_to_csv(graph: nx.DiGraph, edge_csv_path: str | Path, categories_csv_path: str | Path | None = None) -> None:
    edge_rows = []
    for src, dst, data in graph.edges(data=True):
        edge_rows.append({"source": src, "target": dst, "weight": int(data.get("weight", 1))})
    pd.DataFrame(edge_rows).to_csv(edge_csv_path, index=False)

    if categories_csv_path is not None:
        category_rows = [{"path": node, "category": graph.nodes[node].get("category", infer_category_from_path(node))} for node in graph.nodes]
        pd.DataFrame(category_rows).to_csv(categories_csv_path, index=False)


def annotate_graph_metadata(graph: nx.DiGraph, categories: dict[str, str] | None = None) -> None:
    categories = categories or {}
    if ROOT_NODE not in graph:
        graph.add_node(ROOT_NODE)

    depths = compute_bfs_depths(graph, ROOT_NODE)
    for node in graph.nodes:
        graph.nodes[node]["category"] = categories.get(node, infer_category_from_path(node))
        graph.nodes[node]["depth"] = depths.get(node, -1)
        graph.nodes[node]["in_degree"] = graph.in_degree(node)
        graph.nodes[node]["out_degree"] = graph.out_degree(node)
        graph.nodes[node]["is_leaf"] = graph.out_degree(node) == 0


def compute_bfs_depths(graph: nx.DiGraph, root: str) -> dict[str, int]:
    depths: dict[str, int] = {root: 0}
    queue: deque[str] = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in graph.successors(current):
            if neighbor not in depths:
                depths[neighbor] = depths[current] + 1
                queue.append(neighbor)
    return depths


def infer_category_from_path(path: str) -> str:
    stripped = path.strip("/")
    if not stripped:
        return "home"
    first_segment = stripped.split("/")[0]
    return first_segment or "uncategorized"


def shortest_path_length_or_fallback(graph: nx.DiGraph, source: str, target: str, fallback: int = 999) -> int:
    if source == target:
        return 0
    try:
        return nx.shortest_path_length(graph, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return fallback
