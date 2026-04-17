"""Train a second, site-generic detector family without touching live-site models.

This script builds generic models from two sources:

1. A multi-site graph-navigation simulator that varies website topology,
   navigation intent, timing, and bot families across many synthetic websites.
2. Optional public Zenodo session-feature CSVs from:
   https://zenodo.org/records/3477932

Outputs stay under generic_models/artifacts by default.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from wsd.modeling import build_models


ZENODO_FILES = {
    "simple_features.csv": "https://zenodo.org/records/3477932/files/simple_features.csv?download=1",
    "semantic_features.csv": "https://zenodo.org/records/3477932/files/semantic_features.csv?download=1",
}

DEFAULT_MODEL_SET = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "calibrated_svm",
    "xgboost",
    "lightgbm",
    "catboost",
]

PREFIXES = [3, 5, 10, 15, 20]


@dataclass(frozen=True)
class GenericSite:
    site_id: str
    archetype: str
    graph: nx.DiGraph
    reachable_nodes: set[str]
    hub_threshold: float
    bridge_threshold: float


@dataclass(frozen=True)
class GenericSession:
    session_id: str
    site_id: str
    family: str
    label: str
    paths: list[str]
    timestamps: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train generic website graph-navigation bot detectors")
    parser.add_argument("--artifacts-dir", default="generic_models/artifacts")
    parser.add_argument("--external-dir", default="generic_models/data/external/zenodo")
    parser.add_argument("--num-sites", type=int, default=72)
    parser.add_argument("--sessions-per-site", type=int, default=90)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--download-public", action="store_true", help="Download small public Zenodo feature CSVs if missing")
    parser.add_argument("--skip-public", action="store_true", help="Skip the public Zenodo benchmark")
    parser.add_argument("--model-set", nargs="*", default=DEFAULT_MODEL_SET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    data_dir = artifacts_dir / "data"
    model_dir = artifacts_dir / "models"
    report_dir = artifacts_dir / "reports"
    for directory in (data_dir, model_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.download_public:
        download_public_datasets(Path(args.external_dir))

    rng = random.Random(args.random_state)
    sessions, site_summary, sites = generate_multisite_dataset(
        num_sites=args.num_sites,
        sessions_per_site=args.sessions_per_site,
        rng=rng,
    )
    prefix_df = build_prefix_feature_frame(sessions, sites=sites, prefixes=PREFIXES)
    prefix_df.to_csv(data_dir / "generic_prefix_features.csv", index=False)
    site_summary.to_csv(data_dir / "generic_site_summary.csv", index=False)
    pd.DataFrame([session_to_row(session) for session in sessions]).to_csv(data_dir / "generic_session_summary.csv", index=False)

    selected_models = [name for name in args.model_set if name.strip()]
    generic_leaderboard, generic_prefix_metrics = train_generic_models(
        prefix_df,
        model_dir=model_dir,
        selected_models=selected_models,
        random_state=args.random_state,
    )
    generic_leaderboard.to_csv(report_dir / "generic_leaderboard.csv", index=False)
    generic_prefix_metrics.to_csv(report_dir / "generic_prefix_metrics.csv", index=False)

    public_report = pd.DataFrame()
    if not args.skip_public:
        public_report = run_public_zenodo_benchmark(
            external_dir=Path(args.external_dir),
            report_dir=report_dir,
            selected_models=selected_models,
            random_state=args.random_state,
        )

    write_summary(
        report_dir / "summary.md",
        sessions=sessions,
        site_summary=site_summary,
        leaderboard=generic_leaderboard,
        prefix_metrics=generic_prefix_metrics,
        public_report=public_report,
        model_dir=model_dir,
    )
    print(f"Generic artifacts written to: {artifacts_dir.resolve()}")
    print(f"Models: {model_dir.resolve()}")
    print(f"Summary: {(report_dir / 'summary.md').resolve()}")
    print(generic_leaderboard.head(12).to_string(index=False))


def download_public_datasets(external_dir: Path) -> None:
    external_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in ZENODO_FILES.items():
        destination = external_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue
        print(f"Downloading {filename} from Zenodo...")
        urllib.request.urlretrieve(url, destination)


def generate_multisite_dataset(*, num_sites: int, sessions_per_site: int, rng: random.Random) -> tuple[list[GenericSession], pd.DataFrame, dict[str, GenericSite]]:
    archetypes = ["commerce", "news", "docs", "support", "forum", "catalog", "marketing"]
    sessions: list[GenericSession] = []
    sites: dict[str, GenericSite] = {}
    site_rows: list[dict[str, object]] = []
    for site_index in range(num_sites):
        archetype = archetypes[site_index % len(archetypes)]
        site = make_random_site(site_id=f"site_{site_index:03d}", archetype=archetype, rng=rng)
        sites[site.site_id] = site
        site_rows.append(
            {
                "site_id": site.site_id,
                "archetype": site.archetype,
                "nodes": site.graph.number_of_nodes(),
                "edges": site.graph.number_of_edges(),
                "reachable_nodes": len(site.reachable_nodes),
                "density": nx.density(site.graph),
            }
        )
        for session_index in range(sessions_per_site):
            label = "human" if session_index % 2 == 0 else "bot"
            session = generate_session(site, session_index=session_index, label=label, rng=rng)
            sessions.append(session)
    return sessions, pd.DataFrame(site_rows), sites


def make_random_site(*, site_id: str, archetype: str, rng: random.Random) -> GenericSite:
    graph = nx.DiGraph()
    root = f"/{site_id}/"
    graph.add_node(root, category="home")
    category_count = rng.randint(5, 10)
    detail_range = {
        "commerce": (8, 18),
        "news": (10, 24),
        "docs": (6, 16),
        "support": (5, 13),
        "forum": (8, 20),
        "catalog": (10, 22),
        "marketing": (4, 10),
    }[archetype]
    categories = [f"{archetype}_{i}" for i in range(category_count)]
    listing_nodes: list[str] = []
    detail_nodes: list[str] = []
    utility_nodes = [f"/{site_id}/search", f"/{site_id}/about", f"/{site_id}/contact", f"/{site_id}/cart", f"/{site_id}/faq"]
    for utility in utility_nodes:
        graph.add_edge(root, utility)
        graph.nodes[utility]["category"] = utility.rsplit("/", 1)[-1]
        if rng.random() < 0.75:
            graph.add_edge(utility, root)

    for category in categories:
        listing = f"/{site_id}/{category}"
        listing_nodes.append(listing)
        graph.add_edge(root, listing)
        graph.nodes[listing]["category"] = "listing"
        if rng.random() < 0.85:
            graph.add_edge(listing, root)
        for item in range(rng.randint(*detail_range)):
            detail = f"{listing}/item_{item:03d}"
            detail_nodes.append(detail)
            graph.add_edge(listing, detail)
            graph.nodes[detail]["category"] = "detail"
            if rng.random() < 0.65:
                graph.add_edge(detail, listing)
            if rng.random() < 0.22:
                graph.add_edge(detail, rng.choice(utility_nodes))

    all_nodes = list(graph.nodes)
    for node in all_nodes:
        if node == root:
            continue
        for _ in range(rng.randint(0, 3)):
            if rng.random() < 0.35:
                target_pool = detail_nodes if rng.random() < 0.55 else listing_nodes + utility_nodes + [root]
                if target_pool:
                    graph.add_edge(node, rng.choice(target_pool))

    annotate_site_graph(graph, root=root)
    reachable_nodes = set(nx.descendants(graph, root)) | {root}
    out_degrees = [graph.out_degree(node) for node in graph.nodes]
    betweenness = [float(graph.nodes[node].get("betweenness", 0.0)) for node in graph.nodes]
    hub_threshold = float(np.quantile(out_degrees, 0.75)) if out_degrees else 0.0
    bridge_threshold = float(np.quantile(betweenness, 0.75)) if betweenness else 0.0
    return GenericSite(
        site_id=site_id,
        archetype=archetype,
        graph=graph,
        reachable_nodes=reachable_nodes,
        hub_threshold=hub_threshold,
        bridge_threshold=bridge_threshold,
    )


def annotate_site_graph(graph: nx.DiGraph, *, root: str) -> None:
    depths = nx.single_source_shortest_path_length(graph, root)
    pagerank = nx.pagerank(graph, alpha=0.85) if graph.number_of_nodes() > 1 else {root: 1.0}
    betweenness = nx.betweenness_centrality(graph, normalized=True) if graph.number_of_nodes() > 1 else {root: 0.0}
    for node in graph.nodes:
        graph.nodes[node]["depth"] = int(depths.get(node, -1))
        graph.nodes[node]["pagerank"] = float(pagerank.get(node, 0.0))
        graph.nodes[node]["betweenness"] = float(betweenness.get(node, 0.0))
        graph.nodes[node]["in_degree"] = int(graph.in_degree(node))
        graph.nodes[node]["out_degree"] = int(graph.out_degree(node))
        graph.nodes[node]["is_leaf"] = float(graph.out_degree(node) == 0)


def generate_session(site: GenericSite, *, session_index: int, label: str, rng: random.Random) -> GenericSession:
    family = choose_family(label, rng)
    paths = generate_human_path(site, rng) if label == "human" else generate_bot_path(site, family=family, rng=rng)
    timestamps = make_timestamps(paths, label=label, family=family, rng=rng)
    return GenericSession(
        session_id=f"{site.site_id}_{label}_{session_index:04d}",
        site_id=site.site_id,
        family=family,
        label=label,
        paths=paths,
        timestamps=timestamps,
    )


def choose_family(label: str, rng: random.Random) -> str:
    if label == "human":
        return rng.choice(["goal_oriented", "exploratory", "comparison", "support_seeking", "distracted"])
    return rng.choice(["bfs", "dfs", "linear", "focused", "browser_like", "noisy", "deep_harvest"])


def generate_human_path(site: GenericSite, rng: random.Random) -> list[str]:
    graph = site.graph
    root = f"/{site.site_id}/"
    nodes = list(site.reachable_nodes)
    current = root if rng.random() < 0.72 else rng.choice(nodes)
    path = [current]
    target_len = rng.randint(5, 28)
    for _ in range(target_len - 1):
        neighbors = list(graph.successors(current))
        if len(path) >= 3 and rng.random() < 0.18:
            current = path[-2]
        elif rng.random() < 0.13:
            shallow = [node for node in nodes if 0 <= graph.nodes[node].get("depth", 0) <= 2]
            current = rng.choice(shallow or nodes)
        elif neighbors:
            weights = []
            for neighbor in neighbors:
                depth = max(0, int(graph.nodes[neighbor].get("depth", 0)))
                weights.append(1.3 / (1.0 + depth * 0.25) + rng.random())
            current = rng.choices(neighbors, weights=weights, k=1)[0]
        else:
            current = rng.choice(nodes)
        path.append(current)
    return path


def generate_bot_path(site: GenericSite, *, family: str, rng: random.Random) -> list[str]:
    graph = site.graph
    root = f"/{site.site_id}/"
    nodes = list(site.reachable_nodes)
    target_len = rng.randint(12, 46)
    if family == "bfs":
        ordered = bfs_order(graph, root)
    elif family == "dfs":
        ordered = dfs_order(graph, root)
    elif family == "linear":
        ordered = sorted(nodes, key=lambda node: (graph.nodes[node].get("depth", 0), node))
    elif family == "focused":
        prefix = rng.choice([node for node in nodes if graph.nodes[node].get("category") == "listing"] or nodes)
        ordered = [prefix] + [node for node in nodes if node.startswith(prefix + "/")]
    elif family == "deep_harvest":
        ordered = sorted(nodes, key=lambda node: graph.nodes[node].get("depth", 0), reverse=True)
    elif family == "browser_like":
        ordered = browser_like_bot_order(site, rng, target_len)
    else:
        ordered = noisy_bot_order(site, rng, target_len)
    if not ordered:
        ordered = [root]
    path = []
    while len(path) < target_len:
        for node in ordered:
            path.append(node)
            if len(path) >= target_len:
                break
        if family in {"noisy", "browser_like"}:
            path.append(rng.choice(path[-5:] or ordered))
        else:
            break
    return path[:target_len]


def bfs_order(graph: nx.DiGraph, root: str) -> list[str]:
    seen = {root}
    queue: deque[str] = deque([root])
    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for neighbor in sorted(graph.successors(node)):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return ordered


def dfs_order(graph: nx.DiGraph, root: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    def visit(node: str) -> None:
        if node in seen:
            return
        seen.add(node)
        ordered.append(node)
        for neighbor in sorted(graph.successors(node)):
            visit(neighbor)

    visit(root)
    return ordered


def browser_like_bot_order(site: GenericSite, rng: random.Random, target_len: int) -> list[str]:
    graph = site.graph
    nodes = list(site.reachable_nodes)
    listings = [node for node in nodes if graph.nodes[node].get("category") == "listing"]
    details = [node for node in nodes if graph.nodes[node].get("category") == "detail"]
    utilities = [node for node in nodes if graph.nodes[node].get("depth", 0) <= 2]
    ordered = [f"/{site.site_id}/"]
    for _ in range(target_len):
        bucket = rng.choices([details, listings, utilities, nodes], weights=[5, 3, 2, 1], k=1)[0]
        if bucket:
            ordered.append(rng.choice(bucket))
        if len(ordered) >= 3 and rng.random() < 0.2:
            ordered.append(ordered[-2])
    return ordered


def noisy_bot_order(site: GenericSite, rng: random.Random, target_len: int) -> list[str]:
    base = bfs_order(site.graph, f"/{site.site_id}/")
    rng.shuffle(base)
    if len(base) < target_len:
        base.extend(rng.choices(base or list(site.reachable_nodes), k=target_len - len(base)))
    return base


def make_timestamps(paths: list[str], *, label: str, family: str, rng: random.Random) -> list[float]:
    timestamps = [0.0]
    for _ in paths[1:]:
        if label == "human":
            delta = rng.lognormvariate(1.7, 0.85)
            if rng.random() < 0.12:
                delta += rng.uniform(18.0, 90.0)
        elif family in {"browser_like", "noisy"}:
            delta = rng.lognormvariate(0.8, 0.7)
            if rng.random() < 0.10:
                delta += rng.uniform(4.0, 18.0)
        else:
            delta = rng.lognormvariate(-0.55, 0.45)
        timestamps.append(timestamps[-1] + max(0.05, delta))
    return timestamps


def build_prefix_feature_frame(sessions: list[GenericSession], *, sites: dict[str, GenericSite] | None = None, prefixes: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    site_graphs = sites or build_site_registry_from_sessions(sessions)
    for session in sessions:
        site = site_graphs[session.site_id]
        for prefix in prefixes:
            if len(session.paths) < prefix:
                continue
            features = extract_generic_features(session, site, prefix_len=prefix)
            row = {
                "session_id": session.session_id,
                "site_id": session.site_id,
                "family": session.family,
                "label": session.label,
                "prefix_len": prefix,
            }
            row.update(features)
            rows.append(row)
    return pd.DataFrame(rows)


def build_site_registry_from_sessions(sessions: list[GenericSession]) -> dict[str, GenericSite]:
    # Sessions are generated in-process, and this fallback reconstructs a conservative graph
    # from observed transitions so persisted feature extraction stays self-contained.
    registry: dict[str, nx.DiGraph] = {}
    for session in sessions:
        graph = registry.setdefault(session.site_id, nx.DiGraph())
        for path in session.paths:
            graph.add_node(path, category=infer_generic_category(path))
        for src, dst in zip(session.paths[:-1], session.paths[1:]):
            graph.add_edge(src, dst)
    sites: dict[str, GenericSite] = {}
    for site_id, graph in registry.items():
        root = f"/{site_id}/"
        if root not in graph:
            graph.add_node(root, category="home")
        annotate_site_graph(graph, root=root)
        reachable_nodes = set(nx.descendants(graph, root)) | {root}
        if len(reachable_nodes) <= 1:
            reachable_nodes = set(graph.nodes)
        out_degrees = [graph.out_degree(node) for node in graph.nodes]
        betweenness = [float(graph.nodes[node].get("betweenness", 0.0)) for node in graph.nodes]
        sites[site_id] = GenericSite(
            site_id=site_id,
            archetype="observed",
            graph=graph,
            reachable_nodes=reachable_nodes,
            hub_threshold=float(np.quantile(out_degrees, 0.75)) if out_degrees else 0.0,
            bridge_threshold=float(np.quantile(betweenness, 0.75)) if betweenness else 0.0,
        )
    return sites


def infer_generic_category(path: str) -> str:
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) <= 1:
        return "home"
    if parts[-1].startswith("item_"):
        return "detail"
    return "listing" if len(parts) == 2 else "detail"


def extract_generic_features(session: GenericSession, site: GenericSite, *, prefix_len: int) -> dict[str, float]:
    graph = site.graph
    paths = session.paths[:prefix_len]
    timestamps = session.timestamps[:prefix_len]
    transitions = list(zip(paths[:-1], paths[1:]))
    reachable_count = max(1, len(site.reachable_nodes))
    unique_paths = len(set(paths))
    deltas = [max(0.0, b - a) for a, b in zip(timestamps[:-1], timestamps[1:])]
    depths = [float(graph.nodes[path].get("depth", -1)) for path in paths]
    out_degrees = [float(graph.out_degree(path)) if path in graph else 0.0 for path in paths]
    in_degrees = [float(graph.in_degree(path)) if path in graph else 0.0 for path in paths]
    pageranks = [float(graph.nodes[path].get("pagerank", 0.0)) for path in paths]
    betweenness = [float(graph.nodes[path].get("betweenness", 0.0)) for path in paths]
    distances = [shortest_distance(graph, src, dst) for src, dst in transitions]
    reverse_edges = sum(1 for (a, b), (c, d) in zip(transitions[:-1], transitions[1:]) if a == d and b == c)
    backtracks = sum(1 for a, c in zip(paths[:-2], paths[2:]) if a == c)
    hub_visits = sum(1 for path in paths if graph.out_degree(path) >= site.hub_threshold and graph.out_degree(path) > 1)
    leaf_visits = sum(1 for path in paths if graph.out_degree(path) == 0)
    bridge_visits = sum(1 for path in paths if float(graph.nodes[path].get("betweenness", 0.0)) >= site.bridge_threshold and site.bridge_threshold > 0)
    branch = branching_features(paths, graph)
    entry = paths[0]
    exit_node = paths[-1]
    feature_row = {
        "session_length_so_far": float(len(paths)),
        "coverage_ratio": unique_paths / reachable_count,
        "path_entropy": shannon_entropy(paths),
        "normalized_path_entropy": normalized_entropy(paths, support_size=min(reachable_count, max(2, len(paths)))),
        "revisit_rate": 1.0 - unique_paths / max(1, len(paths)),
        "unique_nodes": float(unique_paths),
        "unique_node_ratio": unique_paths / max(1, len(paths)),
        "transition_entropy": shannon_entropy(transitions),
        "normalized_transition_entropy": normalized_entropy(transitions, support_size=max(2, min(len(set(transitions)) + 1, graph.number_of_edges() or 2))),
        "edge_revisit_rate": 1.0 - len(set(transitions)) / max(1, len(transitions)),
        "depth_mean": safe_mean(depths),
        "depth_std": safe_std(depths),
        "depth_max": max(depths) if depths else 0.0,
        "depth_distribution_entropy": shannon_entropy([int(depth) for depth in depths]),
        "deep_visit_ratio": sum(1 for depth in depths if depth >= 3) / max(1, len(depths)),
        "branching_entropy": branch["branching_entropy"],
        "branching_concentration": branch["branching_concentration"],
        "outbound_coverage_mean": branch["outbound_coverage_mean"],
        "deterministic_branch_ratio": branch["deterministic_branch_ratio"],
        "inter_hop_time_mean": safe_mean(deltas),
        "inter_hop_time_std": safe_std(deltas),
        "inter_hop_time_cv": safe_std(deltas) / max(1e-9, safe_mean(deltas)),
        "inter_hop_burstiness": burstiness(deltas),
        "low_latency_ratio": sum(1 for delta in deltas if delta <= 1.0) / max(1, len(deltas)),
        "entry_out_degree_norm": graph.out_degree(entry) / max(1, graph.number_of_nodes()),
        "entry_in_degree_norm": graph.in_degree(entry) / max(1, graph.number_of_nodes()),
        "entry_pagerank": float(graph.nodes[entry].get("pagerank", 0.0)),
        "entry_is_hub": float(graph.out_degree(entry) >= site.hub_threshold and graph.out_degree(entry) > 1),
        "entry_is_leaf": float(graph.out_degree(entry) == 0),
        "entry_is_bridge": float(float(graph.nodes[entry].get("betweenness", 0.0)) >= site.bridge_threshold and site.bridge_threshold > 0),
        "exit_out_degree_norm": graph.out_degree(exit_node) / max(1, graph.number_of_nodes()),
        "exit_in_degree_norm": graph.in_degree(exit_node) / max(1, graph.number_of_nodes()),
        "exit_pagerank": float(graph.nodes[exit_node].get("pagerank", 0.0)),
        "exit_is_hub": float(graph.out_degree(exit_node) >= site.hub_threshold and graph.out_degree(exit_node) > 1),
        "exit_is_leaf": float(graph.out_degree(exit_node) == 0),
        "exit_is_bridge": float(float(graph.nodes[exit_node].get("betweenness", 0.0)) >= site.bridge_threshold and site.bridge_threshold > 0),
        "graph_distance_mean": safe_mean(distances),
        "graph_distance_std": safe_std(distances),
        "graph_distance_max": max(distances) if distances else 0.0,
        "graph_distance_total": float(sum(distances)),
        "far_jump_ratio": sum(1 for distance in distances if distance >= 3) / max(1, len(distances)),
        "backtrack_ratio": backtracks / max(1, len(paths) - 2),
        "reverse_direction_ratio": reverse_edges / max(1, len(transitions) - 1),
        "hub_visit_ratio": hub_visits / max(1, len(paths)),
        "leaf_visit_ratio": leaf_visits / max(1, len(paths)),
        "bridge_visit_ratio": bridge_visits / max(1, len(paths)),
        "visited_pagerank_mean": safe_mean(pageranks),
        "visited_betweenness_mean": safe_mean(betweenness),
        "visited_out_degree_mean": safe_mean(out_degrees),
        "visited_in_degree_mean": safe_mean(in_degrees),
        "mean_depth": safe_mean(depths),
        "out_degree_mean": safe_mean(out_degrees),
        "mean_hop_distance": safe_mean(distances),
    }
    feature_row["navigation_entropy_generic_score"] = generic_navigation_score(feature_row)
    return feature_row


def branching_features(paths: list[str], graph: nx.DiGraph) -> dict[str, float]:
    choices_by_source: dict[str, list[str]] = {}
    for src, dst in zip(paths[:-1], paths[1:]):
        choices_by_source.setdefault(src, []).append(dst)
    entropies = []
    coverages = []
    deterministic = 0
    for src, choices in choices_by_source.items():
        out_degree = max(1, graph.out_degree(src))
        entropies.append(normalized_entropy(choices, support_size=max(2, out_degree)))
        coverages.append(len(set(choices)) / out_degree)
        if len(set(choices)) == 1 and out_degree > 1:
            deterministic += 1
    return {
        "branching_entropy": safe_mean(entropies),
        "branching_concentration": 1.0 - safe_mean(entropies),
        "outbound_coverage_mean": safe_mean(coverages),
        "deterministic_branch_ratio": deterministic / max(1, len(choices_by_source)),
    }


def train_generic_models(
    df: pd.DataFrame,
    *,
    model_dir: Path,
    selected_models: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_ids = sorted(df["site_id"].unique())
    rng = random.Random(random_state)
    rng.shuffle(site_ids)
    n = len(site_ids)
    train_sites = set(site_ids[: int(n * 0.60)])
    val_sites = set(site_ids[int(n * 0.60) : int(n * 0.80)])
    test_sites = set(site_ids[int(n * 0.80) :])
    train_df = df[df["site_id"].isin(train_sites)].copy()
    val_df = df[df["site_id"].isin(val_sites)].copy()
    test_df = df[df["site_id"].isin(test_sites)].copy()
    train_val_df = df[df["site_id"].isin(train_sites | val_sites)].copy()
    feature_columns = [col for col in df.columns if col not in {"session_id", "site_id", "family", "label"}]
    models = prepare_generic_models(build_models(random_state=random_state, selected_models=selected_models))

    leaderboard_rows = []
    prefix_rows = []
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        X_train = train_df[feature_columns]
        y_train = (train_df["label"] == "bot").astype(int)
        X_val = val_df[feature_columns]
        y_val = (val_df["label"] == "bot").astype(int)
        model.fit(X_train, y_train)
        val_score = predict_proba(model, X_val)
        threshold = tune_threshold(y_val.to_numpy(), val_score)

        final_model = prepare_generic_models(build_models(random_state=random_state, selected_models=[model_name]))[model_name]
        final_model.fit(train_val_df[feature_columns], (train_val_df["label"] == "bot").astype(int))
        X_test = test_df[feature_columns]
        y_test = (test_df["label"] == "bot").astype(int).to_numpy()
        test_score = predict_proba(final_model, X_test)
        test_pred = (test_score >= threshold).astype(int)
        metrics = classification_metrics(y_test, test_pred, test_score)
        leaderboard_rows.append(
            {
                "model_name": model_name,
                "dataset": "generic_multisite_leave_site_out",
                "threshold": threshold,
                "train_sites": len(train_sites),
                "validation_sites": len(val_sites),
                "test_sites": len(test_sites),
                **metrics,
            }
        )
        for prefix, group in test_df.assign(score=test_score, pred=test_pred).groupby("prefix_len"):
            y_prefix = (group["label"] == "bot").astype(int).to_numpy()
            score_prefix = group["score"].to_numpy()
            pred_prefix = group["pred"].to_numpy()
            prefix_rows.append({"model_name": model_name, "prefix_len": int(prefix), **classification_metrics(y_prefix, pred_prefix, score_prefix)})
        bundle = {
            "bundle_type": "generic_website_detector",
            "model_name": model_name,
            "feature_columns": feature_columns,
            "model": final_model,
            "threshold": threshold,
            "training_scope": "multi_site_graph_navigation",
            "notes": "Generic model trained on normalized graph/session features across simulated website archetypes.",
        }
        with (model_dir / f"{model_name}_generic_bundle.pkl").open("wb") as handle:
            pickle.dump(bundle, handle)
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)
    prefix_metrics = pd.DataFrame(prefix_rows).sort_values(["model_name", "prefix_len"]).reset_index(drop=True)
    return leaderboard, prefix_metrics


def run_public_zenodo_benchmark(
    *,
    external_dir: Path,
    report_dir: Path,
    selected_models: list[str],
    random_state: int,
) -> pd.DataFrame:
    simple_path = external_dir / "simple_features.csv"
    semantic_path = external_dir / "semantic_features.csv"
    if not simple_path.exists() or not semantic_path.exists():
        return pd.DataFrame([{"dataset": "zenodo_public_session_features", "status": "missing; run with --download-public"}])
    simple = pd.read_csv(simple_path)
    semantic = pd.read_csv(semantic_path)
    simple = simple.dropna(subset=["ROBOT"]).copy()
    semantic = semantic.drop(columns=["ROBOT"], errors="ignore")
    df = simple.merge(semantic, on="ID", how="left")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["ROBOT"].astype(int).to_numpy()
    feature_columns = [col for col in df.columns if col not in {"ID", "ROBOT"}]
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    train_end = int(len(indices) * 0.70)
    val_end = int(len(indices) * 0.85)
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    rows = []
    models = prepare_generic_models(build_models(random_state=random_state, selected_models=selected_models))
    for model_name, model in models.items():
        model.fit(df.iloc[train_idx][feature_columns], y[train_idx])
        val_score = predict_proba(model, df.iloc[val_idx][feature_columns])
        threshold = tune_threshold(y[val_idx], val_score)
        score = predict_proba(model, df.iloc[test_idx][feature_columns])
        pred = (score >= threshold).astype(int)
        rows.append(
            {
                "dataset": "zenodo_public_session_features",
                "model_name": model_name,
                "threshold": threshold,
                "feature_count": len(feature_columns),
                **classification_metrics(y[test_idx], pred, score),
            }
        )
    report = pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)
    report.to_csv(report_dir / "public_zenodo_benchmark.csv", index=False)
    return report


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    metrics["pr_auc"] = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")
    return metrics


def prepare_generic_models(models: dict[str, object]) -> dict[str, object]:
    for model_name, model in models.items():
        if model_name == "catboost" and hasattr(model, "set_params"):
            model.set_params(allow_writing_files=False)
    return models


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(X), dtype=float)


def session_to_row(session: GenericSession) -> dict[str, object]:
    return {
        "session_id": session.session_id,
        "site_id": session.site_id,
        "family": session.family,
        "label": session.label,
        "num_events": len(session.paths),
        "duration_seconds": session.timestamps[-1] - session.timestamps[0] if len(session.timestamps) >= 2 else 0.0,
        "first_path": session.paths[0] if session.paths else "",
        "last_path": session.paths[-1] if session.paths else "",
    }


def write_summary(
    path: Path,
    *,
    sessions: list[GenericSession],
    site_summary: pd.DataFrame,
    leaderboard: pd.DataFrame,
    prefix_metrics: pd.DataFrame,
    public_report: pd.DataFrame,
    model_dir: Path,
) -> None:
    label_counts = Counter(session.label for session in sessions)
    family_counts = Counter(session.family for session in sessions)
    lines = [
        "# Generic Website Detector Results",
        "",
        "This is a separate model family from the live-site thesis models.",
        "",
        "## Data",
        "",
        f"- Simulated websites: {len(site_summary)}",
        f"- Simulated sessions: {len(sessions)}",
        f"- Label counts: {dict(label_counts)}",
        f"- Families: {dict(sorted(family_counts.items()))}",
        "",
        "## Generic Leave-Site-Out Leaderboard",
        "",
        dataframe_block(leaderboard.head(12)),
        "",
        "## Prefix Metrics",
        "",
        dataframe_block(prefix_metrics.head(40)),
        "",
        "## Public Zenodo Benchmark",
        "",
        dataframe_block(public_report.head(12)) if not public_report.empty else "Skipped or unavailable.",
        "",
        "## Models",
        "",
        f"Saved under `{model_dir}`.",
        "",
        "## Feature Coverage",
        "",
        "The generic feature set includes coverage ratio, path entropy, revisit rate, depth distribution, branching decision patterns, inter-hop timing, entry/exit centrality, graph distance traveled, backtrack ratio, and structural roles such as hub/leaf/bridge visits.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    return "```text\n" + df.to_string(index=False) + "\n```"


def generic_navigation_score(features: dict[str, float]) -> float:
    score = (
        0.18 * features["normalized_path_entropy"]
        + 0.14 * features["normalized_transition_entropy"]
        + 0.14 * features["branching_entropy"]
        + 0.12 * (1.0 - features["revisit_rate"])
        + 0.10 * min(1.0, features["coverage_ratio"] * 4.0)
        + 0.10 * (1.0 - min(1.0, features["low_latency_ratio"]))
        + 0.08 * (1.0 - min(1.0, features["deterministic_branch_ratio"]))
        + 0.07 * min(1.0, features["graph_distance_mean"] / 4.0)
        + 0.07 * (1.0 - min(1.0, features["backtrack_ratio"]))
    )
    return float(np.clip(score, 0.0, 1.0))


def shannon_entropy(values: Iterable[object]) -> float:
    values = list(values)
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def normalized_entropy(values: Iterable[object], *, support_size: int | None = None) -> float:
    values = list(values)
    if not values:
        return 0.0
    support = support_size or len(set(values))
    if support <= 1:
        return 0.0
    return float(np.clip(shannon_entropy(values) / math.log2(support), 0.0, 1.0))


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(mean(values)) if values else 0.0


def safe_std(values: Iterable[float]) -> float:
    values = list(values)
    return float(pstdev(values)) if len(values) >= 2 else 0.0


def burstiness(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    m = safe_mean(values)
    s = safe_std(values)
    return 0.0 if s + m == 0 else float((s - m) / (s + m))


def shortest_distance(graph: nx.DiGraph, source: str, target: str) -> float:
    if source == target:
        return 0.0
    try:
        return float(nx.shortest_path_length(graph, source=source, target=target))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float(min(8, graph.number_of_nodes()))


if __name__ == "__main__":
    main()
