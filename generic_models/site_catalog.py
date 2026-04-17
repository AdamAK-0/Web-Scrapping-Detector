"""Four local generic websites with intentionally different graph shapes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from generic_models.train_generic_models import GenericSite, annotate_site_graph


@dataclass(frozen=True)
class PageSpec:
    path: str
    title: str
    category: str
    links: tuple[str, ...]
    body: str


@dataclass(frozen=True)
class WebsiteSpec:
    site_id: str
    name: str
    archetype: str
    port: int
    shape: str
    accent: str
    pages: tuple[PageSpec, ...]


def get_websites() -> dict[str, WebsiteSpec]:
    specs = [
        _atlas_shop(),
        _deep_docs(),
        _news_mesh(),
        _support_funnel(),
    ]
    return {spec.site_id: spec for spec in specs}


def build_generic_site(spec: WebsiteSpec) -> GenericSite:
    graph = nx.DiGraph()
    for page in spec.pages:
        graph.add_node(page.path, category=page.category)
    for page in spec.pages:
        for target in page.links:
            if target in graph:
                graph.add_edge(page.path, target)
    root = root_path(spec.site_id)
    annotate_site_graph(graph, root=root)
    reachable_nodes = set(nx.descendants(graph, root)) | {root}
    out_degrees = [graph.out_degree(node) for node in graph.nodes]
    betweenness = [float(graph.nodes[node].get("betweenness", 0.0)) for node in graph.nodes]
    return GenericSite(
        site_id=spec.site_id,
        archetype=spec.archetype,
        graph=graph,
        reachable_nodes=reachable_nodes,
        hub_threshold=float(np.quantile(out_degrees, 0.75)) if out_degrees else 0.0,
        bridge_threshold=float(np.quantile(betweenness, 0.75)) if betweenness else 0.0,
    )


def build_all_generic_sites() -> dict[str, GenericSite]:
    return {site_id: build_generic_site(spec) for site_id, spec in get_websites().items()}


def write_graph_exports(output_dir: str | Path = "generic_models/live_graphs") -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for spec in get_websites().values():
        site = build_generic_site(spec)
        edge_rows = [
            {"source": source, "target": target, "site_id": spec.site_id, "weight": 1}
            for source, target in site.graph.edges
        ]
        node_rows = [
            {
                "path": node,
                "site_id": spec.site_id,
                "category": site.graph.nodes[node].get("category", ""),
                "depth": site.graph.nodes[node].get("depth", -1),
                "out_degree": site.graph.out_degree(node),
                "in_degree": site.graph.in_degree(node),
                "pagerank": site.graph.nodes[node].get("pagerank", 0.0),
                "betweenness": site.graph.nodes[node].get("betweenness", 0.0),
            }
            for node in site.graph.nodes
        ]
        import pandas as pd

        site_dir = root / spec.site_id
        site_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(edge_rows).to_csv(site_dir / "graph_edges.csv", index=False)
        pd.DataFrame(node_rows).to_csv(site_dir / "graph_nodes.csv", index=False)


def root_path(site_id: str) -> str:
    return f"/{site_id}/"


def _atlas_shop() -> WebsiteSpec:
    sid = "atlas_shop"
    root = root_path(sid)
    pages = [
        _p(root, "Atlas Shop", "home", ["/catalog", "/deals", "/about", "/search", "/cart"], "A broad commerce hub with many sibling product branches."),
        _p("/catalog", "Catalog", "listing", ["/catalog/water", "/catalog/sensors", "/catalog/kits", "/catalog/field", "/search", root], "A high-degree catalog page. Crawlers love this shape."),
        _p("/catalog/water", "Water Systems", "listing", ["/p/aqua", "/p/river", "/p/filter", "/catalog", "/cart"], "Water system category."),
        _p("/catalog/sensors", "Sensors", "listing", ["/p/flow", "/p/node", "/p/camera", "/catalog", "/cart"], "Sensor category."),
        _p("/catalog/kits", "Kits", "listing", ["/p/reservoir", "/p/starter", "/p/pro", "/catalog", "/deals"], "Kit category."),
        _p("/catalog/field", "Field Gear", "listing", ["/p/boots", "/p/pack", "/p/meter", "/catalog"], "Field category."),
        _p("/deals", "Deals", "listing", ["/p/starter", "/p/filter", "/catalog", "/cart"], "Deal hub."),
        _p("/search", "Search", "utility", ["/catalog/water", "/catalog/sensors", "/p/aqua", "/p/flow", root], "Search results."),
        _p("/about", "About", "info", [root, "/catalog", "/contact"], "Company page."),
        _p("/contact", "Contact", "info", [root, "/about"], "Contact page."),
        _p("/cart", "Cart", "cart", ["/catalog", "/deals", root], "Cart page."),
    ]
    for slug, title in [
        ("aqua", "Aqua Core"), ("river", "River Vision"), ("filter", "Pure Filter"),
        ("flow", "Flow Guard"), ("node", "Hydra Node"), ("camera", "Canal Cam"),
        ("reservoir", "Reservoir Kit"), ("starter", "Starter Pack"), ("pro", "Pro Bundle"),
        ("boots", "Field Boots"), ("pack", "Survey Pack"), ("meter", "Water Meter"),
    ]:
        pages.append(_p(f"/p/{slug}", title, "detail", ["/catalog", "/cart", "/search"], f"Product detail page for {title}."))
    return WebsiteSpec(sid, "Atlas Shop", "commerce", 8061, "broad hub-and-spoke catalog", "#0f766e", tuple(pages))


def _deep_docs() -> WebsiteSpec:
    sid = "deep_docs"
    root = root_path(sid)
    pages = [
        _p(root, "Deep Docs", "home", ["/docs", "/api", "/tutorials", "/search"], "A deep documentation tree."),
        _p("/docs", "Docs", "listing", ["/docs/start", "/docs/install", "/docs/advanced", root], "Documentation hub."),
        _p("/docs/start", "Start", "docs", ["/docs/start/overview", "/docs/install", "/docs"], "Start guide."),
        _p("/docs/start/overview", "Overview", "docs", ["/docs/start/overview/concepts", "/docs/start", "/api"], "Overview."),
        _p("/docs/start/overview/concepts", "Concepts", "docs", ["/docs/start/overview/concepts/graph", "/docs/start/overview"], "Concepts."),
        _p("/docs/start/overview/concepts/graph", "Graph Concepts", "docs", ["/docs/start/overview/concepts/graph/examples", "/docs/start/overview/concepts"], "Deep concept page."),
        _p("/docs/start/overview/concepts/graph/examples", "Examples", "docs", ["/docs/advanced/tuning", "/api/reference", "/docs"], "Example page."),
        _p("/docs/install", "Install", "docs", ["/docs/install/windows", "/docs/install/linux", "/docs/start"], "Install guide."),
        _p("/docs/install/windows", "Windows", "docs", ["/docs/install/windows/troubleshooting", "/docs/install"], "Windows install."),
        _p("/docs/install/windows/troubleshooting", "Windows Troubleshooting", "docs", ["/docs/install", "/search"], "Troubleshooting."),
        _p("/docs/install/linux", "Linux", "docs", ["/docs/install/linux/containers", "/docs/install"], "Linux install."),
        _p("/docs/install/linux/containers", "Containers", "docs", ["/docs/advanced", "/docs/install/linux"], "Container docs."),
        _p("/docs/advanced", "Advanced", "docs", ["/docs/advanced/tuning", "/docs/advanced/security", "/docs"], "Advanced hub."),
        _p("/docs/advanced/tuning", "Tuning", "docs", ["/docs/advanced/tuning/thresholds", "/docs/advanced"], "Tuning."),
        _p("/docs/advanced/tuning/thresholds", "Thresholds", "docs", ["/docs/advanced/security", "/api/reference"], "Thresholds."),
        _p("/docs/advanced/security", "Security", "docs", ["/docs/advanced/security/audit", "/docs/advanced"], "Security."),
        _p("/docs/advanced/security/audit", "Audit", "docs", ["/docs", "/api/reference"], "Audit."),
        _p("/api", "API", "listing", ["/api/reference", "/api/examples", root], "API hub."),
        _p("/api/reference", "Reference", "docs", ["/api/reference/events", "/api/reference/models", "/api"], "Reference."),
        _p("/api/reference/events", "Events", "docs", ["/api/reference", "/docs/advanced/tuning"], "Events."),
        _p("/api/reference/models", "Models", "docs", ["/api/reference", "/docs/advanced/security"], "Models."),
        _p("/api/examples", "API Examples", "docs", ["/api/reference", "/tutorials"], "Examples."),
        _p("/tutorials", "Tutorials", "listing", ["/tutorials/first", "/tutorials/hardening", root], "Tutorial hub."),
        _p("/tutorials/first", "First Tutorial", "docs", ["/docs/start", "/tutorials"], "First tutorial."),
        _p("/tutorials/hardening", "Hardening Tutorial", "docs", ["/docs/advanced/security", "/tutorials"], "Hardening."),
        _p("/search", "Search", "utility", ["/docs", "/api/reference", "/tutorials"], "Search."),
    ]
    return WebsiteSpec(sid, "Deep Docs", "docs", 8062, "deep tree with long chains", "#1d4ed8", tuple(pages))


def _news_mesh() -> WebsiteSpec:
    sid = "news_mesh"
    root = root_path(sid)
    topics = ["water", "policy", "security", "field", "research", "markets"]
    pages = [
        _p(root, "News Mesh", "home", [f"/topic/{topic}" for topic in topics] + ["/trending", "/about"], "Dense editorial mesh."),
        _p("/trending", "Trending", "listing", ["/story/water-1", "/story/security-2", "/story/research-3", root], "Trending stories."),
        _p("/about", "About", "info", [root, "/trending"], "About this newsroom."),
    ]
    for topic in topics:
        pages.append(_p(f"/topic/{topic}", topic.title(), "listing", [f"/story/{topic}-{i}" for i in range(1, 4)] + ["/trending", root], f"{topic.title()} topic page."))
    story_slugs = [f"{topic}-{i}" for topic in topics for i in range(1, 4)]
    for index, slug in enumerate(story_slugs):
        related = [f"/story/{story_slugs[(index + step) % len(story_slugs)]}" for step in (3, 7, 11)]
        topic = slug.rsplit("-", 1)[0]
        pages.append(_p(f"/story/{slug}", f"Story {slug}", "article", [f"/topic/{topic}", "/trending", *related], f"Article {slug} with cross-topic related links."))
    return WebsiteSpec(sid, "News Mesh", "news", 8063, "dense article mesh", "#be123c", tuple(pages))


def _support_funnel() -> WebsiteSpec:
    sid = "support_funnel"
    root = root_path(sid)
    pages = [
        _p(root, "Support Funnel", "home", ["/help", "/status", "/login", "/contact"], "A narrow support funnel."),
        _p("/help", "Help Center", "listing", ["/help/account", "/help/billing", "/help/devices", "/search", root], "Help entry."),
        _p("/help/account", "Account Help", "listing", ["/help/account/reset", "/help/account/security", "/help", "/contact"], "Account."),
        _p("/help/account/reset", "Reset Password", "article", ["/help/account/reset/step-1", "/help/account", "/contact"], "Reset intro."),
        _p("/help/account/reset/step-1", "Reset Step 1", "article", ["/help/account/reset/step-2", "/help/account/reset"], "Step one."),
        _p("/help/account/reset/step-2", "Reset Step 2", "article", ["/help/account/reset/step-3", "/contact"], "Step two."),
        _p("/help/account/reset/step-3", "Reset Done", "article", ["/contact", "/help"], "Done."),
        _p("/help/account/security", "Security", "article", ["/help/account/security/mfa", "/help/account"], "Security."),
        _p("/help/account/security/mfa", "MFA", "article", ["/contact", "/help/account/security"], "MFA."),
        _p("/help/billing", "Billing Help", "listing", ["/help/billing/invoice", "/help/billing/refund", "/contact", "/help"], "Billing."),
        _p("/help/billing/invoice", "Invoice", "article", ["/contact", "/help/billing"], "Invoice."),
        _p("/help/billing/refund", "Refund", "article", ["/contact", "/help/billing"], "Refund."),
        _p("/help/devices", "Devices", "listing", ["/help/devices/connect", "/help/devices/repair", "/help"], "Devices."),
        _p("/help/devices/connect", "Connect Device", "article", ["/help/devices/connect/wifi", "/contact"], "Connect."),
        _p("/help/devices/connect/wifi", "Wi-Fi", "article", ["/contact", "/help/devices"], "Wi-Fi."),
        _p("/help/devices/repair", "Repair", "article", ["/contact", "/status"], "Repair."),
        _p("/search", "Search", "utility", ["/help/account", "/help/billing", "/help/devices", root], "Search."),
        _p("/status", "Status", "utility", ["/contact", root], "Status."),
        _p("/login", "Login", "utility", ["/help/account/reset", root], "Login."),
        _p("/contact", "Contact", "terminal", [root, "/help"], "Contact endpoint."),
    ]
    return WebsiteSpec(sid, "Support Funnel", "support", 8064, "narrow funnel with terminal help paths", "#c2410c", tuple(pages))


def _p(path: str, title: str, category: str, links: list[str], body: str) -> PageSpec:
    return PageSpec(path=path, title=title, category=category, links=tuple(links), body=body)

