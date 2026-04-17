"""Controlled bot traffic for the generic multi-website admin demo."""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
import requests

from generic_models.site_catalog import PageSpec, WebsiteSpec, get_websites, root_path


BOT_MODES = [
    "bfs",
    "dfs",
    "linear",
    "focused",
    "browser_like",
    "noisy",
    "deep_harvest",
    "random_walk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled generic-site bot traffic")
    parser.add_argument("--site-id", required=True, choices=sorted(get_websites()))
    parser.add_argument("--mode", default="browser_like", choices=BOT_MODES)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--real-sleep", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    spec = get_websites()[args.site_id]
    base_url = (args.base_url or f"http://127.0.0.1:{spec.port}").rstrip("/")
    for session_index in range(max(1, args.sessions)):
        user_agent = f"GenericWSDTestBot/{args.mode} session-{int(time.time())}-{session_index}"
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"})
        plan = build_plan(spec, mode=args.mode, rng=rng, max_steps=args.max_steps)
        run_plan(session, base_url=base_url, paths=plan, mode=args.mode, rng=rng, real_sleep=args.real_sleep)
    print(f"Completed {args.sessions} {args.mode} session(s) against {spec.site_id} at {base_url}")


def build_plan(spec: WebsiteSpec, *, mode: str, rng: random.Random, max_steps: int) -> list[str]:
    if mode == "bfs":
        return _bfs_plan(spec, max_steps=max_steps)
    if mode == "dfs":
        return _dfs_plan(spec, max_steps=max_steps)
    if mode == "linear":
        pages = sorted(spec.pages, key=lambda page: (_depth(page.path), page.path))
        return _cap([root_path(spec.site_id), *[page.path for page in pages if page.path != root_path(spec.site_id)]], max_steps)
    if mode == "focused":
        return _focused_plan(spec, rng=rng, max_steps=max_steps)
    if mode == "browser_like":
        return _random_walk(spec, rng=rng, max_steps=max_steps, revisit_bias=0.18, jump_bias=0.08)
    if mode == "noisy":
        return _random_walk(spec, rng=rng, max_steps=max_steps, revisit_bias=0.34, jump_bias=0.24)
    if mode == "deep_harvest":
        pages = sorted(spec.pages, key=lambda page: (-_depth(page.path), page.category, page.path))
        return _cap([root_path(spec.site_id), *[page.path for page in pages if page.path != root_path(spec.site_id)]], max_steps)
    if mode == "random_walk":
        return _random_walk(spec, rng=rng, max_steps=max_steps, revisit_bias=0.25, jump_bias=0.16)
    raise ValueError(f"Unsupported mode: {mode}")


def run_plan(session: requests.Session, *, base_url: str, paths: list[str], mode: str, rng: random.Random, real_sleep: bool) -> None:
    previous_url = ""
    for path in paths:
        headers = {"Referer": previous_url} if previous_url else {}
        url = f"{base_url}{path}"
        try:
            response = session.get(url, headers=headers, timeout=8)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[{mode}] request failed for {url}: {exc}")
            break
        previous_url = url
        if real_sleep:
            time.sleep(_delay_for_mode(mode, rng))


def _bfs_plan(spec: WebsiteSpec, *, max_steps: int) -> list[str]:
    page_map = _page_map(spec)
    root = root_path(spec.site_id)
    queue: deque[str] = deque([root])
    seen: set[str] = set()
    result: list[str] = []
    while queue and len(result) < max_steps:
        path = queue.popleft()
        if path in seen or path not in page_map:
            continue
        seen.add(path)
        result.append(path)
        queue.extend(page_map[path].links)
    return result


def _dfs_plan(spec: WebsiteSpec, *, max_steps: int) -> list[str]:
    page_map = _page_map(spec)
    root = root_path(spec.site_id)
    stack = [root]
    seen: set[str] = set()
    result: list[str] = []
    while stack and len(result) < max_steps:
        path = stack.pop()
        if path in seen or path not in page_map:
            continue
        seen.add(path)
        result.append(path)
        stack.extend(reversed(page_map[path].links))
    return result


def _focused_plan(spec: WebsiteSpec, *, rng: random.Random, max_steps: int) -> list[str]:
    page_map = _page_map(spec)
    root = root_path(spec.site_id)
    listings = [page.path for page in spec.pages if page.category in {"listing", "docs", "article"} and page.path != root]
    rng.shuffle(listings)
    result = [root]
    for listing in listings[:4]:
        result.append(listing)
        details = [link for link in page_map[listing].links if link in page_map and page_map[link].category in {"detail", "article", "docs"}]
        rng.shuffle(details)
        result.extend(details[:3])
        if rng.random() < 0.35:
            result.append(listing)
        if len(result) >= max_steps:
            break
    return _cap(result, max_steps)


def _random_walk(spec: WebsiteSpec, *, rng: random.Random, max_steps: int, revisit_bias: float, jump_bias: float) -> list[str]:
    page_map = _page_map(spec)
    root = root_path(spec.site_id)
    current = root
    visited = [root]
    all_paths = [page.path for page in spec.pages]
    for _ in range(max_steps - 1):
        if len(visited) > 2 and rng.random() < revisit_bias:
            current = rng.choice(visited[:-1])
        elif rng.random() < jump_bias:
            current = rng.choice(all_paths)
        else:
            links = [link for link in page_map.get(current, PageSpec(current, current, "unknown", tuple(), "")).links if link in page_map]
            current = rng.choice(links) if links else root
        visited.append(current)
    return visited


def _page_map(spec: WebsiteSpec) -> dict[str, PageSpec]:
    return {page.path: page for page in spec.pages}


def _cap(paths: list[str], max_steps: int) -> list[str]:
    return paths[: max(1, max_steps)]


def _depth(path: str) -> int:
    return len([part for part in path.strip("/").split("/") if part])


def _delay_for_mode(mode: str, rng: random.Random) -> float:
    if mode in {"browser_like", "random_walk"}:
        return rng.uniform(0.28, 1.15)
    if mode == "noisy":
        return rng.choice([rng.uniform(0.08, 0.35), rng.uniform(0.8, 1.8)])
    if mode == "focused":
        return rng.uniform(0.12, 0.65)
    return rng.uniform(0.04, 0.28)


if __name__ == "__main__":
    main()
