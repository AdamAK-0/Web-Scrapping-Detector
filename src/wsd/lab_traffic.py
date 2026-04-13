"""Generate local lab traffic against the included website for research experiments."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HUMAN_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/135.0.0.0 Safari/537.36",
]
BOT_UAS = {
    "bfs": "ResearchScraper-BFS/1.0 (+https://localhost/lab)",
    "dfs": "ResearchScraper-DFS/1.0 (+https://localhost/lab)",
    "linear": "ResearchScraper-Linear/1.0 (+https://localhost/lab)",
    "stealth": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 ResearchStealth/1.0"
    ),
    "products": "ResearchScraper-ProductFocus/1.0 (+https://localhost/lab)",
    "articles": "ResearchScraper-ArticleFocus/1.0 (+https://localhost/lab)",
    "revisit": "ResearchScraper-Revisit/1.0 (+https://localhost/lab)",
}


def is_html_like(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith(('.xml', '.txt', '.json', '.csv', '.css', '.js', '.svg', '.png', '.jpg', '.jpeg', '.webp', '.ico')):
        return False
    return True


def normalize_base(base_url: str) -> str:
    return base_url.rstrip("/") + "/"


def make_session_ua(base_ua: str, label: str, index: int) -> str:
    return f"{base_ua} LabSession/{label}-{index:03d}"


def internal_url(base_url: str, ref: str) -> str | None:
    if not ref or ref.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    resolved = urljoin(base_url, ref)
    parsed_base = urlparse(base_url)
    parsed = urlparse(resolved)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc != parsed_base.netloc:
        return None
    return parsed._replace(fragment="").geturl()


def fetch_page(session: requests.Session, url: str, *, referer: str | None = None, fetch_assets: bool = False, timeout: float = 10.0) -> tuple[list[str], list[str]]:
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    if referer:
        headers["Referer"] = referer
    response = session.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links: list[str] = []
    assets: list[str] = []
    for tag in soup.find_all("a", href=True):
        internal = internal_url(url, tag["href"])
        if internal and is_html_like(internal) and internal not in links:
            links.append(internal)
    for tag in soup.find_all(["img", "script", "link"]):
        attr = "src" if tag.name in {"img", "script"} else "href"
        ref = tag.get(attr)
        if not ref:
            continue
        internal = internal_url(url, ref)
        if internal and internal not in assets:
            assets.append(internal)
    if fetch_assets:
        for asset in assets[:8]:
            try:
                session.get(asset, headers={"Referer": url, "Accept": "*/*"}, timeout=timeout)
            except requests.RequestException:
                continue
    return links, assets


def sleep_jitter(enabled: bool, low: float, high: float) -> None:
    if enabled and high > 0:
        time.sleep(random.uniform(low, high))


def merge_labels(labels_path: Path, new_rows: list[tuple[str, str]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("client_key") and row.get("label"):
                    existing[row["client_key"]] = row["label"]
    for key, label in new_rows:
        existing[key] = label
    with labels_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["client_key", "label"])
        writer.writeheader()
        for key, label in sorted(existing.items()):
            writer.writerow({"client_key": key, "label": label})


def generate_human_sessions(base_url: str, sessions_count: int, *, real_sleep: bool, labels_path: Path, seed: int) -> None:
    random.seed(seed)
    base_url = normalize_base(base_url)
    labels: list[tuple[str, str]] = []
    for idx in range(1, sessions_count + 1):
        ua = make_session_ua(random.choice(HUMAN_UAS), "human", idx)
        labels.append((f"ip_ua:127.0.0.1|{ua}", "human"))
        session = requests.Session()
        session.headers.update({"User-Agent": ua})
        history: list[str] = []
        current = urljoin(base_url, "index.html")
        for _ in range(random.randint(6, 12)):
            links, _ = fetch_page(session, current, referer=history[-1] if history else None, fetch_assets=True)
            history.append(current)
            sleep_jitter(real_sleep, 0.4, 2.0)
            options = [link for link in links if "hidden/" not in link]
            if not options:
                current = urljoin(base_url, "products.html")
                continue
            if random.random() < 0.12:
                current = random.choice(
                    [
                        urljoin(base_url, "search.html"),
                        urljoin(base_url, "cart.html"),
                        urljoin(base_url, "contact.html"),
                    ]
                )
                continue
            if len(history) >= 3 and random.random() < 0.25:
                current = history[-2]
                continue
            product_links = [link for link in options if "/pages/products/" in link]
            article_links = [link for link in options if "/pages/articles/" in link]
            nav_links = [link for link in options if link.endswith(("index.html", "products.html", "articles.html", "about.html", "faq.html", "contact.html", "cart.html"))]
            weighted = (product_links * 3) + (article_links * 2) + (nav_links * 2) + options
            current = random.choice(weighted)
    merge_labels(labels_path, labels)


def collect_site_links(session: requests.Session, base_url: str) -> list[str]:
    seeds = [urljoin(base_url, path) for path in ["index.html", "products.html", "articles.html", "about.html", "faq.html", "contact.html"]]
    seen: set[str] = set()
    queue = list(seeds)
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            links, _ = fetch_page(session, url, fetch_assets=False)
        except requests.RequestException:
            continue
        for link in links:
            if link not in seen and link not in queue:
                queue.append(link)
    return sorted(seen)


def generate_bot_sessions(base_url: str, sessions_count: int, *, mode: str, real_sleep: bool, labels_path: Path, seed: int) -> None:
    random.seed(seed)
    base_url = normalize_base(base_url)
    labels: list[tuple[str, str]] = []
    for idx in range(1, sessions_count + 1):
        ua = make_session_ua(BOT_UAS[mode], f"bot-{mode}", idx)
        labels.append((f"ip_ua:127.0.0.1|{ua}", "bot"))
        session = requests.Session()
        session.headers.update({"User-Agent": ua})
        try:
            session.get(urljoin(base_url, "robots.txt"), timeout=10)
            session.get(urljoin(base_url, "sitemap.xml"), timeout=10)
        except requests.RequestException:
            pass
        site_links = collect_site_links(session, base_url)
        crawl = site_links.copy()
        if mode == "dfs":
            crawl = list(reversed(crawl))
        elif mode == "linear":
            crawl = [link for link in crawl if ("/pages/products/" in link or "/pages/articles/" in link or link.endswith(("products.html", "articles.html")))]
        elif mode == "products":
            crawl = [
                link
                for link in crawl
                if "/pages/products/" in link or link.endswith(("products.html", "cart.html"))
            ]
        elif mode == "articles":
            crawl = [
                link
                for link in crawl
                if "/pages/articles/" in link or link.endswith(("articles.html", "about.html"))
            ]
        elif mode == "revisit":
            crawl = [
                link
                for link in crawl
                if "/pages/products/" in link or "/pages/articles/" in link or link.endswith(("products.html", "articles.html", "faq.html"))
            ]
        limit = min(len(crawl), random.randint(10, 20))
        selected = crawl[:limit]
        if mode == "stealth":
            random.shuffle(selected)
        for link in selected:
            try:
                fetch_page(session, link, fetch_assets=False)
            except requests.RequestException:
                continue
            if mode == "stealth":
                sleep_jitter(real_sleep, 0.35, 1.60)
            else:
                sleep_jitter(real_sleep, 0.02, 0.20)
            if mode == "revisit" and random.random() < 0.30:
                try:
                    fetch_page(session, link, fetch_assets=False)
                except requests.RequestException:
                    continue
        if random.random() < 0.7:
            try:
                session.get(urljoin(base_url, "hidden/diagnostic-offer.html"), timeout=10)
            except requests.RequestException:
                pass
    merge_labels(labels_path, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled local traffic for the included Nginx lab website")
    parser.add_argument("--base-url", default="http://127.0.0.1:8039/", help="Base URL of the locally served website")
    parser.add_argument(
        "--mode",
        choices=["human", "bfs", "dfs", "linear", "stealth", "products", "articles", "revisit"],
        required=True,
        help="Traffic generation mode",
    )
    parser.add_argument("--sessions", type=int, default=12, help="Number of sessions to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--real-sleep", action="store_true", help="Sleep between requests to create more realistic timestamps")
    parser.add_argument("--labels-path", default="data/live_labels/manual_labels.csv", help="Where to append manual labels")
    args = parser.parse_args()

    labels_path = Path(args.labels_path)
    if args.mode == "human":
        generate_human_sessions(args.base_url, args.sessions, real_sleep=args.real_sleep, labels_path=labels_path, seed=args.seed)
    else:
        generate_bot_sessions(args.base_url, args.sessions, mode=args.mode, real_sleep=args.real_sleep, labels_path=labels_path, seed=args.seed)
    print(f"Updated labels: {labels_path.resolve()}")


if __name__ == "__main__":
    main()
