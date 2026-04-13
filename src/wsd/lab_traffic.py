"""Generate local lab traffic against the included website for research experiments."""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

LABEL_COLUMNS = [
    "client_key",
    "session_id",
    "label",
    "participant_id",
    "traffic_family",
    "collection_method",
    "automation_stack",
    "notes",
]

HUMAN_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/135.0.0.0 Safari/537.36",
]

BOT_PROFILES = {
    "bfs": {
        "user_agent": "ResearchScraper-BFS/2.0 (+https://localhost/lab)",
        "traffic_family": "bfs",
        "collection_method": "scripted_requests",
        "automation_stack": "requests",
    },
    "dfs": {
        "user_agent": "ResearchScraper-DFS/2.0 (+https://localhost/lab)",
        "traffic_family": "dfs",
        "collection_method": "scripted_requests",
        "automation_stack": "requests",
    },
    "linear": {
        "user_agent": "ResearchScraper-Linear/2.0 (+https://localhost/lab)",
        "traffic_family": "linear",
        "collection_method": "scripted_requests",
        "automation_stack": "requests",
    },
    "stealth": {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 ResearchStealth/2.0"
        ),
        "traffic_family": "stealth_revisit",
        "collection_method": "scripted_requests",
        "automation_stack": "requests",
    },
    "products": {
        "user_agent": "ResearchScraper-ProductFocus/2.0 (+https://localhost/lab)",
        "traffic_family": "product_focus",
        "collection_method": "scripted_requests",
        "automation_stack": "requests",
    },
    "articles": {
        "user_agent": "ResearchScraper-ArticleFocus/2.0 (+https://localhost/lab)",
        "traffic_family": "article_focus",
        "collection_method": "scripted_requests",
        "automation_stack": "requests",
    },
    "revisit": {
        "user_agent": "ResearchScraper-Revisit/2.0 (+https://localhost/lab)",
        "traffic_family": "stealth_revisit",
        "collection_method": "scripted_requests",
        "automation_stack": "requests",
    },
    "browser_hybrid": {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 HybridCrawler/1.0"
        ),
        "traffic_family": "browser_hybrid",
        "collection_method": "browser_like_requests",
        "automation_stack": "requests_browser_emulation",
    },
    "browser_noise": {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 BrowserNoise/1.0"
        ),
        "traffic_family": "browser_noise",
        "collection_method": "browser_like_requests",
        "automation_stack": "requests_browser_emulation",
    },
    "playwright": {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 PlaywrightBrowser/1.0"
        ),
        "traffic_family": "playwright_browser",
        "collection_method": "browser_automation",
        "automation_stack": "playwright",
    },
    "selenium": {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 SeleniumBrowser/1.0"
        ),
        "traffic_family": "selenium_browser",
        "collection_method": "browser_automation",
        "automation_stack": "selenium",
    },
}


def is_html_like(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith((".xml", ".txt", ".json", ".csv", ".css", ".js", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico")):
        return False
    return True


def normalize_base(base_url: str) -> str:
    return base_url.rstrip("/") + "/"


def make_session_ua(base_ua: str, label: str, index: int) -> str:
    return f"{base_ua} LabSession/{label}-{index:03d}"


def build_label_row(
    client_key: str,
    *,
    label: str,
    participant_id: str = "",
    traffic_family: str = "",
    collection_method: str = "",
    automation_stack: str = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "client_key": client_key,
        "session_id": "",
        "label": label,
        "participant_id": participant_id,
        "traffic_family": traffic_family,
        "collection_method": collection_method,
        "automation_stack": automation_stack,
        "notes": notes,
    }


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


def fetch_page(
    session: requests.Session,
    url: str,
    *,
    referer: str | None = None,
    fetch_assets: bool = False,
    timeout: float = 10.0,
) -> tuple[list[str], list[str]]:
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }
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
        for asset in assets[:10]:
            try:
                session.get(asset, headers={"Referer": url, "Accept": "*/*"}, timeout=timeout)
            except requests.RequestException:
                continue
    return links, assets


def sleep_jitter(enabled: bool, low: float, high: float) -> None:
    if enabled and high > 0:
        time.sleep(random.uniform(low, high))


def merge_labels(labels_path: Path, new_rows: list[dict[str, str]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, str]] = {}
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = str(row.get("client_key", "")).strip() or str(row.get("session_id", "")).strip()
                if not key:
                    continue
                existing[key] = {column: str(row.get(column, "") or "") for column in LABEL_COLUMNS}
    for row in new_rows:
        key = str(row.get("client_key", "")).strip() or str(row.get("session_id", "")).strip()
        if not key:
            continue
        merged = existing.get(key, {column: "" for column in LABEL_COLUMNS})
        for column in LABEL_COLUMNS:
            value = str(row.get(column, "") or "")
            if value:
                merged[column] = value
        existing[key] = merged
    with labels_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for _, row in sorted(existing.items()):
            writer.writerow(row)


def collect_site_links(session: requests.Session, base_url: str) -> list[str]:
    seeds = [
        urljoin(base_url, path)
        for path in ["index.html", "products.html", "articles.html", "about.html", "faq.html", "contact.html", "cart.html"]
    ]
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


def plan_request_crawl(site_links: list[str], *, mode: str) -> list[str]:
    crawl = site_links.copy()
    if mode == "dfs":
        crawl = list(reversed(crawl))
    elif mode == "linear":
        crawl = [link for link in crawl if _matches_any(link, ("/pages/products/", "/pages/articles/", "products.html", "articles.html"))]
    elif mode == "products":
        crawl = [link for link in crawl if _matches_any(link, ("/pages/products/", "products.html", "cart.html"))]
    elif mode == "articles":
        crawl = [link for link in crawl if _matches_any(link, ("/pages/articles/", "articles.html", "about.html"))]
    elif mode in {"revisit", "stealth"}:
        crawl = [link for link in crawl if _matches_any(link, ("/pages/products/", "/pages/articles/", "products.html", "articles.html", "faq.html"))]
    return crawl


def build_browser_like_plan(site_links: list[str], *, mode: str, rng: random.Random, limit: int) -> list[str]:
    product_links = [link for link in site_links if "/pages/products/" in link]
    article_links = [link for link in site_links if "/pages/articles/" in link]
    nav_links = [link for link in site_links if link.endswith(("index.html", "products.html", "articles.html", "about.html", "faq.html", "contact.html", "cart.html", "search.html"))]
    plan: list[str] = []
    options = {
        "product": product_links or nav_links or site_links,
        "article": article_links or nav_links or site_links,
        "nav": nav_links or site_links,
        "mixed": site_links,
    }
    state = "nav"
    for _ in range(limit):
        if mode == "browser_hybrid":
            state = rng.choices(["product", "article", "nav", "mixed"], weights=[4, 3, 2, 1], k=1)[0]
        elif mode == "browser_noise":
            state = rng.choices(["product", "article", "nav", "mixed"], weights=[3, 3, 2, 2], k=1)[0]
        elif mode == "playwright":
            state = rng.choices(["product", "article", "nav"], weights=[4, 2, 2], k=1)[0]
        elif mode == "selenium":
            state = rng.choices(["product", "article", "nav"], weights=[3, 3, 2], k=1)[0]
        candidates = options[state]
        if not candidates:
            break
        choice = rng.choice(candidates)
        plan.append(choice)
        if mode in {"browser_noise", "playwright", "selenium"} and len(plan) >= 2 and rng.random() < 0.22:
            plan.append(plan[-2])
    return plan[:limit]


def generate_human_sessions(
    base_url: str,
    sessions_count: int,
    *,
    real_sleep: bool,
    labels_path: Path,
    seed: int,
    participant_id: str = "",
    collection_method: str = "scripted_human",
    notes: str = "",
) -> None:
    rng = random.Random(seed)
    base_url = normalize_base(base_url)
    labels: list[dict[str, str]] = []
    for idx in range(1, sessions_count + 1):
        ua = make_session_ua(rng.choice(HUMAN_UAS), "human", idx)
        client_key = f"ip_ua:127.0.0.1|{ua}"
        labels.append(
            build_label_row(
                client_key,
                label="human",
                participant_id=participant_id or f"generated_human_{idx:03d}",
                traffic_family="human_navigation",
                collection_method=collection_method,
                automation_stack="requests",
                notes=notes,
            )
        )
        session = requests.Session()
        session.headers.update({"User-Agent": ua})
        history: list[str] = []
        current = urljoin(base_url, "index.html")
        for _ in range(rng.randint(6, 12)):
            links, _ = fetch_page(session, current, referer=history[-1] if history else None, fetch_assets=True)
            history.append(current)
            sleep_jitter(real_sleep, 0.4, 2.1)
            options = [link for link in links if "hidden/" not in link]
            if not options:
                current = urljoin(base_url, "products.html")
                continue
            if rng.random() < 0.14:
                current = rng.choice(
                    [
                        urljoin(base_url, "search.html"),
                        urljoin(base_url, "cart.html"),
                        urljoin(base_url, "contact.html"),
                        urljoin(base_url, "faq.html"),
                    ]
                )
                continue
            if len(history) >= 3 and rng.random() < 0.27:
                current = history[-2]
                continue
            product_links = [link for link in options if "/pages/products/" in link]
            article_links = [link for link in options if "/pages/articles/" in link]
            nav_links = [
                link
                for link in options
                if link.endswith(("index.html", "products.html", "articles.html", "about.html", "faq.html", "contact.html", "cart.html"))
            ]
            weighted = (product_links * 3) + (article_links * 2) + (nav_links * 2) + options
            current = rng.choice(weighted)
    merge_labels(labels_path, labels)


def generate_request_bot_sessions(
    base_url: str,
    sessions_count: int,
    *,
    mode: str,
    real_sleep: bool,
    labels_path: Path,
    seed: int,
    notes: str = "",
) -> None:
    rng = random.Random(seed)
    base_url = normalize_base(base_url)
    labels: list[dict[str, str]] = []
    profile = BOT_PROFILES[mode]

    for idx in range(1, sessions_count + 1):
        ua = make_session_ua(profile["user_agent"], f"bot-{mode}", idx)
        client_key = f"ip_ua:127.0.0.1|{ua}"
        labels.append(
            build_label_row(
                client_key,
                label="bot",
                participant_id=f"{profile['traffic_family']}_{idx:03d}",
                traffic_family=profile["traffic_family"],
                collection_method=profile["collection_method"],
                automation_stack=profile["automation_stack"],
                notes=notes,
            )
        )

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": ua,
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            }
        )
        try:
            session.get(urljoin(base_url, "robots.txt"), timeout=10)
            session.get(urljoin(base_url, "sitemap.xml"), timeout=10)
        except requests.RequestException:
            pass

        site_links = collect_site_links(session, base_url)
        if mode in {"browser_hybrid", "browser_noise"}:
            selected = build_browser_like_plan(site_links, mode=mode, rng=rng, limit=rng.randint(12, 22))
        else:
            crawl = plan_request_crawl(site_links, mode=mode)
            selected = crawl[: min(len(crawl), rng.randint(10, 20))]
            if mode == "stealth":
                rng.shuffle(selected)

        previous_url: str | None = None
        for link in selected:
            try:
                fetch_assets = mode in {"browser_hybrid", "browser_noise"}
                fetch_page(session, link, referer=previous_url, fetch_assets=fetch_assets)
            except requests.RequestException:
                continue
            previous_url = link

            if mode == "browser_noise" and rng.random() < 0.25 and selected:
                extra = rng.choice(selected)
                try:
                    fetch_page(session, extra, referer=previous_url, fetch_assets=True)
                    previous_url = extra
                except requests.RequestException:
                    pass

            if mode == "revisit" and rng.random() < 0.30:
                try:
                    fetch_page(session, link, referer=previous_url, fetch_assets=False)
                except requests.RequestException:
                    pass

            if mode in {"stealth", "browser_hybrid", "browser_noise"}:
                sleep_jitter(real_sleep, 0.30, 1.75)
            else:
                sleep_jitter(real_sleep, 0.02, 0.20)

        if rng.random() < 0.7:
            try:
                session.get(urljoin(base_url, "hidden/diagnostic-offer.html"), timeout=10)
            except requests.RequestException:
                pass
    merge_labels(labels_path, labels)


def generate_playwright_sessions(
    base_url: str,
    sessions_count: int,
    *,
    real_sleep: bool,
    labels_path: Path,
    seed: int,
    notes: str = "",
) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Playwright mode requires `pip install playwright` and `playwright install`.") from exc

    rng = random.Random(seed)
    base_url = normalize_base(base_url)
    labels: list[dict[str, str]] = []
    profile = BOT_PROFILES["playwright"]

    planning_session = requests.Session()
    planning_session.headers.update({"User-Agent": profile["user_agent"]})
    site_links = collect_site_links(planning_session, base_url)

    with sync_playwright() as playwright:  # pragma: no cover - runtime dependency
        browser = playwright.chromium.launch(headless=True)
        try:
            for idx in range(1, sessions_count + 1):
                ua = make_session_ua(profile["user_agent"], "bot-playwright", idx)
                client_key = f"ip_ua:127.0.0.1|{ua}"
                labels.append(
                    build_label_row(
                        client_key,
                        label="bot",
                        participant_id=f"playwright_browser_{idx:03d}",
                        traffic_family=profile["traffic_family"],
                        collection_method=profile["collection_method"],
                        automation_stack=profile["automation_stack"],
                        notes=notes,
                    )
                )
                context = browser.new_context(user_agent=ua, viewport={"width": 1366, "height": 900})
                page = context.new_page()
                plan = build_browser_like_plan(site_links, mode="playwright", rng=rng, limit=rng.randint(10, 18))
                for url in plan:
                    page.goto(url, wait_until="networkidle", timeout=15000)
                    page.mouse.wheel(0, rng.randint(150, 800))
                    if real_sleep:
                        page.wait_for_timeout(rng.randint(300, 1600))
                context.close()
        finally:
            browser.close()

    merge_labels(labels_path, labels)


def generate_selenium_sessions(
    base_url: str,
    sessions_count: int,
    *,
    real_sleep: bool,
    labels_path: Path,
    seed: int,
    notes: str = "",
) -> None:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Selenium mode requires `pip install selenium`.") from exc

    rng = random.Random(seed)
    base_url = normalize_base(base_url)
    labels: list[dict[str, str]] = []
    profile = BOT_PROFILES["selenium"]

    planning_session = requests.Session()
    planning_session.headers.update({"User-Agent": profile["user_agent"]})
    site_links = collect_site_links(planning_session, base_url)

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1366,900")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    for idx in range(1, sessions_count + 1):  # pragma: no cover - runtime dependency
        ua = make_session_ua(profile["user_agent"], "bot-selenium", idx)
        client_key = f"ip_ua:127.0.0.1|{ua}"
        labels.append(
            build_label_row(
                client_key,
                label="bot",
                participant_id=f"selenium_browser_{idx:03d}",
                traffic_family=profile["traffic_family"],
                collection_method=profile["collection_method"],
                automation_stack=profile["automation_stack"],
                notes=notes,
            )
        )
        options_with_ua = Options()
        for arg in options.arguments:
            options_with_ua.add_argument(arg)
        options_with_ua.add_argument(f"--user-agent={ua}")
        driver = webdriver.Chrome(options=options_with_ua)
        try:
            plan = build_browser_like_plan(site_links, mode="selenium", rng=rng, limit=rng.randint(10, 18))
            for url in plan:
                driver.get(url)
                driver.execute_script("window.scrollTo(0, arguments[0]);", rng.randint(150, 900))
                if real_sleep:
                    time.sleep(rng.uniform(0.3, 1.6))
        finally:
            driver.quit()

    merge_labels(labels_path, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled local traffic for the included Nginx lab website")
    parser.add_argument("--base-url", default="http://127.0.0.1:8039/", help="Base URL of the locally served website")
    parser.add_argument(
        "--mode",
        choices=[
            "human",
            "bfs",
            "dfs",
            "linear",
            "stealth",
            "products",
            "articles",
            "revisit",
            "playwright",
            "selenium",
            "browser_hybrid",
            "browser_noise",
        ],
        required=True,
        help="Traffic generation mode",
    )
    parser.add_argument("--sessions", type=int, default=12, help="Number of sessions to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--real-sleep", action="store_true", help="Sleep between requests to create more realistic timestamps")
    parser.add_argument("--labels-path", default="data/live_labels/manual_labels.csv", help="Where to append manual labels")
    parser.add_argument("--participant-id", default="", help="Optional participant identifier for human collection runs")
    parser.add_argument("--collection-method", default="", help="Optional override for collection method metadata")
    parser.add_argument("--notes", default="", help="Optional notes stored alongside manual labels")
    args = parser.parse_args()

    labels_path = Path(args.labels_path)
    if args.mode == "human":
        generate_human_sessions(
            args.base_url,
            args.sessions,
            real_sleep=args.real_sleep,
            labels_path=labels_path,
            seed=args.seed,
            participant_id=args.participant_id,
            collection_method=args.collection_method or "scripted_human",
            notes=args.notes,
        )
    elif args.mode == "playwright":
        generate_playwright_sessions(
            args.base_url,
            args.sessions,
            real_sleep=args.real_sleep,
            labels_path=labels_path,
            seed=args.seed,
            notes=args.notes,
        )
    elif args.mode == "selenium":
        generate_selenium_sessions(
            args.base_url,
            args.sessions,
            real_sleep=args.real_sleep,
            labels_path=labels_path,
            seed=args.seed,
            notes=args.notes,
        )
    else:
        generate_request_bot_sessions(
            args.base_url,
            args.sessions,
            mode=args.mode,
            real_sleep=args.real_sleep,
            labels_path=labels_path,
            seed=args.seed,
            notes=args.notes,
        )
    print(f"Updated labels: {labels_path.resolve()}")


def _matches_any(url: str, needles: Iterable[str]) -> bool:
    lowered = url.lower()
    return any(needle.lower() in lowered for needle in needles)


if __name__ == "__main__":
    main()
