from pathlib import Path

from wsd.lab_traffic import build_browser_like_plan, build_label_row, merge_labels, plan_request_crawl


def test_plan_request_crawl_filters_by_mode() -> None:
    site_links = [
        "http://127.0.0.1:8039/index.html",
        "http://127.0.0.1:8039/products.html",
        "http://127.0.0.1:8039/pages/products/item-1.html",
        "http://127.0.0.1:8039/articles.html",
        "http://127.0.0.1:8039/pages/articles/post-1.html",
    ]
    product_plan = plan_request_crawl(site_links, mode="products")
    assert all(("products" in link or "cart.html" in link) for link in product_plan)


def test_build_browser_like_plan_returns_requested_limit() -> None:
    site_links = [f"http://127.0.0.1:8039/pages/products/item-{idx}.html" for idx in range(1, 8)]
    plan = build_browser_like_plan(site_links, mode="browser_noise", rng=__import__("random").Random(7), limit=6)
    assert 1 <= len(plan) <= 6


def test_merge_labels_preserves_optional_metadata(tmp_path: Path) -> None:
    labels_path = tmp_path / "manual_labels.csv"
    merge_labels(
        labels_path,
        [
            build_label_row(
                "ip_ua:127.0.0.1|ua-1",
                label="bot",
                participant_id="p1",
                traffic_family="playwright_browser",
                collection_method="browser_automation",
                automation_stack="playwright",
                notes="smoke",
            )
        ],
    )
    text = labels_path.read_text(encoding="utf-8")
    assert "traffic_family" in text
    assert "playwright_browser" in text
