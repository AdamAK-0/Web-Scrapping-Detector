"""Shared configuration defaults for the prototype."""

from __future__ import annotations

DEFAULT_SESSION_TIMEOUT_SECONDS = 30 * 60
DEFAULT_RANDOM_SEED = 42
ROOT_NODE = "/"
DEFAULT_POSITIVE_LABEL = "bot"
DEFAULT_NEGATIVE_LABEL = "human"
DEFAULT_UNKNOWN_LABEL = "unknown"
DEFAULT_PREFIXES = [3, 5, 10, 15, 20]
ALLOWED_LABELS = {DEFAULT_POSITIVE_LABEL, DEFAULT_NEGATIVE_LABEL, DEFAULT_UNKNOWN_LABEL}

DEFAULT_PAGE_LIKE_EXTENSIONS_TO_EXCLUDE = {
    ".css",
    ".js",
    ".mjs",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".webp",
    ".woff",
    ".woff2",
    ".ttf",
    ".map",
    ".json",
    ".xml",
    ".txt",
    ".pdf",
    ".zip",
    ".gz",
}

DEFAULT_BOT_USER_AGENT_PATTERNS = [
    r"python-requests",
    r"researchscraper",
    r"researchstealth",
    r"stealth-browser",
    r"hybridcrawler",
    r"browsernoise",
    r"scrapy",
    r"selenium",
    r"webdriver",
    r"playwright",
    r"puppeteer",
    r"aiohttp",
    r"httpclient",
    r"httpx",
    r"curl/",
    r"wget/",
    r"urllib",
    r"headless",
    r"phantomjs",
    r"go-http-client",
]
