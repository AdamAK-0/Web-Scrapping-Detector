"""Entropy utilities used by the navigation feature pipeline."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Hashable


def shannon_entropy_from_counts(counts: Iterable[int]) -> float:
    """Compute Shannon entropy in bits from positive counts."""
    counts = [c for c in counts if c > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def normalized_entropy_from_counts(counts: Iterable[int]) -> float:
    """Normalize entropy to [0, 1] when at least two buckets exist."""
    counts = [c for c in counts if c > 0]
    if len(counts) <= 1:
        return 0.0
    entropy = shannon_entropy_from_counts(counts)
    return entropy / math.log2(len(counts))


def shannon_entropy(items: Iterable[Hashable]) -> float:
    counter = Counter(items)
    return shannon_entropy_from_counts(counter.values())


def normalized_entropy(items: Iterable[Hashable]) -> float:
    counter = Counter(items)
    return normalized_entropy_from_counts(counter.values())


def normalized_entropy_with_support(counts: Iterable[int], support_size: int) -> float:
    """Normalize entropy against a known support size.

    This is useful when a session uses only a subset of the available branches
    from a node but we still want to normalize against all possible options.
    """
    counts = [c for c in counts if c > 0]
    if support_size <= 1 or not counts:
        return 0.0
    entropy = shannon_entropy_from_counts(counts)
    return entropy / math.log2(support_size)


def concentration(items: Iterable[Hashable]) -> float:
    """Return a simple concentration score in [0, 1].

    Values near 1 indicate visits are concentrated on a few repeated nodes or
    transitions; values near 0 indicate more even coverage.
    """
    counter = Counter(items)
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return sum((count / total) ** 2 for count in counter.values())
