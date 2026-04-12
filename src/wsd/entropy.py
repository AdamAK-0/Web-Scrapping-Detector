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
