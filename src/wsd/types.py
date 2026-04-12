"""Typed data structures for navigation sessions and features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RequestEvent:
    session_id: str
    timestamp: float
    path: str
    delta_t: float
    label: str
    page_category: str | None = None
    referrer: str | None = None
    user_agent: str | None = None
    status_code: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Session:
    session_id: str
    label: str
    events: list[RequestEvent]

    def paths(self) -> list[str]:
        return [event.path for event in self.events]

    def deltas(self) -> list[float]:
        return [event.delta_t for event in self.events]


@dataclass(slots=True)
class PrefixFeatureRow:
    session_id: str
    prefix_len: int
    label: str
    features: dict[str, float]
