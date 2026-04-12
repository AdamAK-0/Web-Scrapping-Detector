"""Stateful online scoring for live request streams."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from .config import DEFAULT_UNKNOWN_LABEL
from .features import extract_features_for_events
from .modeling import ModelBundle
from .types import RequestEvent


@dataclass(slots=True)
class SessionState:
    session_id: str
    events: list[RequestEvent] = field(default_factory=list)


class OnlineDetector:
    def __init__(self, bundle: ModelBundle, graph: nx.DiGraph) -> None:
        self.bundle = bundle
        self.graph = graph
        self._sessions: dict[str, SessionState] = {}

    def observe(
        self,
        *,
        session_id: str,
        path: str,
        timestamp: float | None = None,
        page_category: str | None = None,
        referrer: str | None = None,
        user_agent: str | None = None,
        status_code: int | None = None,
    ) -> dict[str, object]:
        timestamp = float(timestamp if timestamp is not None else time.time())
        state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
        last_event = state.events[-1] if state.events else None
        delta_t = 0.0 if last_event is None else max(0.0, timestamp - last_event.timestamp)
        event = RequestEvent(
            session_id=session_id,
            timestamp=timestamp,
            path=path,
            delta_t=delta_t,
            label=DEFAULT_UNKNOWN_LABEL,
            page_category=page_category,
            referrer=referrer or (last_event.path if last_event else None),
            user_agent=user_agent,
            status_code=status_code,
        )
        state.events.append(event)

        if len(state.events) < 2:
            return {
                "session_id": session_id,
                "prefix_len": len(state.events),
                "ready": False,
                "bot_probability": None,
                "predicted_label": None,
            }

        features = extract_features_for_events(state.events, self.graph)
        feature_frame = pd.DataFrame([{name: features.get(name, 0.0) for name in self.bundle.feature_columns}])
        probability = float(self.bundle.model.predict_proba(feature_frame)[0, 1])
        predicted_label = self.bundle.positive_label if probability >= self.bundle.threshold else self.bundle.negative_label
        return {
            "session_id": session_id,
            "prefix_len": len(state.events),
            "ready": True,
            "bot_probability": probability,
            "predicted_label": predicted_label,
            "features": features,
        }

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def active_sessions(self) -> list[str]:
        return sorted(self._sessions)
