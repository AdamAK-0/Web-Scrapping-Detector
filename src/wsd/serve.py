"""FastAPI service for live online scoring."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph_builder import build_graph_from_csv
from .modeling import load_model_bundle
from .online import OnlineDetector

app = FastAPI(title="Web Scraping Detector API", version="0.2.0")
_DETECTOR: OnlineDetector | None = None


class ScoreEvent(BaseModel):
    session_id: str = Field(..., description="Application session identifier or derived client/session key")
    path: str = Field(..., description="Requested page path")
    timestamp: float | None = Field(default=None, description="Unix timestamp in seconds")
    page_category: str | None = None
    referrer: str | None = None
    user_agent: str | None = None
    status_code: int | None = None


@app.get("/health")
def health() -> dict[str, object]:
    detector = _require_detector()
    return {"ok": True, "model_name": detector.bundle.model_name, "active_sessions": len(detector.active_sessions())}


@app.post("/score")
def score(event: ScoreEvent) -> dict[str, object]:
    detector = _require_detector()
    return detector.observe(**event.model_dump())


@app.post("/reset/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    detector = _require_detector()
    detector.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


def _require_detector() -> OnlineDetector:
    if _DETECTOR is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    return _DETECTOR


def create_detector(bundle_path: str | Path, graph_dir: str | Path) -> OnlineDetector:
    bundle = load_model_bundle(bundle_path)
    graph = build_graph_from_csv(Path(graph_dir) / "graph_edges.csv", Path(graph_dir) / "graph_categories.csv")
    return OnlineDetector(bundle=bundle, graph=graph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the online web scraping detector via FastAPI")
    parser.add_argument("--bundle", default=os.getenv("WSD_MODEL_BUNDLE"), required=os.getenv("WSD_MODEL_BUNDLE") is None, help="Path to a saved .pkl model bundle")
    parser.add_argument("--graph-dir", default=os.getenv("WSD_GRAPH_DIR"), required=os.getenv("WSD_GRAPH_DIR") is None, help="Directory containing graph_edges.csv and graph_categories.csv")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    global _DETECTOR
    args = parse_args()
    _DETECTOR = create_detector(args.bundle, args.graph_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
