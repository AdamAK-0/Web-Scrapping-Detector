"""Standalone admin dashboard for live model selection and bot-testing."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import DEFAULT_SESSION_TIMEOUT_SECONDS
from .features import extract_features_for_events
from .graph_builder import build_graph_from_csv
from .log_parsers import read_raw_logs
from .modeling import ModelBundle, load_model_bundle
from .sessionizer import build_sessions_from_dataframe
from .types import Session


BOT_MODES = [
    "bfs",
    "dfs",
    "linear",
    "stealth",
    "products",
    "articles",
    "revisit",
    "browser_hybrid",
    "browser_noise",
    "playwright",
    "selenium",
]


class SelectModelPayload(BaseModel):
    model_name: str
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class RunBotPayload(BaseModel):
    mode: str
    sessions: int = Field(default=3, ge=1, le=50)
    real_sleep: bool = True


@dataclass(slots=True)
class BotRun:
    run_id: str
    mode: str
    sessions: int
    log_path: Path
    process: subprocess.Popen[Any]
    started_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class AdminState:
    model_dir: Path
    graph_dir: Path
    log_path: Path
    labels_path: Path
    base_url: str
    threshold: float = 0.5
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS
    bot_run_dir: Path = Path("data/admin_bot_runs")
    active_model_name: str | None = None
    active_bundle: ModelBundle | None = None
    bot_runs: list[BotRun] = field(default_factory=list)
    ignore_log_before_timestamp: float = 0.0

    def load_model(self, model_name: str, threshold: float | None = None) -> dict[str, Any]:
        models = discover_model_bundles(self.model_dir)
        match = next((item for item in models if item["model_name"] == model_name or item["file_name"] == model_name), None)
        if match is None:
            raise FileNotFoundError(f"Model bundle not found: {model_name}")
        bundle = load_model_bundle(match["path"])
        if threshold is not None:
            bundle.threshold = float(threshold)
        elif self.threshold is not None:
            bundle.threshold = float(self.threshold)
        self.active_model_name = str(match["model_name"])
        self.active_bundle = bundle
        self.threshold = float(bundle.threshold)
        return bundle_to_public_dict(match, bundle, active=True)


def discover_model_bundles(model_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(model_dir)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_bundle.pkl")):
        model_name = path.stem.removesuffix("_bundle")
        rows.append(
            {
                "model_name": model_name,
                "file_name": path.name,
                "path": str(path),
                "size_kb": round(path.stat().st_size / 1024, 1),
                "modified_at": path.stat().st_mtime,
            }
        )
    return rows


def create_app(state: AdminState) -> FastAPI:
    app = FastAPI(title="Web Scraping Detector Admin Panel", version="0.1.0")
    app.state.admin_state = state
    _load_default_model_if_available(state)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/models")
    def api_models(request: Request) -> dict[str, Any]:
        state = _state(request)
        models = discover_model_bundles(state.model_dir)
        active = state.active_model_name
        public_models = []
        for item in models:
            public = dict(item)
            public.pop("path", None)
            public["active"] = item["model_name"] == active
            public_models.append(public)
        return {
            "models": public_models,
            "active_model": active,
            "threshold": state.threshold,
            "model_dir": str(state.model_dir),
        }

    @app.post("/api/model")
    def api_select_model(payload: SelectModelPayload, request: Request) -> dict[str, Any]:
        state = _state(request)
        try:
            selected = state.load_model(payload.model_name, threshold=payload.threshold)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        selected.pop("path", None)
        return {"selected": selected, "threshold": state.threshold}

    @app.get("/api/status")
    def api_status(request: Request) -> dict[str, Any]:
        state = _state(request)
        return score_live_log(state)

    @app.post("/api/run-bot")
    def api_run_bot(payload: RunBotPayload, request: Request) -> dict[str, Any]:
        state = _state(request)
        if payload.mode not in BOT_MODES:
            raise HTTPException(status_code=400, detail=f"Unsupported bot mode: {payload.mode}")
        run = launch_bot_run(state, mode=payload.mode, sessions=payload.sessions, real_sleep=payload.real_sleep)
        return bot_run_to_public_dict(run)

    @app.post("/api/reset-live")
    def api_reset_live(request: Request) -> dict[str, Any]:
        state = _state(request)
        return reset_live_view(state)

    @app.get("/api/bot-runs")
    def api_bot_runs(request: Request) -> dict[str, Any]:
        state = _state(request)
        return {"runs": [bot_run_to_public_dict(run) for run in state.bot_runs[-12:]][::-1]}

    return app


def score_live_log(state: AdminState) -> dict[str, Any]:
    models = discover_model_bundles(state.model_dir)
    if state.active_bundle is None:
        _load_default_model_if_available(state)

    issues = []
    if not state.log_path.exists():
        issues.append(f"Live log not found: {state.log_path}")
    if not (state.graph_dir / "graph_edges.csv").exists():
        issues.append(f"Graph not found: {state.graph_dir / 'graph_edges.csv'}")
    if state.active_bundle is None:
        issues.append(f"No model bundle loaded from: {state.model_dir}")
    if issues:
        return {
            "ok": False,
            "issues": issues,
            "models": _public_model_list(models, state.active_model_name),
            "active_model": state.active_model_name,
            "threshold": state.threshold,
            "summary": _empty_summary(),
            "sessions": [],
            "bot_runs": [bot_run_to_public_dict(run) for run in state.bot_runs[-6:]][::-1],
            "updated_at": time.time(),
        }

    if state.log_path.stat().st_size == 0:
        return {
            "ok": True,
            "issues": [],
            "models": _public_model_list(models, state.active_model_name),
            "active_model": state.active_model_name,
            "threshold": state.threshold,
            "summary": summarize_scored_sessions([], threshold=state.threshold),
            "sessions": [],
            "bot_runs": [bot_run_to_public_dict(run) for run in state.bot_runs[-6:]][::-1],
            "updated_at": time.time(),
            "paths": {
                "log_path": str(state.log_path),
                "model_dir": str(state.model_dir),
                "graph_dir": str(state.graph_dir),
            },
        }

    try:
        raw_df = read_raw_logs(state.log_path, log_format="auto")
        if state.ignore_log_before_timestamp > 0 and not raw_df.empty:
            raw_df = raw_df[raw_df["timestamp"] >= state.ignore_log_before_timestamp].copy()
        if raw_df.empty:
            return {
                "ok": True,
                "issues": [],
                "models": _public_model_list(models, state.active_model_name),
                "active_model": state.active_model_name,
                "threshold": state.threshold,
                "summary": summarize_scored_sessions([], threshold=state.threshold),
                "sessions": [],
                "bot_runs": [bot_run_to_public_dict(run) for run in state.bot_runs[-6:]][::-1],
                "updated_at": time.time(),
                "paths": {
                    "log_path": str(state.log_path),
                    "model_dir": str(state.model_dir),
                    "graph_dir": str(state.graph_dir),
                },
            }
        graph = build_graph_from_csv(state.graph_dir / "graph_edges.csv", state.graph_dir / "graph_categories.csv")
        sessions = build_sessions_from_dataframe(
            raw_df,
            session_timeout_seconds=state.session_timeout_seconds,
            min_session_length=1,
        )
        session_rows = score_sessions(sessions, state.active_bundle, graph=graph)
    except Exception as exc:  # pragma: no cover - defensive for dashboard visibility
        return {
            "ok": False,
            "issues": [f"Could not score live log: {exc}"],
            "models": _public_model_list(models, state.active_model_name),
            "active_model": state.active_model_name,
            "threshold": state.threshold,
            "summary": _empty_summary(),
            "sessions": [],
            "bot_runs": [bot_run_to_public_dict(run) for run in state.bot_runs[-6:]][::-1],
            "updated_at": time.time(),
        }

    summary = summarize_scored_sessions(session_rows, threshold=state.threshold)
    return {
        "ok": True,
        "issues": [],
        "models": _public_model_list(models, state.active_model_name),
        "active_model": state.active_model_name,
        "threshold": state.threshold,
        "summary": summary,
        "sessions": session_rows,
        "bot_runs": [bot_run_to_public_dict(run) for run in state.bot_runs[-6:]][::-1],
        "updated_at": time.time(),
        "paths": {
            "log_path": str(state.log_path),
            "model_dir": str(state.model_dir),
            "graph_dir": str(state.graph_dir),
        },
    }


def score_sessions(sessions: list[Session], bundle: ModelBundle, *, graph: Any) -> list[dict[str, Any]]:
    rows = []
    for session in sessions:
        if not session.events:
            continue
        event_count = len(session.events)
        first_event = session.events[0]
        last_event = session.events[-1]
        ready = event_count >= 2
        probability: float | None = None
        predicted_label = "waiting"
        features: dict[str, float] = {}
        if ready:
            features = extract_features_for_events(session.events, graph)
            frame = pd.DataFrame([{name: features.get(name, 0.0) for name in bundle.feature_columns}])
            probability = float(bundle.model.predict_proba(frame)[0, 1])
            predicted_label = bundle.positive_label if probability >= bundle.threshold else bundle.negative_label
        paths = [event.path for event in session.events]
        rows.append(
            {
                "session_id": session.session_id,
                "ready": ready,
                "event_count": event_count,
                "bot_probability": probability,
                "bot_probability_pct": None if probability is None else round(probability * 100, 1),
                "predicted_label": predicted_label,
                "first_path": first_event.path,
                "last_path": last_event.path,
                "recent_paths": paths[-8:],
                "duration_seconds": round(max(0.0, last_event.timestamp - first_event.timestamp), 1),
                "last_seen_seconds_ago": round(max(0.0, time.time() - last_event.timestamp), 1),
                "user_agent": last_event.user_agent or "",
                "client_key": str(first_event.extra.get("client_key", "")),
                "navigation_entropy_score_v2": _rounded_or_none(features.get("navigation_entropy_score_v2")),
                "mean_delta_t": _rounded_or_none(features.get("mean_delta_t")),
                "low_latency_ratio": _rounded_or_none(features.get("low_latency_ratio")),
                "unique_node_ratio": _rounded_or_none(features.get("unique_node_ratio")),
            }
        )
    rows.sort(key=lambda item: (item["bot_probability"] is not None, item["bot_probability"] or -1.0, item["event_count"]), reverse=True)
    return rows


def summarize_scored_sessions(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    ready = [row for row in rows if row["ready"]]
    flagged = [row for row in ready if row["bot_probability"] is not None and row["bot_probability"] >= threshold]
    highest = max((row["bot_probability"] or 0.0 for row in ready), default=0.0)
    return {
        "total_sessions": len(rows),
        "ready_sessions": len(ready),
        "flagged_sessions": len(flagged),
        "highest_probability": round(highest, 4),
        "highest_probability_pct": round(highest * 100, 1),
        "threshold": threshold,
    }


def launch_bot_run(state: AdminState, *, mode: str, sessions: int, real_sleep: bool) -> BotRun:
    state.bot_run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{mode}_{int(time.time())}"
    log_path = state.bot_run_dir / f"{run_id}.log"
    command = [
        sys.executable,
        "-m",
        "wsd.lab_traffic",
        "--mode",
        mode,
        "--sessions",
        str(sessions),
        "--base-url",
        state.base_url,
        "--labels-path",
        str(state.labels_path),
    ]
    if real_sleep:
        command.append("--real-sleep")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Command: {' '.join(command)}\n\n")
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    run = BotRun(run_id=run_id, mode=mode, sessions=sessions, log_path=log_path, process=process)
    state.bot_runs.append(run)
    return run


def reset_live_view(state: AdminState) -> dict[str, Any]:
    messages: list[str] = []
    warnings: list[str] = []
    reset_at = time.time()
    state.ignore_log_before_timestamp = reset_at

    for run in state.bot_runs:
        if run.process.poll() is not None:
            continue
        try:
            run.process.terminate()
            run.process.wait(timeout=3)
            messages.append(f"Stopped running bot test: {run.run_id}")
        except Exception as exc:  # pragma: no cover - defensive cleanup
            warnings.append(f"Could not stop bot test {run.run_id}: {exc}")
    state.bot_runs.clear()

    for log_path in [state.log_path, state.log_path.with_name("error.log")]:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("", encoding="utf-8")
            messages.append(f"Cleared {log_path}")
        except Exception as exc:
            warnings.append(f"Could not clear {log_path}; dashboard will ignore old entries: {exc}")

    try:
        if state.labels_path.exists():
            state.labels_path.unlink()
            messages.append(f"Deleted {state.labels_path}")
    except Exception as exc:
        warnings.append(f"Could not delete {state.labels_path}: {exc}")

    try:
        if state.bot_run_dir.exists():
            shutil.rmtree(state.bot_run_dir)
            messages.append(f"Deleted {state.bot_run_dir}")
    except Exception as exc:
        warnings.append(f"Could not delete {state.bot_run_dir}: {exc}")

    return {
        "ok": not warnings,
        "reset_at": reset_at,
        "messages": messages,
        "warnings": warnings,
    }


def bot_run_to_public_dict(run: BotRun) -> dict[str, Any]:
    poll = run.process.poll()
    if poll is None:
        status = "running"
    elif poll == 0:
        status = "finished"
    else:
        status = f"failed ({poll})"
    return {
        "run_id": run.run_id,
        "mode": run.mode,
        "sessions": run.sessions,
        "status": status,
        "pid": run.process.pid,
        "age_seconds": round(time.time() - run.started_at, 1),
        "log_path": str(run.log_path),
    }


def bundle_to_public_dict(model_row: dict[str, Any], bundle: ModelBundle, *, active: bool) -> dict[str, Any]:
    public = dict(model_row)
    public["active"] = active
    public["threshold"] = bundle.threshold
    public["feature_count"] = len(bundle.feature_columns)
    return public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the standalone WSD admin dashboard")
    parser.add_argument("--model-dir", default="data/prepared_live/models")
    parser.add_argument("--graph-dir", default="data/prepared_live")
    parser.add_argument("--log-path", default="data/live_logs/access.log")
    parser.add_argument("--labels-path", default="data/live_labels/manual_labels.csv")
    parser.add_argument("--base-url", default="http://127.0.0.1:8039")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--session-timeout", type=float, default=DEFAULT_SESSION_TIMEOUT_SECONDS)
    parser.add_argument("--model", default=None, help="Optional model name to activate at startup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = AdminState(
        model_dir=Path(args.model_dir),
        graph_dir=Path(args.graph_dir),
        log_path=Path(args.log_path),
        labels_path=Path(args.labels_path),
        base_url=args.base_url,
        threshold=args.threshold,
        session_timeout_seconds=args.session_timeout,
    )
    if args.model:
        state.load_model(args.model, threshold=args.threshold)
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)


def _load_default_model_if_available(state: AdminState) -> None:
    if state.active_bundle is not None:
        return
    models = discover_model_bundles(state.model_dir)
    if not models:
        return
    preferred_order = [
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "logistic_regression",
        "calibrated_svm",
        "xgboost",
        "lightgbm",
        "catboost",
    ]
    by_name = {item["model_name"]: item for item in models}
    selected = next((by_name[name] for name in preferred_order if name in by_name), models[0])
    state.load_model(str(selected["model_name"]), threshold=state.threshold)


def _public_model_list(models: list[dict[str, Any]], active_model_name: str | None) -> list[dict[str, Any]]:
    public_models = []
    for item in models:
        public = dict(item)
        public.pop("path", None)
        public["active"] = item["model_name"] == active_model_name
        public_models.append(public)
    return public_models


def _state(request: Request) -> AdminState:
    return request.app.state.admin_state


def _rounded_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return round(numeric, 4)


def _empty_summary() -> dict[str, Any]:
    return {
        "total_sessions": 0,
        "ready_sessions": 0,
        "flagged_sessions": 0,
        "highest_probability": 0.0,
        "highest_probability_pct": 0.0,
        "threshold": 0.5,
    }


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>WSD Command Deck</title>
  <style>
    :root {
      --ink: #102331;
      --muted: #607483;
      --paper: #f6efe3;
      --card: rgba(255, 251, 242, 0.86);
      --line: rgba(16, 35, 49, 0.12);
      --teal: #0f766e;
      --teal-2: #14b8a6;
      --coral: #e85d43;
      --amber: #f0ad35;
      --navy: #112c3d;
      --green: #1f9d63;
      --shadow: 0 24px 70px rgba(17, 44, 61, 0.18);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Aptos Display", "Trebuchet MS", "Gill Sans", sans-serif;
      background:
        radial-gradient(circle at 12% 8%, rgba(20, 184, 166, 0.24), transparent 31rem),
        radial-gradient(circle at 92% 16%, rgba(232, 93, 67, 0.18), transparent 28rem),
        linear-gradient(135deg, #fff8ea 0%, #edf7f3 47%, #f7e5d6 100%);
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.22;
      background-image:
        linear-gradient(rgba(16, 35, 49, 0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(16, 35, 49, 0.08) 1px, transparent 1px);
      background-size: 48px 48px;
      mask-image: linear-gradient(to bottom, black, transparent 80%);
    }

    .shell {
      position: relative;
      width: min(1440px, calc(100% - 32px));
      margin: 0 auto;
      padding: 30px 0 48px;
    }

    header {
      display: grid;
      grid-template-columns: 1.4fr 0.9fr;
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
      animation: rise 0.55s ease both;
    }

    .hero, .panel, .session {
      border: 1px solid var(--line);
      background: var(--card);
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
    }

    .hero {
      border-radius: 34px;
      padding: 34px;
      min-height: 220px;
      overflow: hidden;
      position: relative;
    }

    .hero::after {
      content: "";
      position: absolute;
      right: 34px;
      top: 28px;
      width: 150px;
      height: 150px;
      border-radius: 44px;
      background:
        linear-gradient(135deg, rgba(15, 118, 110, 0.95), rgba(232, 93, 67, 0.82)),
        repeating-linear-gradient(45deg, transparent 0 10px, rgba(255,255,255,0.22) 10px 12px);
      transform: rotate(8deg);
      opacity: 0.9;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--teal);
    }

    .pulse {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--green);
      box-shadow: 0 0 0 0 rgba(31, 157, 99, 0.7);
      animation: pulse 1.7s infinite;
    }

    h1 {
      max-width: 720px;
      margin: 18px 0 12px;
      font-size: clamp(42px, 7vw, 86px);
      line-height: 0.88;
      letter-spacing: -0.07em;
    }

    .hero p {
      max-width: 680px;
      color: var(--muted);
      font-size: 18px;
      line-height: 1.55;
      margin: 0;
    }

    .controls {
      border-radius: 34px;
      padding: 24px;
      display: grid;
      gap: 16px;
    }

    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    select, input {
      width: 100%;
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 13px 14px;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.72);
      font: inherit;
      outline: none;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 0.7fr;
      gap: 12px;
    }

    button {
      border: 0;
      border-radius: 18px;
      padding: 14px 16px;
      color: #fff;
      background: linear-gradient(135deg, var(--navy), var(--teal));
      font-weight: 850;
      letter-spacing: -0.01em;
      cursor: pointer;
      box-shadow: 0 14px 30px rgba(15, 118, 110, 0.24);
      transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    button:hover { transform: translateY(-2px); box-shadow: 0 18px 34px rgba(15, 118, 110, 0.28); }
    button.secondary { background: linear-gradient(135deg, var(--coral), var(--amber)); }
    button.ghost {
      color: var(--navy);
      background: rgba(255, 255, 255, 0.64);
      border: 1px solid var(--line);
      box-shadow: none;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
      animation: rise 0.65s 0.05s ease both;
    }

    .metric {
      border-radius: 28px;
      padding: 22px;
      min-height: 130px;
      position: relative;
      overflow: hidden;
    }

    .metric strong {
      display: block;
      margin-top: 18px;
      font-size: 42px;
      letter-spacing: -0.06em;
    }

    .metric span { color: var(--muted); font-weight: 700; }

    .grid {
      display: grid;
      grid-template-columns: 1fr 370px;
      gap: 18px;
      align-items: start;
    }

    .panel {
      border-radius: 30px;
      padding: 22px;
      animation: rise 0.7s 0.08s ease both;
    }

    .panel-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 18px;
    }

    h2 {
      margin: 0;
      font-size: 24px;
      letter-spacing: -0.04em;
    }

    .small {
      color: var(--muted);
      font-size: 13px;
    }

    .sessions {
      display: grid;
      gap: 12px;
    }

    .session {
      border-radius: 24px;
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) 140px 130px;
      gap: 16px;
      align-items: center;
    }

    .sid {
      font-family: "Cascadia Mono", "Consolas", monospace;
      font-size: 13px;
      font-weight: 800;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .paths {
      color: var(--muted);
      margin-top: 6px;
      font-size: 13px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .score {
      height: 12px;
      background: rgba(16, 35, 49, 0.1);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }

    .score > i {
      display: block;
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--teal-2), var(--amber), var(--coral));
      transition: width 0.35s ease;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 8px 10px;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #fff;
      background: var(--green);
    }

    .badge.bot { background: var(--coral); }
    .badge.waiting { background: var(--muted); }

    .side-list {
      display: grid;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
    }

    .run {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.48);
    }

    .issue {
      border-radius: 18px;
      padding: 12px;
      color: #7a271a;
      background: rgba(232, 93, 67, 0.13);
      border: 1px solid rgba(232, 93, 67, 0.22);
    }

    @keyframes pulse {
      70% { box-shadow: 0 0 0 12px rgba(31, 157, 99, 0); }
      100% { box-shadow: 0 0 0 0 rgba(31, 157, 99, 0); }
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 980px) {
      header, .grid, .metrics { grid-template-columns: 1fr; }
      .session { grid-template-columns: 1fr; }
      .hero::after { opacity: 0.18; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <section class="hero">
        <div class="eyebrow"><i class="pulse"></i> Live Navigation Watch</div>
        <h1>WSD Command Deck</h1>
        <p>Choose a trained model, watch live sessions from Nginx, and launch controlled bot families to test early detection without retraining.</p>
      </section>
      <section class="panel controls">
        <div>
          <label for="modelSelect">Active Model</label>
          <select id="modelSelect"></select>
        </div>
        <div class="row">
          <div>
            <label for="thresholdInput">Decision Threshold</label>
            <input id="thresholdInput" type="number" min="0" max="1" step="0.01" value="0.50" />
          </div>
          <div style="align-self:end">
            <button id="activateBtn">Activate</button>
          </div>
        </div>
        <div class="row">
          <div>
            <label for="botMode">Test Bot Family</label>
            <select id="botMode"></select>
          </div>
          <div>
            <label for="botSessions">Sessions</label>
            <input id="botSessions" type="number" min="1" max="50" value="3" />
          </div>
        </div>
        <button class="secondary" id="runBotBtn">Run Selected Bot</button>
        <button class="ghost" id="resetLiveBtn">Clear Live Sessions</button>
        <div class="small">Dashboard: <span id="clock">starting...</span></div>
      </section>
    </header>

    <section class="metrics">
      <article class="panel metric"><span>Total sessions</span><strong id="mTotal">0</strong></article>
      <article class="panel metric"><span>Ready to score</span><strong id="mReady">0</strong></article>
      <article class="panel metric"><span>Flagged bots</span><strong id="mFlagged">0</strong></article>
      <article class="panel metric"><span>Highest risk</span><strong id="mHigh">0%</strong></article>
    </section>

    <section class="grid">
      <section class="panel">
        <div class="panel-title">
          <h2>Live Sessions</h2>
          <div class="small" id="activeModel">No model loaded</div>
        </div>
        <div id="issues"></div>
        <div class="sessions" id="sessions"></div>
      </section>
      <aside class="panel">
        <div class="panel-title">
          <h2>Bot Runs</h2>
          <button id="refreshBtn">Refresh</button>
        </div>
        <div class="side-list" id="botRuns"></div>
      </aside>
    </section>
  </main>
  <script>
    const botModes = ["bfs","dfs","linear","stealth","products","articles","revisit","browser_hybrid","browser_noise","playwright","selenium"];
    const botMode = document.querySelector("#botMode");
    botModes.forEach(mode => botMode.append(new Option(mode.replaceAll("_", " "), mode)));

    async function getJson(url, options) {
      const res = await fetch(url, options);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }

    async function loadModels() {
      const data = await getJson("/api/models");
      const select = document.querySelector("#modelSelect");
      select.innerHTML = "";
      data.models.forEach(model => {
        const label = `${model.model_name} (${model.size_kb} KB)${model.active ? " active" : ""}`;
        select.append(new Option(label, model.model_name, model.active, model.active));
      });
      document.querySelector("#thresholdInput").value = Number(data.threshold || 0.5).toFixed(2);
    }

    async function activateModel() {
      const model_name = document.querySelector("#modelSelect").value;
      const threshold = Number(document.querySelector("#thresholdInput").value);
      await getJson("/api/model", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({model_name, threshold})
      });
      await loadModels();
      await refresh();
    }

    async function runBot() {
      const mode = document.querySelector("#botMode").value;
      const sessions = Number(document.querySelector("#botSessions").value || 3);
      await getJson("/api/run-bot", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({mode, sessions, real_sleep: true})
      });
      await refresh();
    }

    async function resetLive() {
      const confirmed = confirm("Clear live sessions, live labels, and admin bot-run logs? Archived human sessions are kept.");
      if (!confirmed) return;
      await getJson("/api/reset-live", { method: "POST" });
      await refresh();
    }

    function renderIssues(issues) {
      const box = document.querySelector("#issues");
      box.innerHTML = "";
      (issues || []).forEach(issue => {
        const div = document.createElement("div");
        div.className = "issue";
        div.textContent = issue;
        box.append(div);
      });
    }

    function renderSessions(rows) {
      const root = document.querySelector("#sessions");
      root.innerHTML = "";
      if (!rows.length) {
        root.innerHTML = `<div class="small">No sessions in the live log yet. Open the site or run a bot test.</div>`;
        return;
      }
      rows.slice(0, 40).forEach(row => {
        const prob = row.bot_probability_pct ?? 0;
        const label = row.ready ? row.predicted_label : "waiting";
        const badgeClass = label === "bot" ? "bot" : (label === "waiting" ? "waiting" : "");
        const div = document.createElement("article");
        div.className = "session";
        div.innerHTML = `
          <div>
            <div class="sid">${row.session_id}</div>
            <div class="paths">${row.recent_paths.join("  ›  ")}</div>
            <div class="score"><i style="width:${prob}%"></i></div>
          </div>
          <div>
            <strong>${row.bot_probability_pct === null ? "..." : prob + "%"}</strong>
            <div class="small">entropy v2: ${row.navigation_entropy_score_v2 ?? "n/a"}</div>
          </div>
          <div>
            <span class="badge ${badgeClass}">${label}</span>
            <div class="small">${row.event_count} req · last ${row.last_seen_seconds_ago}s</div>
          </div>
        `;
        root.append(div);
      });
    }

    function renderRuns(runs) {
      const root = document.querySelector("#botRuns");
      root.innerHTML = "";
      if (!runs.length) {
        root.innerHTML = `<div class="small">No bot tests launched from this panel yet.</div>`;
        return;
      }
      runs.forEach(run => {
        const div = document.createElement("div");
        div.className = "run";
        div.innerHTML = `<strong>${run.mode}</strong><div>${run.status} · ${run.sessions} sessions · ${run.age_seconds}s</div><div class="small">${run.log_path}</div>`;
        root.append(div);
      });
    }

    async function refresh() {
      try {
        const data = await getJson("/api/status");
        renderIssues(data.issues || []);
        const s = data.summary;
        document.querySelector("#mTotal").textContent = s.total_sessions;
        document.querySelector("#mReady").textContent = s.ready_sessions;
        document.querySelector("#mFlagged").textContent = s.flagged_sessions;
        document.querySelector("#mHigh").textContent = `${s.highest_probability_pct}%`;
        document.querySelector("#activeModel").textContent = data.active_model ? `${data.active_model} @ ${Number(data.threshold).toFixed(2)}` : "No model loaded";
        document.querySelector("#clock").textContent = new Date().toLocaleTimeString();
        renderSessions(data.sessions || []);
        renderRuns(data.bot_runs || []);
      } catch (error) {
        renderIssues([String(error)]);
      }
    }

    document.querySelector("#activateBtn").addEventListener("click", activateModel);
    document.querySelector("#runBotBtn").addEventListener("click", runBot);
    document.querySelector("#resetLiveBtn").addEventListener("click", resetLive);
    document.querySelector("#refreshBtn").addEventListener("click", refresh);

    loadModels().then(refresh);
    setInterval(refresh, 2500);
    setInterval(loadModels, 15000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
