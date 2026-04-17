"""Admin dashboard for generic website models and multi-site live testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
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

from generic_models.generic_traffic import BOT_MODES
from generic_models.site_catalog import WebsiteSpec, build_all_generic_sites, get_websites
from generic_models.train_generic_models import GenericSession, GenericSite, extract_generic_features, predict_proba


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_TIMEOUT_SECONDS = 30 * 60
MIN_EVENTS_TO_SCORE = 3
MAX_PREFIX_TO_SCORE = 20


class SelectGenericModelPayload(BaseModel):
    model_name: str
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class RunGenericBotPayload(BaseModel):
    site_id: str
    mode: str
    sessions: int = Field(default=3, ge=1, le=50)
    real_sleep: bool = True


@dataclass(slots=True)
class GenericBotRun:
    run_id: str
    site_id: str
    mode: str
    sessions: int
    log_path: Path
    process: subprocess.Popen[Any]
    started_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class GenericAdminState:
    model_dir: Path
    log_dir: Path
    host: str = "127.0.0.1"
    threshold: float = 0.5
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS
    bot_run_dir: Path = Path("generic_models/admin_bot_runs")
    sites: dict[str, WebsiteSpec] = field(default_factory=get_websites)
    generic_sites: dict[str, GenericSite] = field(default_factory=build_all_generic_sites)
    active_model_name: str | None = None
    active_bundle: dict[str, Any] | None = None
    bot_runs: list[GenericBotRun] = field(default_factory=list)
    ignore_log_before_timestamp: float = 0.0

    def load_model(self, model_name: str, threshold: float | None = None) -> dict[str, Any]:
        models = discover_generic_model_bundles(self.model_dir)
        match = next((item for item in models if item["model_name"] == model_name or item["file_name"] == model_name), None)
        if match is None:
            raise FileNotFoundError(f"Generic model bundle not found: {model_name}")
        bundle = load_generic_bundle(match["path"])
        if threshold is not None:
            bundle["threshold"] = float(threshold)
        else:
            bundle["threshold"] = float(bundle.get("threshold", self.threshold))
        self.active_model_name = str(match["model_name"])
        self.active_bundle = bundle
        self.threshold = float(bundle.get("threshold", self.threshold))
        return generic_bundle_to_public_dict(match, bundle, active=True)


def discover_generic_model_bundles(model_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(model_dir)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_generic_bundle.pkl")):
        model_name = path.stem.removesuffix("_generic_bundle")
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


def load_generic_bundle(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise TypeError(f"Unsupported generic bundle format: {path}")
    if "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError(f"Generic bundle is missing model or feature_columns: {path}")
    return bundle


def create_app(state: GenericAdminState) -> FastAPI:
    app = FastAPI(title="Generic WSD Admin Panel", version="0.1.0")
    app.state.generic_admin_state = state
    _load_default_model_if_available(state)

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/models")
    def api_models(request: Request) -> dict[str, Any]:
        state = _state(request)
        models = discover_generic_model_bundles(state.model_dir)
        return {
            "models": _public_model_list(models, state.active_model_name),
            "active_model": state.active_model_name,
            "threshold": state.threshold,
            "model_dir": str(state.model_dir),
        }

    @app.post("/api/model")
    def api_select_model(payload: SelectGenericModelPayload, request: Request) -> dict[str, Any]:
        state = _state(request)
        try:
            selected = state.load_model(payload.model_name, threshold=payload.threshold)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        selected.pop("path", None)
        return {"selected": selected, "threshold": state.threshold}

    @app.get("/api/sites")
    def api_sites(request: Request) -> dict[str, Any]:
        state = _state(request)
        return {
            "sites": [site_to_public_dict(spec, state.host) for spec in state.sites.values()],
            "bot_modes": BOT_MODES,
        }

    @app.get("/api/status")
    def api_status(request: Request) -> dict[str, Any]:
        state = _state(request)
        return score_live_logs(state)

    @app.post("/api/run-bot")
    def api_run_bot(payload: RunGenericBotPayload, request: Request) -> dict[str, Any]:
        state = _state(request)
        if payload.site_id not in state.sites:
            raise HTTPException(status_code=400, detail=f"Unsupported site id: {payload.site_id}")
        if payload.mode not in BOT_MODES:
            raise HTTPException(status_code=400, detail=f"Unsupported bot mode: {payload.mode}")
        run = launch_generic_bot_run(
            state,
            site_id=payload.site_id,
            mode=payload.mode,
            sessions=payload.sessions,
            real_sleep=payload.real_sleep,
        )
        return generic_bot_run_to_public_dict(run)

    @app.post("/api/reset-live")
    def api_reset_live(request: Request) -> dict[str, Any]:
        state = _state(request)
        return reset_generic_live_view(state)

    @app.get("/api/bot-runs")
    def api_bot_runs(request: Request) -> dict[str, Any]:
        state = _state(request)
        return {"runs": [generic_bot_run_to_public_dict(run) for run in state.bot_runs[-12:]][::-1]}

    return app


def score_live_logs(state: GenericAdminState) -> dict[str, Any]:
    models = discover_generic_model_bundles(state.model_dir)
    if state.active_bundle is None:
        _load_default_model_if_available(state)

    issues: list[str] = []
    if state.active_bundle is None:
        issues.append(f"No generic model bundle loaded from: {state.model_dir}")

    raw_df = read_generic_log_frame(state.log_dir, ignore_before=state.ignore_log_before_timestamp)
    sessions = build_sessions_from_log_frame(raw_df, state=state)
    session_rows = []
    if state.active_bundle is not None:
        try:
            session_rows = score_generic_sessions(sessions, state=state)
        except Exception as exc:  # pragma: no cover - defensive dashboard message
            issues.append(f"Could not score generic live sessions: {exc}")

    return {
        "ok": not issues,
        "issues": issues,
        "models": _public_model_list(models, state.active_model_name),
        "active_model": state.active_model_name,
        "threshold": state.threshold,
        "summary": summarize_scored_sessions(session_rows, threshold=state.threshold),
        "sites": site_summaries(state, session_rows),
        "sessions": session_rows,
        "bot_runs": [generic_bot_run_to_public_dict(run) for run in state.bot_runs[-6:]][::-1],
        "updated_at": time.time(),
        "paths": {"log_dir": str(state.log_dir), "model_dir": str(state.model_dir)},
    }


def read_generic_log_frame(log_dir: str | Path, *, ignore_before: float = 0.0) -> pd.DataFrame:
    root = Path(log_dir)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame(rows)
    for path in sorted(root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                timestamp = float(record.get("timestamp", 0.0) or 0.0)
                if ignore_before > 0 and timestamp < ignore_before:
                    continue
                rows.append(
                    {
                        "timestamp": timestamp,
                        "site_id": str(record.get("site_id", path.stem)),
                        "ip": str(record.get("ip", "")),
                        "user_agent": str(record.get("user_agent", "")),
                        "path": str(record.get("path", "")),
                        "status_code": int(record.get("status_code", 0) or 0),
                        "referrer": str(record.get("referrer", "")),
                    }
                )
    return pd.DataFrame(rows)


def build_sessions_from_log_frame(raw_df: pd.DataFrame, *, state: GenericAdminState) -> list[GenericSession]:
    if raw_df.empty:
        return []
    valid_site_ids = set(state.sites)
    rows = raw_df[(raw_df["status_code"] == 200) & raw_df["site_id"].isin(valid_site_ids)].copy()
    if rows.empty:
        return []
    sessions: list[GenericSession] = []
    for (site_id, ip, user_agent), group in rows.sort_values("timestamp").groupby(["site_id", "ip", "user_agent"], dropna=False):
        site = state.generic_sites[str(site_id)]
        valid_paths = set(site.graph.nodes)
        current_paths: list[str] = []
        current_timestamps: list[float] = []
        session_index = 0
        last_timestamp: float | None = None
        for record in group.to_dict("records"):
            path = str(record["path"])
            timestamp = float(record["timestamp"])
            if path not in valid_paths:
                continue
            if last_timestamp is not None and timestamp - last_timestamp > state.session_timeout_seconds:
                _append_session(
                    sessions,
                    site_id=str(site_id),
                    ip=str(ip),
                    user_agent=str(user_agent),
                    session_index=session_index,
                    paths=current_paths,
                    timestamps=current_timestamps,
                )
                session_index += 1
                current_paths = []
                current_timestamps = []
            current_paths.append(path)
            current_timestamps.append(timestamp)
            last_timestamp = timestamp
        _append_session(
            sessions,
            site_id=str(site_id),
            ip=str(ip),
            user_agent=str(user_agent),
            session_index=session_index,
            paths=current_paths,
            timestamps=current_timestamps,
        )
    return sessions


def score_generic_sessions(sessions: list[GenericSession], *, state: GenericAdminState) -> list[dict[str, Any]]:
    if state.active_bundle is None:
        return []
    feature_columns = list(state.active_bundle.get("feature_columns", []))
    model = state.active_bundle["model"]
    rows: list[dict[str, Any]] = []
    now = time.time()
    for session in sessions:
        spec = state.sites[session.site_id]
        site = state.generic_sites[session.site_id]
        ready = len(session.paths) >= MIN_EVENTS_TO_SCORE
        score: float | None = None
        features: dict[str, float] = {}
        if ready:
            prefix_len = min(len(session.paths), MAX_PREFIX_TO_SCORE)
            features = extract_generic_features(session, site, prefix_len=prefix_len)
            features["prefix_len"] = float(prefix_len)
            frame = pd.DataFrame([{column: float(features.get(column, 0.0)) for column in feature_columns}])
            score = float(predict_proba(model, frame)[0])
        predicted_label = "waiting"
        if score is not None:
            predicted_label = "bot" if score >= state.threshold else "human"
        rows.append(
            {
                "session_id": session.session_id,
                "site_id": session.site_id,
                "site_name": spec.name,
                "graph_shape": spec.shape,
                "family": session.family,
                "ready": ready,
                "event_count": len(session.paths),
                "distinct_paths": len(set(session.paths)),
                "duration_seconds": round(max(0.0, session.timestamps[-1] - session.timestamps[0]) if len(session.timestamps) >= 2 else 0.0, 2),
                "last_seen_seconds_ago": int(max(0.0, now - session.timestamps[-1])) if session.timestamps else 0,
                "first_path": session.paths[0] if session.paths else "",
                "last_path": session.paths[-1] if session.paths else "",
                "recent_paths": session.paths[-7:],
                "bot_probability": score,
                "bot_probability_pct": None if score is None else int(round(score * 100)),
                "predicted_label": predicted_label,
                "threshold": state.threshold,
                "coverage_ratio": _round_feature(features.get("coverage_ratio")),
                "path_entropy": _round_feature(features.get("path_entropy")),
                "revisit_rate": _round_feature(features.get("revisit_rate")),
                "navigation_entropy_generic_score": _round_feature(features.get("navigation_entropy_generic_score")),
                "graph_distance_mean": _round_feature(features.get("graph_distance_mean")),
                "backtrack_ratio": _round_feature(features.get("backtrack_ratio")),
            }
        )
    rows.sort(key=lambda row: (row["bot_probability"] is not None, row["bot_probability"] or -1.0, row["event_count"]), reverse=True)
    return rows


def launch_generic_bot_run(
    state: GenericAdminState,
    *,
    site_id: str,
    mode: str,
    sessions: int,
    real_sleep: bool,
) -> GenericBotRun:
    state.bot_run_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{site_id}_{mode}"
    log_path = state.bot_run_dir / f"{run_id}.log"
    spec = state.sites[site_id]
    command = [
        sys.executable,
        "-m",
        "generic_models.generic_traffic",
        "--site-id",
        site_id,
        "--mode",
        mode,
        "--sessions",
        str(sessions),
        "--base-url",
        f"http://{state.host}:{spec.port}",
    ]
    if real_sleep:
        command.append("--real-sleep")
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_handle.close()
    run = GenericBotRun(run_id=run_id, site_id=site_id, mode=mode, sessions=sessions, log_path=log_path, process=process)
    state.bot_runs.append(run)
    return run


def reset_generic_live_view(state: GenericAdminState) -> dict[str, Any]:
    for run in state.bot_runs:
        if run.process.poll() is None:
            run.process.terminate()
    if state.log_dir.exists():
        for path in state.log_dir.glob("*.jsonl"):
            path.unlink()
    state.log_dir.mkdir(parents=True, exist_ok=True)
    for site_id in state.sites:
        (state.log_dir / f"{site_id}.jsonl").write_text("", encoding="utf-8")
    if state.bot_run_dir.exists():
        shutil.rmtree(state.bot_run_dir)
    state.bot_run_dir.mkdir(parents=True, exist_ok=True)
    state.bot_runs.clear()
    state.ignore_log_before_timestamp = time.time()
    return {"ok": True, "message": "Generic live logs and panel bot-run history were cleared."}


def summarize_scored_sessions(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    ready = [row for row in rows if row["ready"]]
    scored = [row for row in ready if row["bot_probability"] is not None]
    flagged = [row for row in scored if row["bot_probability"] is not None and row["bot_probability"] >= threshold]
    highest = max([float(row["bot_probability"] or 0.0) for row in scored], default=0.0)
    return {
        "total_sessions": len(rows),
        "ready_sessions": len(ready),
        "flagged_sessions": len(flagged),
        "highest_probability_pct": int(round(highest * 100)),
        "sites_with_activity": len({row["site_id"] for row in rows}),
    }


def site_summaries(state: GenericAdminState, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for spec in state.sites.values():
        site_rows = [row for row in rows if row["site_id"] == spec.site_id]
        ready_rows = [row for row in site_rows if row["ready"]]
        flagged = [row for row in ready_rows if row["predicted_label"] == "bot"]
        highest = max([float(row["bot_probability"] or 0.0) for row in ready_rows], default=0.0)
        public = site_to_public_dict(spec, state.host)
        public.update(
            {
                "sessions": len(site_rows),
                "ready_sessions": len(ready_rows),
                "flagged_sessions": len(flagged),
                "highest_probability_pct": int(round(highest * 100)),
            }
        )
        result.append(public)
    return result


def site_to_public_dict(spec: WebsiteSpec, host: str) -> dict[str, Any]:
    return {
        "site_id": spec.site_id,
        "name": spec.name,
        "archetype": spec.archetype,
        "port": spec.port,
        "shape": spec.shape,
        "accent": spec.accent,
        "url": f"http://{host}:{spec.port}/",
        "pages": len(spec.pages),
        "edges": sum(len(page.links) for page in spec.pages),
    }


def generic_bundle_to_public_dict(match: dict[str, Any], bundle: dict[str, Any], *, active: bool) -> dict[str, Any]:
    public = dict(match)
    public.update(
        {
            "active": active,
            "threshold": float(bundle.get("threshold", 0.5)),
            "feature_count": len(bundle.get("feature_columns", [])),
            "training_scope": bundle.get("training_scope", "generic"),
        }
    )
    return public


def generic_bot_run_to_public_dict(run: GenericBotRun) -> dict[str, Any]:
    poll = run.process.poll()
    return {
        "run_id": run.run_id,
        "site_id": run.site_id,
        "mode": run.mode,
        "sessions": run.sessions,
        "status": "running" if poll is None else f"exit {poll}",
        "age_seconds": int(time.time() - run.started_at),
        "log_path": str(run.log_path),
    }


def _append_session(
    sessions: list[GenericSession],
    *,
    site_id: str,
    ip: str,
    user_agent: str,
    session_index: int,
    paths: list[str],
    timestamps: list[float],
) -> None:
    if not paths:
        return
    token = hashlib.sha1(f"{site_id}|{ip}|{user_agent}|{session_index}".encode("utf-8")).hexdigest()[:10]
    family = _infer_family(user_agent)
    sessions.append(
        GenericSession(
            session_id=f"generic_live_{site_id}_{token}_{session_index:02d}",
            site_id=site_id,
            family=family,
            label="unknown",
            paths=list(paths),
            timestamps=list(timestamps),
        )
    )


def _infer_family(user_agent: str) -> str:
    lowered = user_agent.lower()
    if "genericwsdtestbot/" in lowered:
        return lowered.split("genericwsdtestbot/", 1)[1].split()[0].split("/", 1)[0]
    if any(marker in lowered for marker in ["chrome", "firefox", "safari", "edge"]):
        return "browser_human_or_browser_bot"
    if any(marker in lowered for marker in ["bot", "crawler", "spider", "python", "requests"]):
        return "bot_like_client"
    return "unknown_client"


def _load_default_model_if_available(state: GenericAdminState) -> None:
    if state.active_bundle is not None:
        return
    models = discover_generic_model_bundles(state.model_dir)
    if not models:
        return
    preferred = ["hist_gradient_boosting", "lightgbm", "random_forest", "xgboost", "logistic_regression"]
    selected = None
    for name in preferred:
        selected = next((model for model in models if model["model_name"] == name), None)
        if selected is not None:
            break
    selected = selected or models[0]
    state.load_model(str(selected["model_name"]))


def _public_model_list(models: list[dict[str, Any]], active_model_name: str | None) -> list[dict[str, Any]]:
    public_models = []
    for item in models:
        public = dict(item)
        public.pop("path", None)
        public["active"] = item["model_name"] == active_model_name
        public_models.append(public)
    return public_models


def _state(request: Request) -> GenericAdminState:
    return request.app.state.generic_admin_state


def _round_feature(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the generic WSD admin panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--model-dir", default="generic_models/artifacts/models")
    parser.add_argument("--log-dir", default="generic_models/live_logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = GenericAdminState(model_dir=Path(args.model_dir), log_dir=Path(args.log_dir), host=args.host)
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Generic WSD Command Deck</title>
  <style>
    :root {
      --ink: #102331;
      --muted: #607483;
      --paper: #f6efe3;
      --card: rgba(255, 251, 242, 0.88);
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
      opacity: 0.2;
      background-image:
        linear-gradient(rgba(16, 35, 49, 0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(16, 35, 49, 0.08) 1px, transparent 1px);
      background-size: 48px 48px;
      mask-image: linear-gradient(to bottom, black, transparent 80%);
    }
    .shell {
      position: relative;
      width: min(1480px, calc(100% - 32px));
      margin: 0 auto;
      padding: 30px 0 48px;
    }
    header {
      display: grid;
      grid-template-columns: 1.35fr 0.95fr;
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
      animation: rise 0.55s ease both;
    }
    .hero, .panel, .session, .site-card {
      border: 1px solid var(--line);
      background: var(--card);
      backdrop-filter: blur(18px);
      box-shadow: var(--shadow);
    }
    .hero {
      border-radius: 34px;
      padding: 34px;
      min-height: 230px;
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
      font-weight: 900;
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
      max-width: 780px;
      margin: 18px 0 12px;
      font-size: clamp(40px, 7vw, 82px);
      line-height: 0.88;
      letter-spacing: -0.07em;
    }
    .hero p {
      max-width: 740px;
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
      font-weight: 900;
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
      font-weight: 900;
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
    .metric span { color: var(--muted); font-weight: 800; }
    .sites {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
      animation: rise 0.66s 0.06s ease both;
    }
    .site-card {
      border-radius: 26px;
      padding: 18px;
      overflow: hidden;
      position: relative;
    }
    .site-card::before {
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 7px;
      background: var(--site-accent, var(--teal));
    }
    .site-card h3 {
      margin: 0 0 6px;
      font-size: 22px;
      letter-spacing: -0.04em;
    }
    .site-card a { color: var(--teal); font-weight: 900; text-decoration: none; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 380px;
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
      grid-template-columns: minmax(0, 1.15fr) 150px 142px;
      gap: 16px;
      align-items: center;
    }
    .sid {
      font-family: "Cascadia Mono", "Consolas", monospace;
      font-size: 13px;
      font-weight: 900;
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
    .run, .issue {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.48);
    }
    .issue {
      color: #7a271a;
      background: rgba(232, 93, 67, 0.13);
      border-color: rgba(232, 93, 67, 0.22);
      margin-bottom: 10px;
    }
    @keyframes pulse {
      70% { box-shadow: 0 0 0 12px rgba(31, 157, 99, 0); }
      100% { box-shadow: 0 0 0 0 rgba(31, 157, 99, 0); }
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 1100px) {
      header, .grid, .metrics, .sites { grid-template-columns: 1fr; }
      .session { grid-template-columns: 1fr; }
      .hero::after { opacity: 0.18; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <section class="hero">
        <div class="eyebrow"><i class="pulse"></i> Generic Multi-Site Watch</div>
        <h1>Generic WSD Command Deck</h1>
        <p>Choose a generic model, test it across four local websites with different graph shapes, and launch controlled bot families without touching the original single-site dashboard.</p>
      </section>
      <section class="panel controls">
        <div>
          <label for="modelSelect">Active Generic Model</label>
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
        <div>
          <label for="siteSelect">Target Website</label>
          <select id="siteSelect"></select>
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
        <button class="secondary" id="runBotBtn">Run Bot On Selected Site</button>
        <button class="ghost" id="resetLiveBtn">Clear Generic Live Sessions</button>
        <div class="small">Dashboard: <span id="clock">starting...</span></div>
      </section>
    </header>

    <section class="metrics">
      <article class="panel metric"><span>Total sessions</span><strong id="mTotal">0</strong></article>
      <article class="panel metric"><span>Ready to score</span><strong id="mReady">0</strong></article>
      <article class="panel metric"><span>Flagged bots</span><strong id="mFlagged">0</strong></article>
      <article class="panel metric"><span>Highest risk</span><strong id="mHigh">0%</strong></article>
    </section>

    <section class="sites" id="siteCards"></section>

    <section class="grid">
      <section class="panel">
        <div class="panel-title">
          <h2>Live Generic Sessions</h2>
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

    async function loadSites() {
      const data = await getJson("/api/sites");
      const siteSelect = document.querySelector("#siteSelect");
      siteSelect.innerHTML = "";
      data.sites.forEach(site => {
        siteSelect.append(new Option(`${site.name} : ${site.shape} : ${site.port}`, site.site_id));
      });
      const modeSelect = document.querySelector("#botMode");
      modeSelect.innerHTML = "";
      data.bot_modes.forEach(mode => modeSelect.append(new Option(mode.replaceAll("_", " "), mode)));
      renderSites(data.sites);
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
      const site_id = document.querySelector("#siteSelect").value;
      const mode = document.querySelector("#botMode").value;
      const sessions = Number(document.querySelector("#botSessions").value || 3);
      await getJson("/api/run-bot", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({site_id, mode, sessions, real_sleep: true})
      });
      await refresh();
    }

    async function resetLive() {
      const confirmed = confirm("Clear generic live sessions and panel bot-run logs? The trained generic models are kept.");
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

    function renderSites(sites) {
      const root = document.querySelector("#siteCards");
      root.innerHTML = "";
      sites.forEach(site => {
        const div = document.createElement("article");
        div.className = "site-card";
        div.style.setProperty("--site-accent", site.accent || "#0f766e");
        div.innerHTML = `
          <h3>${site.name}</h3>
          <div class="small">${site.shape}</div>
          <div class="small">${site.pages} pages / ${site.edges} links / port ${site.port}</div>
          <div style="margin-top:10px"><a href="${site.url}" target="_blank">Open site</a></div>
          <div class="small" style="margin-top:8px">${site.sessions || 0} sessions, ${site.flagged_sessions || 0} flagged, high ${site.highest_probability_pct || 0}%</div>
        `;
        root.append(div);
      });
    }

    function renderSessions(rows) {
      const root = document.querySelector("#sessions");
      root.innerHTML = "";
      if (!rows.length) {
        root.innerHTML = `<div class="small">No generic live sessions yet. Open one of the four sites or run a bot test.</div>`;
        return;
      }
      rows.slice(0, 48).forEach(row => {
        const prob = row.bot_probability_pct ?? 0;
        const label = row.ready ? row.predicted_label : "waiting";
        const badgeClass = label === "bot" ? "bot" : (label === "waiting" ? "waiting" : "");
        const div = document.createElement("article");
        div.className = "session";
        div.innerHTML = `
          <div>
            <div class="sid">${row.site_name} / ${row.session_id}</div>
            <div class="paths">${(row.recent_paths || []).join(" -> ")}</div>
            <div class="score"><i style="width:${prob}%"></i></div>
          </div>
          <div>
            <strong>${row.bot_probability_pct === null ? "..." : prob + "%"}</strong>
            <div class="small">generic entropy: ${row.navigation_entropy_generic_score ?? "n/a"}</div>
            <div class="small">coverage: ${row.coverage_ratio ?? "n/a"} / revisit: ${row.revisit_rate ?? "n/a"}</div>
          </div>
          <div>
            <span class="badge ${badgeClass}">${label}</span>
            <div class="small">${row.event_count} req / last ${row.last_seen_seconds_ago}s</div>
            <div class="small">${row.graph_shape}</div>
          </div>
        `;
        root.append(div);
      });
    }

    function renderRuns(runs) {
      const root = document.querySelector("#botRuns");
      root.innerHTML = "";
      if (!runs.length) {
        root.innerHTML = `<div class="small">No generic bot tests launched yet.</div>`;
        return;
      }
      runs.forEach(run => {
        const div = document.createElement("div");
        div.className = "run";
        div.innerHTML = `<strong>${run.site_id} / ${run.mode}</strong><div>${run.status} / ${run.sessions} sessions / ${run.age_seconds}s</div><div class="small">${run.log_path}</div>`;
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
        renderSites(data.sites || []);
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

    Promise.all([loadModels(), loadSites()]).then(refresh);
    setInterval(refresh, 2500);
    setInterval(loadModels, 15000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
