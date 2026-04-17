"""One-command runtime for the generic multi-site dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from generic_models.generic_admin_panel import GenericAdminState, create_app
from generic_models.generic_site_server import start_all_site_servers, stop_site_servers
from generic_models.site_catalog import write_graph_exports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the generic WSD lab: four sites plus admin dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--admin-port", type=int, default=8050)
    parser.add_argument("--model-dir", default="generic_models/artifacts/models")
    parser.add_argument("--log-dir", default="generic_models/live_logs")
    parser.add_argument("--graph-dir", default="generic_models/live_graphs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_graph_exports(args.graph_dir)
    servers = start_all_site_servers(host=args.host, log_dir=log_dir)
    print("Generic websites are running:")
    for running in servers:
        print(f"  {running.spec.name:16s} {running.url}  ({running.spec.shape})")
    print(f"Generic admin panel: http://{args.host}:{args.admin_port}/")

    state = GenericAdminState(model_dir=Path(args.model_dir), log_dir=log_dir, host=args.host)
    app = create_app(state)
    try:
        uvicorn.run(app, host=args.host, port=args.admin_port)
    finally:
        stop_site_servers(servers)


if __name__ == "__main__":
    main()
