# Local Nginx + Website Lab Setup (Windows)

This repository includes a complete local lab for the research objective described in the proposal: early detection of scraping behavior from website traversal logs and navigation entropy.

## What this setup does

- hosts `website_lab/` through **Nginx for Windows**
- writes access logs through Nginx's native `tools/nginx-*/logs/` directory and exposes the same files at `data/live_logs/` via a Windows junction when available
- generates labeled human-like and bot traffic against the site
- prepares a trainable dataset from the collected log
- runs the existing training and experiment pipeline on the resulting data

## One-time setup

From **PowerShell** in the project root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\lab\scripts\setup_nginx_windows.ps1
```

## Start the local website

```bat
lab\scripts\start_nginx_windows.bat
```

Open `http://127.0.0.1:8039` in your browser.
If Nginx is already running, the start script reloads it and returns immediately instead of blocking the terminal.
If the site loads but `data/live_logs/access.log` stays empty, you likely still have an older admin-started `nginx.exe` process bound to port `8039`; clear those stale listeners from an elevated shell and start again.

## Stop Nginx

```bat
lab\scripts\stop_nginx_windows.bat
```

## Reset logs before a new experiment

```bat
lab\scripts\reset_live_logs.bat
```

Run the reset before `lab\scripts\start_nginx_windows.bat` when you want a clean capture window.

## Generate local labeled traffic

```bat
lab\scripts\generate_human_traffic.bat
lab\scripts\generate_bot_bfs_traffic.bat
lab\scripts\generate_bot_dfs_traffic.bat
lab\scripts\generate_bot_linear_traffic.bat
lab\scripts\generate_bot_stealth_traffic.bat
lab\scripts\generate_bot_products_traffic.bat
lab\scripts\generate_bot_articles_traffic.bat
lab\scripts\generate_bot_revisit_traffic.bat
```

These scripts update `data/live_labels/manual_labels.csv`.

## Prepare a dataset from the collected logs

```bat
lab\scripts\prepare_live_dataset.bat
```

## Train and run research experiments

```bat
lab\scripts\run_research_pipeline.bat
```


## Logging note

The Windows setup now keeps Nginx's source-of-truth logs in `tools/nginx-*/logs/` because that is the most reliable location for Nginx on Windows. The setup script then maps `data/live_logs/` to the same directory, so the rest of the project can still use `data/live_logs/access.log`.
