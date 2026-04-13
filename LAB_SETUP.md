# Local Nginx Website Lab Setup (Windows)

This lab supports the thesis workflow: collect website traffic, label sessions, build graph/entropy features, and evaluate early bot detection.

## One-time setup

From PowerShell in the repo root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\lab\scripts\setup_nginx_windows.ps1
```

## Clean start

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\stop_nginx_windows.ps1
lab\scripts\reset_live_logs.bat
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\start_nginx_windows.ps1
```

Open:

```text
http://127.0.0.1:8039/
```

If the site loads but `data\live_logs\access.log` stays empty, clear stale old `nginx.exe` processes that are still bound to port `8039`, then run the clean start block again.

## Stop Nginx

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\stop_nginx_windows.ps1
```

## Traffic generation

Human-like traffic:

```bat
lab\scripts\generate_human_traffic.bat
```

Classic scripted bot families:

```bat
lab\scripts\generate_bot_bfs_traffic.bat
lab\scripts\generate_bot_dfs_traffic.bat
lab\scripts\generate_bot_linear_traffic.bat
lab\scripts\generate_bot_stealth_traffic.bat
lab\scripts\generate_bot_products_traffic.bat
lab\scripts\generate_bot_articles_traffic.bat
lab\scripts\generate_bot_revisit_traffic.bat
```

Harder browser-style bot families:

```bat
lab\scripts\generate_bot_browser_hybrid_traffic.bat
lab\scripts\generate_bot_browser_noise_traffic.bat
lab\scripts\generate_bot_playwright_traffic.bat
lab\scripts\generate_bot_selenium_traffic.bat
```

Notes:

- `playwright` mode requires `playwright install`
- Selenium mode uses Python Selenium WebDriver
- Generated labels are written to `data\live_labels\manual_labels.csv`

## Manual human collection

The label CSV now supports optional metadata columns:

- `participant_id`
- `traffic_family`
- `collection_method`
- `automation_stack`
- `notes`

Recommended human collection tasks:

- explore the homepage and navigation pages
- compare 2-3 products
- read an article, then return to products
- use search, cart, FAQ, and contact pages
- backtrack and revisit naturally

Recommended metadata example:

```csv
client_key,label,participant_id,traffic_family,collection_method,notes
ip_ua:127.0.0.1|Mozilla/5.0...,human,p07,human_navigation,manual_browser,article-to-product task
```

## Prepare the live dataset

```bat
lab\scripts\prepare_live_dataset.bat
```

This produces `data\prepared_live\` with:

- `requests.csv`
- `session_summary.csv`
- `graph_edges.csv`
- `graph_categories.csv`

## Run the thesis experiment pipeline

```bat
lab\scripts\run_research_pipeline.bat
```

That now runs:

- training on `data\prepared_live`
- grouped session-split experiments
- hard-prefix reporting
- leave-one-bot-family-out
- time-based split
- leave-one-human-user-out when `participant_id` is present
- shortcut/leakage audits
- entropy-variant comparison

## Useful output paths

- `data\prepared_live\experiments\summary.md`
- `data\prepared_live\experiments\leaderboard.csv`
- `data\prepared_live\experiments\leakage_audit.csv`
- `data\prepared_live\experiments\shortcut_audit.csv`
- `data\prepared_live\experiments\entropy_variant_comparison.csv`
