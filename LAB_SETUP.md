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

Real collected human traffic from a saved access log:

```bat
lab\scripts\reset_live_logs.bat
lab\scripts\import_human_sessions.bat logs.txt
```

This archives the raw human log in `data\human_session_archives\`, creates `identified_human_sessions.csv` with one row per session, copies a cleaned working log into `data\live_logs\access.log`, and writes session-level human labels into `data\live_labels\manual_labels.csv`. The full raw copy is preserved, but common link-preview/crawler agents such as WhatsApp and Google Read Aloud are filtered out of the working import by default. Use this instead of `generate_human_traffic.bat` for thesis runs with real people.

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

If you want to manually curate sessions after preparation, export an annotation sheet:

```powershell
python -m wsd.export_label_template `
  --session-summary data\prepared_live\session_summary.csv `
  --output-path data\prepared_live\annotation_template.csv
```

That sheet is keyed by `session_id`, which makes it easier to add real human metadata such as `participant_id`, `collection_method`, and notes before rerunning preparation/experiments.

## Run the thesis experiment pipeline

```bat
lab\scripts\run_research_pipeline.bat
```

For a clean thesis run that restores archived real-human sessions, generates bot traffic, prepares the dataset, and logs the full output:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\run_research_pipeline.ps1 -CleanLiveData -GenerateSampleTraffic -RunPipeline -SkipDependencyInstall -LogPath run_logs.txt
```

When real-human archives exist, this skips the scripted human generator and uses the archived human sessions instead. Add `-UseGeneratedHumanTraffic` only for a synthetic smoke test.

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
- `data\prepared_live\experiments\shortcut_red_flags.csv`
- `data\prepared_live\experiments\entropy_variant_comparison.csv`

## Standalone admin panel

After models are trained, run:

```bat
lab\scripts\start_admin_panel.bat
```

Open:

```text
http://127.0.0.1:8040/
```

The panel is separate from training. It loads existing bundles from `data\prepared_live\models\`, lets you activate a model, watches the live Nginx access log, and includes buttons for running controlled bot-family tests against the local website.

To clear the panel sessions and bot-run history:

```bat
lab\scripts\reset_live_logs.bat
```

You can also press **Clear Live Sessions** inside the admin panel. The reset keeps `data\human_session_archives\` safe, but clears live logs, live labels, and `data\admin_bot_runs\`.
