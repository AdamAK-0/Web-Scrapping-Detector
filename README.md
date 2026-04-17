# Real-Time Web Scraping Detection Research Prototype

This repository implements a thesis-oriented pipeline for early web-scraping detection using:

- website traversal graphs
- Navigation Entropy features
- prefix-based online classification
- detection-speed metrics such as mean first detected prefix
- real Nginx log ingestion and a local Windows lab

The project is positioned around early detection from partial sessions, not just full-session classification.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

One-command Windows startup:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_project_windows.ps1
```

Useful variants:

```powershell
.\start_project_windows.ps1 -CleanLiveData
.\start_project_windows.ps1 -CleanLiveData -GenerateSampleTraffic
.\start_project_windows.ps1 -CleanLiveData -GenerateSampleTraffic -RunPipeline
```

If `data\human_session_archives\` contains imported real-human logs, `-GenerateSampleTraffic` restores those real sessions automatically and then generates only the bot families. Add `-UseGeneratedHumanTraffic` only when you intentionally want the scripted fake-human generator.

Browser automation traffic support:

- `playwright` mode requires `playwright install`
- Selenium uses the Python Selenium stack and local browser automation

## Core workflow

### 1. Prepare a dataset from logs

```powershell
python -m wsd.prepare_dataset `
  --input-path path\to\access.log `
  --format nginx_combined `
  --manual-labels path\to\manual_labels.csv `
  --output-dir data\prepared_run
```

Outputs:

- `normalized_requests.csv`
- `requests.csv`
- `session_summary.csv`
- `graph_edges.csv`
- `graph_categories.csv`

### 2. Train baseline models

```powershell
python -m wsd.train --data-dir data\prepared_run --prefixes 3 5 10 15 20
```

### 3. Run the thesis experiment suite

```powershell
python -m wsd.experiment `
  --data-dir data\prepared_run `
  --prefixes 3 5 10 15 20 `
  --hard-prefixes 3 5 10 `
  --protocols session_split hard_prefix_session_split leave_one_bot_family_out time_split leave_one_human_user_out `
  --group-key path_signature `
  --n-bootstrap 300 `
  --save-models
```

Key outputs:

- `leaderboard.csv`
- `summary.md`
- `split_summary.csv`
- `leakage_audit.csv`
- `shortcut_audit.csv`
- `shortcut_red_flags.csv`
- `entropy_variant_comparison.csv`
- `predictions_*.csv`
- `metrics_*.csv`
- `detection_delay_*.csv`
- plots for F1/recall by prefix, entropy distributions, score histograms, detection speed, and example score trajectories

If you want a clean sheet for manual session labeling after preparation:

```powershell
python -m wsd.export_label_template `
  --session-summary data\prepared_run\session_summary.csv `
  --output-path data\prepared_run\annotation_template.csv
```

That exports a session-by-session annotation CSV keyed by `session_id`, which is helpful for collecting and curating real human sessions with `participant_id` and notes.

### Import real human sessions from a saved log

After collecting real manual browsing, save the collected Nginx access log as `logs.txt` or pass its path explicitly:

```bat
lab\scripts\reset_live_logs.bat
lab\scripts\import_human_sessions.bat logs.txt
```

The importer archives the raw log under `data\human_session_archives\`, identifies page-level sessions, writes `identified_human_sessions.csv`, and updates `data\live_labels\manual_labels.csv` with `session_id,label=human` rows. The working live log filters common link-preview/crawler agents such as WhatsApp and Google Read Aloud by default while preserving the full raw copy in the archive. After that, run the bot-generation scripts and `prepare_live_dataset.bat` as usual.

## Manual labels format

The label file stays backward compatible with the original format:

```csv
client_key,label
ip_ua:203.0.113.10|Mozilla/5.0,human
ip_ua:203.0.113.99|python-requests/2.31,bot
```

It now also supports optional research metadata:

```csv
client_key,label,participant_id,traffic_family,collection_method,automation_stack,notes
ip_ua:203.0.113.10|Mozilla/5.0,human,p01,human_navigation,manual_browser,,product-comparison task
ip_ua:203.0.113.99|PlaywrightBrowser/1.0,bot,playwright_01,playwright_browser,browser_automation,playwright,randomized waits
```

Useful optional columns:

- `participant_id`
- `traffic_family`
- `collection_method`
- `automation_stack`
- `notes`

These values flow into session summaries and experiment metadata, enabling leave-one-human-user-out evaluation and cleaner provenance analysis.

## Strong next action for thesis quality

The code is now ready for the hardest missing step:

1. collect 40-60 more real human sessions
2. export an annotation template from `session_summary.csv`
3. label those sessions with `participant_id` and `collection_method`
4. rerun `wsd.prepare_dataset` and `wsd.experiment`
5. inspect `shortcut_red_flags.csv` before making strong claims about entropy or graph novelty

## Supported traffic families

Request-based and browser-like families:

- `human`
- `bfs`
- `dfs`
- `linear`
- `stealth`
- `products`
- `articles`
- `revisit`
- `browser_hybrid`
- `browser_noise`
- `playwright`
- `selenium`

## Supported experiment models

- `logistic_regression`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `calibrated_svm`
- `xgboost` when installed
- `lightgbm` when installed
- `catboost` when installed

Use `--model-set` to run only a subset.

## Admin detection panel

After training has produced model bundles in `data\prepared_live\models\`, start the standalone admin panel with:

```bat
lab\scripts\start_admin_panel.bat
```

This starts the local website on `http://127.0.0.1:8039/` and opens the dashboard server on:

```text
http://127.0.0.1:8040/
```

The panel is independent from training. It loads existing `*_bundle.pkl` files, lets you choose the active model and threshold, watches `data\live_logs\access.log`, and scores live sessions as traffic arrives. It also has a controlled test button for launching the existing bot families: `bfs`, `dfs`, `linear`, `stealth`, `products`, `articles`, `revisit`, `browser_hybrid`, `browser_noise`, `playwright`, and `selenium`.

To clear the dashboard back to a clean live-session state, stop Nginx or the panel if needed, then run:

```bat
lab\scripts\reset_live_logs.bat
```

You can also use the **Clear Live Sessions** button inside the admin panel. Both options clear the live access/error logs, manual live labels, and admin bot-run logs. They do not delete archived real-human sessions under `data\human_session_archives\`.

## Windows local lab

The lab serves `website_lab/` through Nginx on `http://127.0.0.1:8039/`.

One-time setup:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\lab\scripts\setup_nginx_windows.ps1
```

Clean start:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\stop_nginx_windows.ps1
lab\scripts\reset_live_logs.bat
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\start_nginx_windows.ps1
```

Generate traffic:

```bat
lab\scripts\generate_human_traffic.bat
lab\scripts\generate_bot_bfs_traffic.bat
lab\scripts\generate_bot_dfs_traffic.bat
lab\scripts\generate_bot_linear_traffic.bat
lab\scripts\generate_bot_stealth_traffic.bat
lab\scripts\generate_bot_products_traffic.bat
lab\scripts\generate_bot_articles_traffic.bat
lab\scripts\generate_bot_revisit_traffic.bat
lab\scripts\generate_bot_browser_hybrid_traffic.bat
lab\scripts\generate_bot_browser_noise_traffic.bat
lab\scripts\generate_bot_playwright_traffic.bat
lab\scripts\generate_bot_selenium_traffic.bat
```

Prepare and run research:

```bat
lab\scripts\prepare_live_dataset.bat
lab\scripts\run_research_pipeline.bat
```

If you want the live training and experiment output captured to a log while it still prints in the console:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File lab\scripts\run_research_pipeline.ps1 -LogPath run_logs.txt
```

Detailed lab instructions are in `LAB_SETUP.md`.

## Project structure

```text
src/wsd/
  config.py
  entropy.py
  experiment.py
  features.py
  labeling.py
  lab_traffic.py
  log_parsers.py
  modeling.py
  online.py
  prepare_dataset.py
  serve.py
  sessionizer.py
  synthetic_data.py
  train.py
```

## Thesis-focused status

The codebase now supports:

- multiple entropy variants and Navigation Entropy Score v2
- grouped session splits by exact path signature
- leave-one-bot-family-out, time-based, and leave-one-human-user-out evaluation
- threshold tuning by prefix
- richer shortcut and leakage audits
- broader tabular model benchmarks
- browser-style bot generation via Playwright, Selenium, and safe emulated browser families

What still depends on real data collection:

- 40-60 more real human browsing sessions
- stronger live evidence against harder browser-like bots
- final thesis-quality results on the expanded live dataset
