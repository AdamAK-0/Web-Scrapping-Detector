# Real-Time Web Scraping Detection Research Prototype

This repository is a research-oriented implementation for **real-time web scraping detection** using:

- website traversal graphs
- incremental session-level features
- a proposed **Navigation Entropy Score**
- early bot detection from **partial sessions**
- real log ingestion and online scoring

The codebase is designed to support a **research workflow** rather than only a toy demo. It now includes:

- a reproducible synthetic data generator for humans and multiple scraper strategies
- export of **Nginx combined logs** for realistic pipeline validation
- raw log ingestion for **CSV, JSONL, and Nginx combined** formats
- request-log sessionization with inactivity-based splitting
- weak-label and manual-label support for building experimental datasets
- graph construction from session transitions
- incremental feature extraction over prefixes of sessions
- baseline model training and evaluation
- early-detection metrics such as **requests-to-detection**
- a **FastAPI** service for live online scoring

## Research positioning

Graph-based bot detection already has prior art, so this prototype is positioned around **early detection from partial sessions**, not merely graph usage by itself. The strongest research angle is:

1. represent a website as a traversal graph,
2. compute graph/entropy features as a session unfolds,
3. classify the session in real time,
4. evaluate both **accuracy** and **detection speed**.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 1) Synthetic research pipeline

Generate synthetic human + bot sessions:

```bash
python -m wsd.train --generate-synthetic --output-dir data/synthetic_run
```

Train and evaluate on the generated dataset:

```bash
python -m wsd.train --data-dir data/synthetic_run --prefixes 3 5 10 15 20
```

This produces:

- `requests.csv`
- `graph_edges.csv`
- `graph_categories.csv`
- `prefix_features.csv`
- per-model metrics CSVs
- detection-delay summaries
- saved `.pkl` model bundles

## 2) Real-log dataset preparation

Prepare a trainable dataset from raw logs:

```bash
python -m wsd.prepare_dataset \
  --input-path path/to/access.log \
  --format nginx_combined \
  --manual-labels path/to/manual_labels.csv \
  --output-dir data/prepared_run
```

Supported formats:

- `csv`
- `jsonl`
- `nginx_combined`
- `auto`

The preparation pipeline exports:

- `normalized_requests.csv` — normalized raw log records
- `requests.csv` — sessionized page-level records for modeling
- `session_summary.csv` — session statistics and proposed labels
- `graph_edges.csv`
- `graph_categories.csv`

### Manual labels

The optional manual-label CSV can contain either:

- `client_key,label`
- `session_id,label`

Examples:

```csv
client_key,label
ip_ua:203.0.113.10|Mozilla/5.0,human
ip_ua:203.0.113.99|python-requests/2.31,bot
```

## 3) Train on the prepared dataset

```bash
python -m wsd.train --data-dir data/prepared_run --prefixes 3 5 10 15 20
```

Saved model bundles appear under:

```text
<data-dir>/models/*.pkl
```

## 4) Online scoring API

Serve a trained model:

```bash
python -m wsd.serve \
  --bundle data/prepared_run/models/logistic_regression_bundle.pkl \
  --graph-dir data/prepared_run
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "live_session_1",
    "path": "/blog/post-1",
    "timestamp": 1712910000.0,
    "user_agent": "python-requests/2.31"
  }'
```

## Project structure

```text
src/wsd/
  config.py              # shared constants and defaults
  types.py               # typed data structures
  graph_builder.py       # graph loading/building helpers
  sessionizer.py         # raw request -> session conversion
  log_parsers.py         # CSV / JSONL / Nginx log normalization
  labeling.py            # weak labels and manual labels
  entropy.py             # entropy utilities
  features.py            # incremental graph + timing feature extraction
  synthetic_data.py      # synthetic site, traffic, and Nginx demo generator
  modeling.py            # training, evaluation, bundle persistence
  prepare_dataset.py     # raw log -> research dataset CLI
  online.py              # stateful online detector
  serve.py               # FastAPI live scoring service
  train.py               # training CLI

tests/
  test_entropy.py
  test_features.py
  test_log_pipeline.py
  test_online.py
```

## Included demo assets

The repository now includes a realistic demo pipeline under:

- `data/nginx_demo_raw/`
- `data/nginx_demo_prepared/`

These are generated from synthetic sessions but exported in **Nginx combined-log style** and then re-ingested through the real dataset preparation pipeline. That makes it easy to validate the full workflow before plugging in real logs.

## Current scope

This version supports a strong **research starter pipeline**:

- ingest logs
- build sessions
- assign manual/weak labels
- construct a navigation graph
- extract online features
- train baseline models
- measure early detection
- score sessions in a live API

## Strong next extensions

- honeypot-trigger features
- JA4 / JA4H or header-based comparison baseline
- sequence model or GNN over traversal prefixes
- drift-aware thresholding for deployment
- evaluation on a real production website dataset
- ablation studies for entropy vs graph vs timing features
