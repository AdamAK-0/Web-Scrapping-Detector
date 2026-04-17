# Generic Website Detector Track

This folder contains a second, separate model family for generic website experiments.

It does not overwrite or modify the existing live-site thesis models in `data/prepared_live`.

## What This Trains

The generic pipeline trains on normalized graph-navigation features across many simulated website archetypes:

- commerce
- news
- docs
- support
- forum
- catalog
- marketing

It also optionally evaluates on the public Zenodo web-robot session-feature dataset:

<https://zenodo.org/records/3477932>

## Feature Coverage

The generic feature set includes the existing graph/timing/entropy spirit plus these explicitly generic features:

- coverage ratio
- path entropy
- revisit rate
- depth distribution
- branching decision pattern
- inter-hop timing
- entry/exit centrality
- graph distance traveled
- backtrack ratio
- structural role of visited nodes, including hub, leaf, and bridge pages

## Commands

Start the generic admin demo with four local websites:

```bat
generic_models\scripts\start_generic_admin_panel.bat
```

This opens the generic dashboard at:

```text
http://127.0.0.1:8050/
```

The dashboard controls these separate generic test websites:

- Atlas Shop, broad hub-and-spoke catalog: `http://127.0.0.1:8061/`
- Deep Docs, deep tree with long chains: `http://127.0.0.1:8062/`
- News Mesh, dense article mesh: `http://127.0.0.1:8063/`
- Support Funnel, narrow support funnel: `http://127.0.0.1:8064/`

Full generic run:

```bat
generic_models\scripts\train_generic_models.bat
```

Faster smoke run:

```bat
generic_models\scripts\train_generic_models_fast.bat
```

Download public benchmark CSVs only:

```bat
generic_models\scripts\download_public_datasets.bat
```

## Outputs

Full-run outputs are written to:

```text
generic_models/artifacts/
```

Important files:

- `generic_models/artifacts/models/*_generic_bundle.pkl`
- `generic_models/artifacts/reports/generic_leaderboard.csv`
- `generic_models/artifacts/reports/generic_prefix_metrics.csv`
- `generic_models/artifacts/reports/public_zenodo_benchmark.csv`
- `generic_models/artifacts/reports/summary.md`
- `generic_models/live_logs/*.jsonl`
- `generic_models/live_graphs/*/graph_edges.csv`

## Generic Admin Panel

The generic admin panel is independent from the original single-website panel. It loads only:

```text
generic_models/artifacts/models/*_generic_bundle.pkl
```

It scores live sessions from the four generic sites, lets you switch between generic models, launches controlled generic bot families, and includes a clear-live button for resetting the generic dashboard view.

## Important Research Interpretation

These models are more generic than the original live-site models because they train across many randomized site graphs and evaluate with leave-site-out splits.

They are still not proof of universal real-world generalization. For a final thesis claim, treat this as a second generic benchmark track and say:

> The method is designed to be website-agnostic, and this generic track tests that claim across simulated website topologies plus a public session-feature benchmark.
