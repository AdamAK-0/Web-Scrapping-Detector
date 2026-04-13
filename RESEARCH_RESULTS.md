# Research Results

This document summarizes the current state of the implementation against the proposal in
`web_scraping_detection_research_proposal.docx`.

## Proposal alignment

The codebase now includes:

- website-graph and prefix-based feature extraction
- multiple entropy variants plus Navigation Entropy Score v2
- graph/timing/session features with entropy slopes and concentration signals
- early-detection experiments on partial sessions
- heuristic, static-feature, and full-session baselines
- split leakage and shortcut auditing
- leave-one-bot-family-out evaluation
- time-based and leave-one-human-user-out evaluation
- richer controlled bot-generation modes for the local lab, including browser-style families

## Main artifact locations

- Live demo experiment report: `data/nginx_demo_prepared/experiments/summary.md`
- Stronger synthetic experiment report: `data/synthetic_research_v3/experiments/summary.md`
- Live demo leaderboard: `data/nginx_demo_prepared/experiments/leaderboard.csv`
- Synthetic leaderboard: `data/synthetic_research_v3/experiments/leaderboard.csv`

## Key findings

### 1. The live demo dataset is still too easy

The leakage-aware split removed exact full-session path duplication across train, validation, and test, but the live dataset still produces near-perfect results across most models and baselines.

Observed in `data/nginx_demo_prepared/experiments`:

- `all_features` logistic regression: final F1 `1.000`
- heuristic baseline: final F1 `0.923`
- static baseline: final F1 `1.000`
- mean first bot detection: about `3` requests

Interpretation:

- the pipeline works end to end
- the current live demo labels and traffic patterns are still too structured
- prefix-level overlap remains even after grouping exact path signatures, so the demo data is not yet strong enough for final thesis claims

### 2. The stronger synthetic evaluation gives a more useful early-detection result

Observed in `data/synthetic_research_v3/experiments` with grouped session splitting:

- `all_features` logistic regression:
  - prefix 3 F1 `0.857`
  - prefix 5 F1 `0.920`
  - prefix 10 F1 `1.000`
  - final F1 `1.000`
  - mean first bot detection prefix `3.46`

- `graph_plus_entropy` logistic regression:
  - final F1 `0.966`
  - mean first bot detection prefix `3.64`

- `entropy_only` logistic regression:
  - final F1 `0.929`
  - mean first bot detection prefix `4.05`

- heuristic baseline:
  - final F1 `1.000`
  - mean first bot detection prefix `3.00`

- full-session baseline:
  - final F1 `1.000`
  - mean first bot detection prefix about `22.92`

Interpretation:

- the early-detection claim is supported in the controlled setting
- entropy contributes useful signal, but entropy alone detects later than the stronger combined model
- the operational advantage of the prefix model is clear: it reaches strong detection around request 3 to 5, while the full-session baseline only fires at session completion

### 3. Generalization is mixed across unseen bot families

Leave-one-bot-family-out results on `data/synthetic_research_v3/experiments` show:

- strong transfer for `bfs`, `dfs`, `product_focus`, and `stealth_revisit`
- complete failure for held-out `linear` and `article_focus` families in the current setup

Interpretation:

- the current feature set is not yet universally robust to every unseen scraper style
- the project has moved beyond a toy proof-of-concept because the harder evaluation now exposes real failure modes

## Current research conclusion

The implementation now matches the proposal much more closely and supports a defensible prototype claim:

- early detection from partial sessions is feasible
- graph/timing/entropy features can detect bots within the first few requests in a controlled environment
- exact-path leakage is now reduced by grouped splitting and is explicitly audited
- the detector still needs stronger human diversity and harder unseen-bot coverage before making broad final claims

## Highest-value next research step

Collect a larger live dataset with:

- 40 to 100 human sessions from multiple people and browsers
- the new browser-style bot families, especially `browser_hybrid`, `browser_noise`, `playwright`, and `selenium`
- more varied session goals and backtracking behavior

Then rerun:

- `lab\scripts\prepare_live_dataset.bat`
- `lab\scripts\run_research_pipeline.bat`

That is the next step most likely to turn this from a strong research prototype into a stronger final thesis result.
