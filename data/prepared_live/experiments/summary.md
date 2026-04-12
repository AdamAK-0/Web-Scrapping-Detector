# Experiment Summary

This report was generated automatically from the research experiment runner.

## Best configuration

- Model: **logistic_regression**
- Feature set: **all_features**
- Tuned threshold: **0.05**
- Final-prefix F1: **1.000**
- Final-prefix ROC-AUC: **nan**
- Mean first detection prefix for bots: **3.00**

## Leaderboard

| model_name | ablation_name | threshold | final_prefix | accuracy | precision | recall | f1 | roc_auc | pr_auc | bot_detection_rate | mean_first_detected_prefix_bot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | all_features | 0.0500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 3.0000 |
| random_forest | all_features | 0.1000 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 3.0000 |
| logistic_regression | graph_plus_entropy | 0.0500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 5.0000 |
| random_forest | graph_plus_entropy | 0.0500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 3.6667 |
| logistic_regression | graph_only | 0.0500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 3.0000 |
| random_forest | graph_only | 0.1000 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 5.3333 |
| logistic_regression | entropy_only | 0.0500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 10.0000 |
| random_forest | entropy_only | 0.1000 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 11.6667 |
| logistic_regression | timing_only | 0.0500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 3.0000 |
| random_forest | timing_only | 0.2000 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan | 1.0000 | 3.0000 |

## Confidence intervals for the best configuration

| prefix_len | metric | point_estimate | ci_low | ci_high | n_bootstrap |
| --- | --- | --- | --- | --- | --- |
| 3 | f1 | 1.0000 | 1.0000 | 1.0000 | 300 |
| 3 | roc_auc | 1.0000 | 1.0000 | 1.0000 | 197 |
| 3 | pr_auc | 1.0000 | 1.0000 | 1.0000 | 197 |
| 5 | f1 | 1.0000 | 1.0000 | 1.0000 | 300 |
| 5 | roc_auc | 1.0000 | 1.0000 | 1.0000 | 197 |
| 5 | pr_auc | 1.0000 | 1.0000 | 1.0000 | 197 |
| 10 | f1 | 1.0000 | 1.0000 | 1.0000 | 300 |
| 10 | roc_auc | nan | nan | nan | 0 |
| 10 | pr_auc | nan | nan | nan | 0 |
| 15 | f1 | 1.0000 | 1.0000 | 1.0000 | 300 |
| 15 | roc_auc | nan | nan | nan | 0 |
| 15 | pr_auc | nan | nan | nan | 0 |
| 20 | f1 | 1.0000 | 1.0000 | 1.0000 | 300 |
| 20 | roc_auc | nan | nan | nan | 0 |
| 20 | pr_auc | nan | nan | nan | 0 |

## Detection delay summary

| label | num_sessions | detection_rate | mean_first_detected_prefix |
| --- | --- | --- | --- |
| bot | 9 | 1.0000 | 3.0000 |
| human | 1 | 0.0000 | nan |
