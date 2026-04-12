# Experiment Summary

This report was generated automatically from the research experiment runner.

## Best configuration

- Model: **logistic_regression**
- Feature set: **all_features**
- Tuned threshold: **0.10**
- Final-prefix F1: **1.000**
- Final-prefix ROC-AUC: **1.000**
- Mean first detection prefix for bots: **3.00**

## Leaderboard

| model_name | ablation_name | threshold | final_prefix | accuracy | precision | recall | f1 | roc_auc | pr_auc | bot_detection_rate | mean_first_detected_prefix_bot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | all_features | 0.1000 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 |
| random_forest | all_features | 0.1500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 |
| logistic_regression | graph_plus_entropy | 0.6500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 |
| random_forest | graph_plus_entropy | 0.3500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 |
| logistic_regression | graph_only | 0.6500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 |
| random_forest | graph_only | 0.3000 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 |
| logistic_regression | entropy_only | 0.7500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.6667 |
| random_forest | entropy_only | 0.5500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 |
| logistic_regression | timing_only | 0.1000 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 |
| random_forest | timing_only | 0.0500 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 |

## Confidence intervals for the best configuration

| prefix_len | metric | point_estimate | ci_low | ci_high | n_bootstrap |
| --- | --- | --- | --- | --- | --- |
| 3 | f1 | 0.8000 | 0.6667 | 0.9633 | 50 |
| 3 | roc_auc | 1.0000 | 1.0000 | 1.0000 | 50 |
| 3 | pr_auc | 1.0000 | 1.0000 | 1.0000 | 50 |
| 5 | f1 | 0.7500 | 0.5989 | 0.8947 | 50 |
| 5 | roc_auc | 1.0000 | 1.0000 | 1.0000 | 50 |
| 5 | pr_auc | 1.0000 | 1.0000 | 1.0000 | 50 |
| 10 | f1 | 0.9600 | 0.8739 | 1.0000 | 50 |
| 10 | roc_auc | 1.0000 | 1.0000 | 1.0000 | 50 |
| 10 | pr_auc | 1.0000 | 1.0000 | 1.0000 | 50 |
| 15 | f1 | 0.9524 | 0.8421 | 1.0000 | 50 |
| 15 | roc_auc | 1.0000 | 1.0000 | 1.0000 | 50 |
| 15 | pr_auc | 1.0000 | 1.0000 | 1.0000 | 50 |
| 20 | f1 | 1.0000 | 1.0000 | 1.0000 | 50 |
| 20 | roc_auc | 1.0000 | 1.0000 | 1.0000 | 34 |
| 20 | pr_auc | 1.0000 | 1.0000 | 1.0000 | 34 |

## Detection delay summary

| label | num_sessions | detection_rate | mean_first_detected_prefix |
| --- | --- | --- | --- |
| bot | 12 | 1.0000 | 3.0000 |
| human | 12 | 0.0000 | nan |
