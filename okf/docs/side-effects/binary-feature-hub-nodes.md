---
type: Side Effect
title: Binary features become hub nodes on RandomForest DPGs
description: A 0/1 column can only be split at 0.5, so every RandomForest split on it merges into one predicate node per side; those hubs top betweenness, collective influence and percolation even when the model ignores the feature, while continuous features fragment into hundreds of low-traffic nodes.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [metrics, node-identity, one-hot, randomforest, label-contract, limitation]
generated: { by: claude_code/claude-opus-5-5, at: 2026-10-15T00:00:00Z }
status: stable
---

# What happens

Node identity is the predicate text (see [label contract](/conventions/label-contract.md)). A binary column
such as a one-hot `F5_A` has one possible threshold, `0.5`, so every RandomForest split on it produces the
same two labels, `F5_A <= 0.5` and `F5_A > 0.5`. All those splits, across every tree, merge into **two hub
nodes**. A continuous feature gets a different threshold, and so a different label, on almost every
split; its traffic spreads over hundreds of nodes.

Node-level metrics read the hub position as importance. Under the reference scoring
(`extract_top_k_features`), the one-hot hubs fill the top `TOP_K` even when the model barely uses them.

ExtraTrees draws random thresholds, and on these datasets a one-hot column still got 125-296 distinct
thresholds, so it fragments like a continuous column and no hubs form.

# Evidence (2026-10-15)

From the causal-synthetic run `2026-10-15T05-08-31`, split 0. Scripts in a scratchpad (not in the repo):
`/private/tmp/claude-502/-Users-vinicius-Projects-DPG/5a616ab8-3204-4917-ac36-d2e9741001fb/scratchpad/rf_noise/`.

## The model does not rely on the one-hot columns

Model refit as the runner builds it: `RandomForestClassifier(random_state=42)` on
`StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)`, split 0. Permutation importance on the
test split, 30 repeats. Scenario 3, test accuracy 0.95:

| Feature | `feature_importances_` | Permutation importance |
| --- | --- | --- |
| F1 (true) | 0.557 | 0.354 |
| F2 (true) | 0.295 | 0.206 |
| F3 / F4 | 0.064 / 0.070 | 0.004 / 0.006 |
| F5_A / F5_B / F5_C | 0.005 / 0.005 / 0.006 | -0.001 / 0.004 / 0.004 |

## The graph turns them into hubs

Scenario 3, from `results/node_metrics_2026-10-15T05-08-31.csv`. The node counts and degrees for the
RandomForest graph were re-checked directly from that CSV:

| Graph | Feature | Predicate nodes | Mean in-degree | Mean out-degree | Max single-node traffic |
| --- | --- | --- | --- | --- | --- |
| RandomForest | F1 | 1034 | 1.6 | 1.9 | 1061 |
| RandomForest | F2 | 1036 | 1.5 | 1.9 | 1058 |
| RandomForest | F5_A / F5_B / F5_C | 2 / 2 / 2 | 119.5 / 110.0 / 108.0 | 152.0 / 134.0 / 138.0 | 8245 / 6857 / 10066 |
| ExtraTrees | F1 | 2095 | | | 438 |
| ExtraTrees | F5_A / F5_B / F5_C | 228 / 247 / 216 | | | 418 / 207 / 372 |

What the numbers show:

* **Concentrated, not larger.** The one-hot columns carry 12-14% of total in-traffic in every
  RandomForest graph. That traffic sits in 2 nodes per column, so a single hub carries 4-7 times the
  traffic of the busiest continuous node (scenarios 3 and 5).
* **Close to the roots.** The hubs sit 1 hop from a root.

## Effect on the reference scores

Share of the top-3 nodes that are one-hot, mean of 2 splits:

| Scenario, model | Betweenness | Collective influence | Percolation | Closeness | Harmonic | LRC |
| --- | --- | --- | --- | --- | --- | --- |
| s2, RandomForest | 1.00 | 1.00 | 1.00 | 0.17 | 0.50 | 0.00 |
| s3, RandomForest | 1.00 | 1.00 | 1.00 | 0.83 | 0.83 | 0.00 |
| s5, RandomForest | 0.83 | 1.00 | 0.83 | 0.33 | 0.67 | 0.00 |
| s3, ExtraTrees | 0.00 | 0.00 | 0.17 | 0.17 | 0.50 | 0.17 |

* **LRC never picks a hub,** but it picks root-adjacent nodes of continuous noise features (F3, F4), so its
  RandomForest precision is still near 0.
* **Mean precision by model.** RandomForest: 0.083 (s1), 0.111 (s2), 0.042 (s3), 0.056 (s5). ExtraTrees:
  0.542, 0.292, 0.542, 0.347.

# What was ruled out

* **The model uses the noise.** Its own importances say the opposite.
* **A scoring or label bug.** `extract_top_k_features` and the metrics behave as designed on this shape of
  graph.
* **A run mismatch.** The pickles were matched to the CSV rows by exact node-id set.

# Open

* **Which property drives each metric.** Traffic, degree and distance to the roots are all high on the
  hubs, so it is not isolated which one each metric responds to.
* **No counterfactual.** The fix that removes the merge (for example, perturbing thresholds) was not
  tested.
* **Not checked on other suites.** The same mechanism likely affects any binary or low-cardinality
  feature, such as COMPAS or ACIC one-hot columns, but this was not checked there.
* **Fixes are opportunities.** Any change to the scoring that corrects for this must go through
  [opportunities](/opportunities/index.md) first.
