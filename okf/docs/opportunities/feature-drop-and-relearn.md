---
type: Opportunity
title: Feature drop and relearn
description: Following Bénard & Josse (arXiv 2308.03369), retrain the ensemble without one feature, rebuild the DPG, and score the feature by how much the graph or its outputs change; untested, not in any runner.
resource: https://arxiv.org/abs/2308.03369v1
tags: [opportunity, scoring, ablation, drop-and-relearn, causal-forest]
generated: { by: claude_code/claude-opus-5-5, at: 2026-10-15T00:00:00Z }
status: proposed
---

# The idea

Bénard & Josse, [Variable importance for causal forests: breaking down the heterogeneity of treatment
effects](https://arxiv.org/abs/2308.03369v1) (2023), measure a variable's importance by "drop and relearn":
remove the variable, retrain the causal forest, and measure how much the treatment-effect heterogeneity
changes. When the dropped variable is also a confounder, retraining alone is biased; they use local
centering and, when the confounder also drives heterogeneity, a correction term. Only the abstract has
been read so far; the estimator and proofs are in the full paper.

For DPG: for each feature, retrain the ensemble without it, rebuild the DPG, and score the feature by the
change. Candidate measures of change:

* the change in predictions or in DPG `Pred` leaf values (for a `CausalForest` DPG, the change in the
  estimated effect);
* the change in graph structure, such as reach to the sinks or the node metric distribution.

A prediction-loss version on a classifier is ordinary drop-column importance and needs no DPG. The DPG
contribution would be a graph-based measure of change.

# Why

It removes a feature from the model, not a node from a fixed graph, so it measures what the model relies
on rather than where a node sits. It is the closest match to the paper's setting and complements
[role-guided node ablation](role-guided-node-ablation.md), which removes nodes without refitting.

# Caveats

* **Output shape.** The score is per feature, not per node, so it cannot go through
  `extract_top_k_features` unchanged. Ranking features by the score and taking the top `TOP_K` would be a
  deviation that has to be named.
* **Cost.** One refit and one DPG build per feature per split.
* **Correlated features.** When a correlated proxy remains, the refit model routes through it and the
  dropped feature looks unimportant. The paper's corrections target exactly this, and do not carry over to
  DPG on their own.
* **Causal forest `Pred` values.** DPG `Pred` leaves are per-tree `tree_.value`, which is not
  `CausalForest.predict` (see [causal forest support](/concepts/causal-forest-support.md)).

# Open test

On the 2026-10-15 causal-synthetic run, refit each model once per dropped feature with the same seed and
`dpg_config`, rebuild the DPG, compute one graph measure of change, rank features, and report precision
and recall against the same ground truth as the reference rows.
