---
type: Opportunity
title: Class-separation gain (C1)
description: A per-feature score computed from DPG's trace log, the traffic-weighted information gain of each split on the leaf classes, summed per feature; tested once on the four original scenarios, where it tied in-degree at the ceiling; not in any runner.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [opportunity, metric, information-gain, trace-log]
generated: { by: claude_code/claude-opus-5-5, at: 2026-10-08T00:00:00Z }
status: proposed
---

# The idea

Score each split by how well it separates the classes, using DPG's own replay of the trees rather than
sklearn's impurity.

DPG replays every training sample down every tree and logs each predicate passed and the `Class` leaf
reached (one case per sample × tree; see [pipeline](/pipeline.md)). For a feature `f` and rounded threshold
`t`, count the cases through `f <= t` (left) and `f > t` (right) per class. Then:

* `IG(f, t) = H(L + R) - nL/n · H(L) - nR/n · H(R)`, in bits, with `n = nL + nR`.
* Score per split: `n · IG(f, t)` (traffic-weighted).
* Feature score (primary variant, C1-IG-sum): the sum over `t`. Variants: max over `t`, and Jensen–Shannon
  divergence of the two class distributions in place of `IG`.

## Example

A split `x1 <= 0.5` is passed by 1,000 cases: left 500 cases, 450 of class 0 and 50 of class 1; right 500
cases, 50 of class 0 and 450 of class 1. `H(L+R) = 1.0`, `H(L) = H(R) ≈ 0.469`, so `IG ≈ 0.531` and the
split scores `≈ 531`. A noise split passed by the same 1,000 cases with the same class mix on both sides
scores 0.

# Why

DPG's node centralities describe graph position, not whether a split changes the outcome. C1 measures the
latter directly on the graph's own data.

# Evidence so far

Tested on 2026-10-07 in a scratchpad analysis (not in the repo): the four original causal-synthetic
scenarios × {RandomForest, ExtraTrees} × 2 splits = 16 runs, feature-level average precision (AP).
C1-IG-sum reached AP 1.000 in 16/16 runs, tying unweighted in-degree (also 1.000) and MDI. Its mean Spearman
correlation with in-degree was 0.85. Because the baseline was at the ceiling, the test could not show an
advantage.

# Caveats

* It is two untested ideas at once: a new metric and a per-feature **sum**. The sum may favour features
  with many thresholds; see [per-feature score aggregation](per-feature-score-aggregation.md).
* It is close to an MDI computed on leaf-predicted classes, and may add little beyond sklearn's
  `feature_importances_`.
* It scores features, not nodes, so it does not fit the reference protocol (top-`TOP_K` node labels)
  without a node-level variant (`n · IG` per node).

# Open test

On the SCM scenarios, where the old benchmark's ceiling does not apply, compare against the reference
protocol on the same runs: a node-level variant ranked like any metric in `METRICS`, and the feature-level
sum, each beside the reference precision and recall.
