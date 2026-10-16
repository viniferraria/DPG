---
type: Opportunity
title: Role-guided node ablation
description: Label each DPG node by its metric profile (hub, choke point, ...) without ground truth, then either score nodes by the metric that fits their role or drop candidate nodes and measure the lost reach to the class sinks; untested, not in any runner.
resource: https://github.com/viniferraria/DPG/blob/main/metrics/nodes.py
tags: [opportunity, scoring, ablation, node-role, centrality]
generated: { by: claude_code/claude-opus-5-5, at: 2026-10-15T00:00:00Z }
status: proposed
---

# The idea

Classify every predicate node by its own metric profile, with no ground truth involved. Then use the
role in one of two ways.

* **(a) Role-percentile score.** Score each node by the metric that fits its role (degree for a hub,
  betweenness for a choke point, LRC for a near-root node), expressed as the node's percentile within its
  own graph so that scores from different metrics can be compared. Rank by that score through the
  reference `extract_top_k_features`.
* **(b) Role-guided ablation.** Remove candidate nodes, starting with high-betweenness choke points,
  recompute a graph-level quantity, and score each node by how much the quantity drops. The proposed
  quantity is reach to the `Class` sinks: the share of traffic (or of root-to-sink paths) that still
  reaches a class after removal. Rank nodes by the drop and score through the reference functions.

Variant (b) is the graph-level analogue of the "drop and relearn" principle in Bénard & Josse,
[Variable importance for causal forests](https://arxiv.org/abs/2308.03369v1) (2023): remove something,
recompute, measure the loss. It removes nodes from a fixed graph and does not refit the forest. The
refitting version is [feature drop and relearn](feature-drop-and-relearn.md).

## Roles

Thresholds are within-graph percentiles, because fixed cutoffs do not carry from a 50-node MONK's graph to
a 500-node ACIC graph. A starting set, with 75th percentile as "high":

| Role | Profile | Expected meaning in a DPG |
| --- | --- | --- |
| Hub | high out-degree | many distinct predicates follow it |
| Choke point | high betweenness | many shortest predicate paths pass through it |
| Near-root | high LRC / closeness | reaches most of the graph |
| Peripheral | none high | late, rarely shared split |

A node can hold several roles. Which metric represents each group comes from the correlation check in
[combined centrality score](combined-centrality-score.md): reach (LRC, closeness, harmonic), flow
(betweenness, percolation) and local (out-degree, collective influence).

## Example

A graph has 200 predicate nodes. `F1 <= 0.4` is at the 95th betweenness percentile, so it is a choke point;
removing it cuts the traffic reaching a `Class` sink from 100% to 70%, a drop of 0.30. `F3 > 0.9` is a hub
at the 90th out-degree percentile, but its successors are reachable by other routes; removing it drops
reach by 0.02. Variant (b) ranks `F1 <= 0.4` first.

# Why

* LRC alone measures one property, reach. The correlation check shows betweenness and out-degree carry
  different information (correlation about 0.4-0.65 with LRC), so a role label uses more of the graph.
* Ablation measures what the graph depends on, not just where a node sits.

# Caveats

* **No "bridge between components".** A DPG is almost always one connected component, with every path
  running from root predicates to the `Class` sinks. High betweenness means a choke point between
  predicates, not a link between two separate components.
* **Shortest paths are not tree paths.** Betweenness counts shortest paths between node pairs. The trees'
  actual decision paths need not be shortest, so a node many traces pass through can have low betweenness.
  Traffic from the trace log (edge weights) is a closer measure of real flow.
* **Comparable scores.** Variant (a) mixes metrics, so it depends on percentile normalization, itself an
  untested choice.
* **Many-node bias.** Features with many thresholds produce more hub-like nodes (see
  [per-feature score aggregation](per-feature-score-aggregation.md)).
* **Cost.** Variant (b) recomputes a graph quantity once per removed node. Reach to the sinks is one
  traversal per removal, cheap next to recomputing all centralities.
* **Not causal.** Like every graph metric, the graph replays the trees' learned routing; a proxy the trees
  rely on is as hard to remove as a cause.

# Evidence so far

Tested on 2026-10-15 in a scratchpad prototype, not in the repo:
`/private/tmp/claude-502/-Users-vinicius-Projects-DPG/5a616ab8-3204-4917-ac36-d2e9741001fb/scratchpad/role_ablation/role_ablation.py`.
Run it from the repo root with `uv run python <path>`; it writes `results.csv`.

* **Inputs.** The 2026-10-15T05-08-31 causal-synthetic run (16 graphs, from the saved `DPGExplainer`
  pickles, no refit) and the 2026-10-15T05-17-45 MONK's run (4 graphs).
* **Scoring.** Every score goes through the reference `extract_top_k_features` and
  `evaluate_causal_accuracy`; MONK's uses `run_monk.evaluate_causal_accuracy`. The existing metrics,
  recomputed this way, match the saved `causal_accuracy` CSVs exactly.
* **Variant (a), `role_pct`:** a node's score is the highest of its within-graph percentiles on LRC,
  betweenness and out-degree.
* **Variant (b1), `ablation_flow`:** the absorbed traffic lost when the node is removed, in an absorbing
  Markov chain over the row-normalized edge weights.
* **Variant (b2), `ablation_struct`:** the fraction of the other predicate nodes that can no longer reach
  any `Class` sink once the node is removed, computed from the dominator tree of the reversed graph.
* **Check.** Both ablation scores use a closed form instead of deleting nodes one at a time. On one
  476-node graph they match literal deletion to within 2.5e-12.

Mean precision / recall:

| Metric | Causal synthetic P | Causal synthetic R | MONK's P | MONK's R |
| --- | --- | --- | --- | --- |
| Local reaching centrality | 0.260 | 0.469 | 0.583 | 0.583 |
| Closeness centrality | 0.281 | 0.563 | 0.750 | 0.750 |
| Harmonic centrality | 0.260 | 0.531 | 0.833 | 0.833 |
| Betweenness centrality | 0.219 | 0.313 | 0.833 | 0.833 |
| Collective influence | 0.292 | 0.375 | 0.667 | 0.667 |
| Percolation centrality | 0.198 | 0.313 | 0.750 | 0.750 |
| role_pct (a) | 0.292 | 0.375 | 0.667 | 0.667 |
| ablation_flow (b1) | 0.396 | 0.531 | 0.583 | 0.583 |
| ablation_struct (b2) | 0.417 | 0.594 | 0.667 | 0.667 |

Causal-synthetic precision by model:

| Metric | ExtraTrees | RandomForest |
| --- | --- | --- |
| Local reaching centrality | 0.417 | 0.104 |
| Collective influence | 0.583 | 0.000 |
| role_pct | 0.542 | 0.042 |
| ablation_flow | 0.667 | 0.125 |
| ablation_struct | 0.604 | 0.229 |

What the results show:

* **Ablation leads on causal synthetic.** Both ablation scores have the highest precision on that suite.
  `ablation_struct` also has the highest recall there.
* **The lead comes from ExtraTrees.** On RandomForest, every metric is near 0 in scenarios 2, 3 and 5. The
  top predicates there are the one-hot categorical noise columns (`F5_*`, `F7_*`, `F8_*`).
* **`role_pct` adds nothing.** Its mean precision and recall equal collective influence's on both suites; per-row
  agreement was not checked.
* **MONK's is not decisive.** With 4 graphs, the ablation scores do not beat betweenness or harmonic.
* **`ablation_flow` is close to traffic.** Its Spearman correlation with weighted in-flow has a mean of
  0.735 over the 16 causal-synthetic graphs.

Limits:

* 16 graphs and no significance test. Differences of a few hundredths are noise.
* Ties are broken by CSV row order, because the reference sort is not stable. `ablation_struct` is 0 for
  many nodes. The number of ties at the top-k boundary was not measured.

## Rejected variant

Choosing, per dataset, the metric that scores best against ground truth is test-set leakage: precision and
recall would be inflated. Role labels must come from the graph alone, as above.

# Open test

In a scratchpad script, on the 2026-10-15 causal-synthetic run (node metrics
`results/node_metrics_2026-10-15T05-08-31.csv`; graphs from the matching `states/*_dpg_explainer_*.pkl`) and
the 2026-10-15T05-17-45 MONK's run:

1. Label roles by within-graph percentile.
2. Variant (a): role-percentile score as one more metric.
3. Variant (b): reach-to-sink drop per removed node as one more metric.
4. Score both through the reference `extract_top_k_features` and `evaluate_causal_accuracy` with
   `TOP_K` (MONK's de-duplicating as its runner does), and report precision and recall beside each metric
   in `METRICS` for the same runs.
