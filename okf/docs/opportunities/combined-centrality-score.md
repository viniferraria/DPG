---
type: Opportunity
title: Combined centrality score
description: Combine LRC, betweenness, closeness and out-degree into one node score instead of ranking by LRC alone; a correlation check on 2026-10-15 shows the metrics form three near-redundant groups, so only a cross-group combination can add information; untested for precision and recall, not in any runner.
resource: https://github.com/viniferraria/DPG/blob/main/experiments/causal_synthetic_scenarios/run_dpg_causal_synthetic.py
tags: [opportunity, scoring, centrality, composite, correlation]
generated: { by: claude_code/claude-opus-5-5, at: 2026-10-15T00:00:00Z }
status: proposed
---

# The idea

The reference ranks predicate nodes by one metric at a time (`METRICS` in
`run_dpg_causal_synthetic.py`). The idea is that a node that scores high on several metrics at once
(high out-degree, high betweenness, high LRC or closeness) is more likely to be important, possibly causal,
than a node that is high on LRC alone. A combined score would rank nodes by, for example, the mean of
their within-graph ranks on the chosen metrics. It would be scored like any metric in `METRICS`, through
`extract_top_k_features` and `evaluate_causal_accuracy`.

## Example

In one graph, node `F1 <= 0.4` ranks 2nd by LRC, 5th by betweenness and 1st by out-degree, for a mean rank
of 2.7. Node `F3 > 0.9` ranks 1st by LRC, 40th by betweenness and 30th by out-degree, for 23.7. The
combined score puts `F1 <= 0.4` first, while LRC alone puts `F3 > 0.9` first.

# Evidence so far

## How much the metrics agree (2026-10-15)

The script is in the scratchpad, not the repo:
`/private/tmp/claude-502/-Users-vinicius-Projects-DPG/5a616ab8-3204-4917-ac36-d2e9741001fb/scratchpad/metric_correlation.py`.
Run it from the repo root with `uv run python <path>`. It takes the Spearman correlation between metrics
over the non-sink nodes of each graph and averages over graphs. Inputs:

* causal synthetic, `results/node_metrics_2026-10-15T05-08-31.csv`: 16 graphs (4 scenarios, RandomForest
  and ExtraTrees, 2 splits);
* MONK's, `results/node_metrics_2026-10-15T05-17-45.csv`: 4 graphs.

Causal synthetic:

| | outdeg | betw | lrc | clos | harm | ci | perc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outdeg | 1.00 | 0.47 | 0.56 | 0.57 | 0.59 | 0.73 | 0.48 |
| betw | 0.47 | 1.00 | 0.59 | 0.53 | 0.56 | 0.38 | 0.98 |
| lrc | 0.56 | 0.59 | 1.00 | 0.96 | 0.98 | 0.57 | 0.60 |
| clos | 0.57 | 0.53 | 0.96 | 1.00 | 0.99 | 0.59 | 0.54 |
| harm | 0.59 | 0.56 | 0.98 | 0.99 | 1.00 | 0.60 | 0.57 |
| ci | 0.73 | 0.38 | 0.57 | 0.59 | 0.60 | 1.00 | 0.38 |
| perc | 0.48 | 0.98 | 0.60 | 0.54 | 0.57 | 0.38 | 1.00 |

MONK's:

| | outdeg | betw | lrc | clos | harm | ci | perc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outdeg | 1.00 | 0.61 | 0.55 | 0.66 | 0.64 | -0.15 | 0.61 |
| betw | 0.61 | 1.00 | 0.53 | 0.61 | 0.65 | -0.15 | 0.98 |
| lrc | 0.55 | 0.53 | 1.00 | 0.83 | 0.86 | 0.22 | 0.57 |
| clos | 0.66 | 0.61 | 0.83 | 1.00 | 0.97 | 0.04 | 0.66 |
| harm | 0.64 | 0.65 | 0.86 | 0.97 | 1.00 | 0.09 | 0.69 |
| ci | -0.15 | -0.15 | 0.22 | 0.04 | 0.09 | 1.00 | -0.13 |
| perc | 0.61 | 0.98 | 0.57 | 0.66 | 0.69 | -0.13 | 1.00 |

The metrics fall into three groups:

| Group | Metrics | Correlation within the group |
| --- | --- | --- |
| Reach | LRC, closeness, harmonic | 0.96-0.99 synthetic, 0.83-0.97 MONK's |
| Flow | betweenness, percolation | 0.98 on both |
| Local | out-degree, collective influence | 0.73 synthetic; -0.15 on MONK's |

Correlation between groups is about 0.4-0.65.

## What follows

* Combining metrics from one group (LRC with closeness or harmonic, betweenness with percolation) cannot
  add information: they rank nodes almost identically.
* A combination should take one metric per group, for example LRC + betweenness + out-degree. These
  disagree enough that the combination ranks nodes differently from any one of them.
* Whether that difference raises precision and recall is untested.

# Caveats

* **Not causal.** All of these metrics measure a node's position in the graph, which comes from the
  order in which impurity-based splitting chose features. A proxy or a confounder that the trees split
  on early scores like a cause. A combined score can make predictive importance more stable but cannot
  separate cause from correlate.
* **Many-node bias.** Out-degree counts distinct successors, which grows with the number of thresholds a
  feature has. [Per-feature score aggregation](per-feature-score-aggregation.md) already found
  features with many nodes favoured (V22 with 507 nodes against true feature V8 with 102 on one ACIC
  dataset). Out-degree may make this worse.
* **Scale.** Raw metric values have different scales, so the combination has to use within-graph ranks
  or percentiles. That is a second untested choice.

# Open test

On the same 2026-10-15 causal-synthetic and MONK's runs, add the combined score (mean within-graph rank of
LRC, betweenness and out-degree) as one more metric, score it through the reference
`extract_top_k_features` and `evaluate_causal_accuracy` with `TOP_K` (MONK's de-duplicating as its runner
does), and report precision and recall beside each single metric.

# See also

* [Role-guided node ablation](role-guided-node-ablation.md) - Uses the same metric groups to label nodes.
* [Feature drop and relearn](feature-drop-and-relearn.md) - The feature-level, refitting counterpart.
