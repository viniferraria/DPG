---
type: Opportunity
title: Per-feature score aggregation (sum, mean, max)
description: Combine a feature's node scores by sum, mean or max instead of the reference's top-k node labels; partly tested, with one smoke dataset suggesting that sum favours features with many nodes.
resource: https://github.com/viniferraria/DPG/blob/main/experiments/causal_synthetic_scenarios/run_dpg_causal_synthetic.py
tags: [opportunity, scoring, aggregation, centrality, acic2019, compas, monks]
generated: { by: claude_code/claude-sonnet-5-5, at: 2026-10-08T00:00:00Z }
status: proposed
---

# The idea

The reference ranks predicate **nodes** by a metric, takes the top `TOP_K` node labels and maps each to its
feature (`extract_top_k_features`, `drop_duplicates=False`). The alternative is to first combine each
feature's node scores into one feature score (sum, mean or max over its nodes) and rank the features.

Not in any runner. The ACIC and COMPAS runners briefly used summing and were rebased onto the reference
(see [ACIC 2019](/experiments/acic2019.md) and [COMPAS](/experiments/compas.md)).

# Evidence so far

## Spam smoke (ACIC DGP 39, one dataset)

Script (scratchpad, not in the repo):
`/private/tmp/claude-502/-Users-vinicius-Projects-DPG/62293039-6f3d-4aef-bc1f-7ba068d47813/scratchpad/acic_smoke39/nodecount.py`,
run from the repo root with `uv run python <path>` on the node metrics of files `low2007.csv..low2198.csv`
(n = 10,000, 20 merged files). Its output on 2026-10-08:

* Node counts and distinct values: V22 has 507 nodes and 2010 distinct values; V8 (a true feature) has 102
  nodes and 310 distinct values.
* Harmonic centrality, computed from the same node-metrics CSV: V8's best node is 0.118 against V22's
  0.082, and its median node 0.062 against 0.028. Summed, V8 gets 6.43 and V22 gets 13.80.
* The script's rank table: summing ranks V22 first on all six metrics. Max ranks V8 first on five of six;
  Collective influence is the exception (V22 first under max, V8 second).
* Under harmonic centrality, V8's per-node scores are higher than V22's, so per-node scores are not
  inflated by node count there; the bias in the sum comes from adding up many nodes. Only harmonic
  centrality was inspected at node level.

One dataset, one DGP. This is a smoke observation, not a result.

## Earlier synthetic analysis

`/private/tmp/claude-502/-Users-vinicius-Projects-DPG/62293039-6f3d-4aef-bc1f-7ba068d47813/scratchpad/m5/out/eval.txt`
reports, over 16 runs (4 synthetic scenarios x 2 models x 2 splits): method `B-in` reached AP 1.000 (SD 0.000)
with top-1 correct in 16 of 16 runs. Per `m5/PREREG.txt` and `m5/compute.py`, `B-in` is the unweighted
in-degree of each node (networkx `in_degree`, no weight) **summed per feature**. In the same table the
sum variants `C1-IG-sum` and `C1-JSD-sum` also reach AP 1.000, while `C2-max[b]` reaches 0.838 and
`B-lrc` 0.493.

**Hypothesis, untested:** this success may partly reflect the causal features also having the most nodes
in those graphs, so a node-count effect would help the sum there and hurt it where a non-causal feature
has the most nodes (as V22 does in the spam smoke). Nothing read so far checks this.

# Open test

Compare three scorings on the same runs: sum, max, and the reference (top-k node labels), across MONK's,
COMPAS, ACIC (DGP 64 and 39) and the synthetic scenarios. Use the reference's `METRICS`, `TOP_K`,
precision and recall; report per suite and per metric. Also record each true feature's node count, so the
node-count hypothesis can be checked directly. Do this in a scratchpad or a separate analysis script
reading the runners' `node_metrics_*.csv` files; do not change a runner until the user approves.
