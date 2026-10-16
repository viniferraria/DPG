---
type: Opportunity
title: Intervention contrast (C2)
description: A per-feature score from an absorbing Markov chain over the DPG, the traffic-weighted difference in class-absorption probabilities between the two sides of a split, summed per feature; tested once on the four original scenarios, slightly below in-degree; not in any runner.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [opportunity, metric, markov-chain, intervention]
generated: { by: claude_code/claude-opus-5-5, at: 2026-10-08T00:00:00Z }
status: proposed
---

# The idea

Ask, for each split: if a sample were forced down the other branch, how much would the predicted class
change?

Treat the DPG as a random walk:

* Transition probabilities `P` are the row-normalized edge weights. `Class` nodes are absorbing. A
  non-class node with no out-edges is a zero row.
* The absorption matrix is `B = (I - Q)^-1 R`, each row renormalized to sum to 1 (rows with no absorption
  mass are dropped). `B[v, c]` is the probability that a walk from node `v` ends in class `c`.
* Pair `f <= t` with `f > t` when both have the same predecessor context (always true at
  `context_order = 1`). Contrast: `TV = 0.5 · Σ_c |B_L[c] - B_R[c]|`.
* Weight by traffic (weighted out-degree of both nodes) and, in the primary variant C2-sum, sum the pairs
  per feature. Variant: max per feature.

## Example

From `x1 <= 0.5` walks end in class 0 with probability 0.9; from `x1 > 0.5`, with probability 0.2. Then
`TV = 0.5 · (|0.9 - 0.2| + |0.1 - 0.8|) = 0.7`. With 1,000 units of traffic the pair scores 700. A noise
split whose two sides both end in class 0 with probability 0.55 scores 0.

# Why

It is the closest graph-only proxy to an intervention on a feature, and it uses the graph's structure
rather than node position alone. It can be computed on DPG-k graphs (`context_order > 1`), where pooled-path
recombination is removed and absorption probabilities should follow the real subtrees more closely.

# Evidence so far

Tested on 2026-10-07 in a scratchpad analysis (not in the repo), on the same 16 runs as
[class-separation gain](class-separation-gain.md). Mean AP of C2-sum: 0.990 on the default graph, 0.990 on
`execution_trace` with `context_order = 1`, and 0.979 with `context_order = "auto"`, versus 1.000 for unweighted
in-degree. DPG-k did not help. The baseline was at the ceiling, so no advantage could be shown.

# Caveats

* Two untested ideas at once: a new metric and a per-feature **sum** (see
  [per-feature score aggregation](per-feature-score-aggregation.md)).
* It is observational: the chain replays the trees' learned routing, so a proxy or an upstream cause that
  the trees split on gets credit like a direct cause.
* Only `Class` sinks are absorbing; on a regression DPG (`Pred` sinks) it needs a different contrast.
* Dead-end nodes left by `perc_var` filtering lose absorption mass (up to about 20% of nodes on
  ExtraTrees DPG-k graphs in the test), which the renormalization hides.

# Open test

As for C1: on the SCM scenarios, a node-level variant ranked like any metric in `METRICS` and the
feature-level sum, each beside the reference precision and recall on the same runs.
