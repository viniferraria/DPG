---
type: Side Effect
title: Betweenness and percolation were inverted before 2026-10-07
description: Betweenness and percolation centrality treated the edge weight (a transition frequency) as a path length, so shortest paths ran through rare transitions; results produced before 2026-10-07 are inverted.
resource: https://github.com/viniferraria/DPG/blob/main/metrics/backends/igraph_backend.py
tags: [metrics, betweenness, percolation, backends, regression]
generated: { by: claude_code/claude-sonnet-5, at: 2026-10-07T00:00:00Z }
status: stable
---

# What changed

A DPG edge `weight` is a transition **frequency**: bigger means a stronger link. Shortest-path metrics
need a **length**. Closeness, harmonic and local reaching centrality already used
`distance = W / w` (`W` = total weight); betweenness and percolation passed the raw `weight` instead,
so their shortest paths went through the *rarest* transitions. All three backends were affected
(`networkx`, `igraph`, `graph_tool`). Since 2026-10-07 every shortest-path metric uses `W / w`; see
[metrics backends](/modules/metrics-backends.md).

# Example

Graph `root -> frequent (w=100) -> Class 1 (w=100)` and `root -> rare (w=1) -> Class 1 (w=1)`
(networkx and igraph backends, executed):

| Node | Betweenness before | Betweenness after | Percolation before | Percolation after |
| --- | --- | --- | --- | --- |
| `frequent` | 0.0 | 0.167 | 0.0 | 0.249 |
| `rare` | 0.167 | 0.0 | 0.167 | 0.0 |

Local reaching centrality was never affected (`frequent` 0.66, `rare` 0.007 both before and after),
which is how the two metrics ended up contradicting each other.

# Impact

Any betweenness or percolation value produced before 2026-10-07 is inverted: this includes
`experiments/*/results` CSVs and any `DPGExplanation.node_metrics` saved earlier, plus
`mean_bc` and `sample_bc_weights` derived from them. Re-run to regenerate. Ground-truth fixtures in
`tests/metrics_ground_truth/` were regenerated (betweenness and percolation columns; harmonic changed
only in the last float digit).

# Parity detail

The networkx backend computes betweenness and percolation on a graph whose `dist` values are exact
`Fraction`s, because float sums of `W / w` break mathematically tied shortest paths that igraph's
epsilon comparison keeps tied.
