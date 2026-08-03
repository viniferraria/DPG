---
type: Python Module
title: metrics.nodes — NodeMetrics
description: Computes per-node degree and centrality metrics for a DPG by converting the NetworkX DiGraph to igraph and returning a labelled pandas DataFrame.
resource: https://github.com/viniferraria/DPG/blob/main/metrics/nodes.py
tags: [metrics, nodes, centrality, igraph, networkx, pandas, performance]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`metrics/nodes.py` owns node-level graph metrics for a built DPG: degree, in/out degree,
betweenness centrality, local reaching centrality, closeness centrality, and harmonic
centrality. It exports the class `NodeMetrics` (re-exported from `metrics/__init__.py`
alongside `EdgeMetrics` and `GraphMetrics`), plus module-level helpers `get_logger`,
`log_timer`, `_nx_to_igraph`, `calc_node_metrics`, `calc_betweenness_centrality`,
`calc_local_reaching_centrality`, `calc_closeness_centrality`, `calc_harmonic_centrality`.

## Stateless contract

`NodeMetrics` has no `__init__` and no instance state. Its single public entry point is a
`@staticmethod`. The real signature is:

```python
NodeMetrics.extract_node_metrics(dpg_model: nx.DiGraph, nodes_list: list[list[str]]) -> Any
```

Note: it takes **only** `(dpg_model, nodes_list)` — there is no `target_names` parameter here.
`target_names` is only part of the `GraphMetrics` interface
([/modules/metrics-graph.md](/modules/metrics-graph.md)). `nodes_list` is the
`[node_id, label]` pair list returned by `DecisionPredicateGraph.to_networkx`
([/modules/dpg-core.md](/modules/dpg-core.md)).

## Dependency direction

`metrics/` is imported by `dpg/` — never the reverse for module-level imports. `dpg/explainer.py`
and `dpg/sklearn_dpg.py` both do `from metrics.nodes import NodeMetrics`. The one exception is
deliberate and local: the three weight-validation guards inside this module perform a
*function-body* import `from dpg.exceptions import DPGMetricError` so the top-level import graph
stays one-directional.

# API

| Name | Signature | Returns | What it computes |
|---|---|---|---|
| `NodeMetrics.extract_node_metrics` | `(dpg_model: nx.DiGraph, nodes_list: list[list[str]]) -> Any` (`@staticmethod`, `@log_timer`) | `pd.DataFrame` | Runs all calculators below and joins them with node labels |
| `get_logger` | `(name: str, log_file: str \| None = None) -> logging.Logger` | Logger at `INFO`, console handler always, file handler when `log_file` given | Returns early if the logger already `hasHandlers()` so handlers are never duplicated |
| `log_timer` | `(func: Callable[..., Any]) -> Callable[..., Any]` | Wrapped function | Logs `Execution time for <name>: <secs>.4f seconds`; the wrapper signature is `wrapper(self, *args, **kwargs)` |
| `_nx_to_igraph` | `(nx_graph: nx.DiGraph) -> tuple[ig.Graph, list[str]]` | `(igraph.Graph, node_ids)` where `node_ids[i]` is the NetworkX id of igraph vertex `i` | Copies `weight` onto `es["weight"]` (default `1.0`) and derives `es["distance"] = total_weight / w`, or `inf` when `w <= 0` |
| `calc_node_metrics` | `(dpg_model: nx.DiGraph) -> tuple[dict[str,int], dict[str,int], dict[str,int]]` | `(in_nodes, out_nodes, degree)` | `degree[node] = in_degree + out_degree`, computed on the NetworkX graph, not igraph |
| `calc_betweenness_centrality` | `(ig_graph: ig.Graph) -> dict[int, float]` | Dict keyed by **igraph vertex index**, not node id | `ig_graph.betweenness(directed=True, normalized=False, weights="weight")` divided by `(n-1)*(n-2)` (or `1.0` when `n <= 2`) to match the NetworkX normalization |
| `calc_local_reaching_centrality` | `(ig_graph: ig.Graph, node_ids: list[str]) -> dict[str, float]` | Dict keyed by node id | Per source, shortest paths on `distance`; for each reachable path sums the original edge weights and divides by path length; the summed averages are divided by `total_weight / num_edges` then by `(n-1)` |
| `calc_closeness_centrality` | `(ig_graph: ig.Graph, node_ids: list[str]) -> dict[str, float]` | Dict keyed by node id | Wasserman–Faust over *reachable* nodes only: `((reachable-1)/totsp) * ((reachable-1)/(n-1))`, else `0.0` |
| `calc_harmonic_centrality` | `(ig_graph: ig.Graph, node_ids: list[str]) -> dict[str, float]` | Dict keyed by node id | `sum(1/d)` over outgoing `distance` lengths that are finite and `> 0`; **not** normalized by `n-1` |

## Output DataFrame

Columns, in order: `Node`, `Degree`, `In degree nodes`, `Out degree nodes`,
`Betweenness centrality`, `Local reaching centrality`, `Closeness centrality`,
`Harmonic centrality`, `Label`. Built by indexing the metric frame on `Node`, indexing
`nodes_list` (as columns `["Node", "Label"]`) on `Node`, then `pd.concat(..., axis=1,
join="inner").reset_index()`.

# Behavior

## Why igraph

`_nx_to_igraph` exists because the two expensive metrics — betweenness centrality and local
reaching centrality — are O(V·E)-ish and run far faster in igraph's C core than in pure-Python
NetworkX. **This conversion plus the centrality calls are the hot path of the whole metrics
layer**; every calculator except `calc_node_metrics` operates on the igraph copy. Each
calculator is wrapped in `@log_timer`, so an `INFO`-level run prints a per-stage timing
breakdown that makes the hot path visible without a profiler.

## Weight-to-distance inversion

igraph shortest paths minimize a *cost*, but DPG edge weights are directly-follows
*frequencies* — a heavier edge should be closer, not farther. `_nx_to_igraph` therefore stores
`distance = total_weight / weight` on every edge and every path computation passes
`weights="distance"`, while weight *averages* inside local reaching centrality are taken from
the original `weight` attribute. `tests/test_metrics.py::TestCentralityNetworkXParity` pins
closeness and harmonic to NetworkX computed on this same inverted metric, including the
reversal (`nx.closeness_centrality` measures incoming distance; this module measures outgoing).

## Guards

`calc_local_reaching_centrality`, `calc_closeness_centrality`, and `calc_harmonic_centrality`
each raise a `DPGMetricError` (`non_positive_lrc_weight`, `non_positive_closeness_weight`,
`non_positive_harmonic_weight`) when `sum(es["weight"]) <= 0`. All three lazily import
`dpg.exceptions` inside the function body. igraph's `Couldn't reach some` `RuntimeWarning` is
suppressed per-call via `warnings.catch_warnings()`.

# Gotchas

- **`calc_betweenness_centrality` returns integer keys.** It is the only calculator not keyed by
  node id; `extract_node_metrics` gets away with `list(...values())` because dict insertion order
  matches `list(dpg_model.nodes())`. Consuming it directly requires mapping through `node_ids`.
- **`extract_node_metrics` assumes iteration-order alignment.** `data_node` is assembled from
  `list(dpg_model.nodes())` and four `list(dict.values())` calls; correctness depends on every
  calculator iterating the graph in the same order that `_nx_to_igraph` captured.
- **The join is `inner`.** A node present in the graph but missing from `nodes_list` is silently
  dropped from the result, and vice versa — a row-count shortfall is a labelling problem, not a
  metrics bug.
- **Harmonic centrality is unnormalized** and scales with `total_weight`, so values are not
  comparable across graphs with different edge counts. Betweenness, closeness, and local
  reaching centrality are all normalized to `[0, 1]`.
- **Self-distance is excluded everywhere**: harmonic skips `d == 0`, closeness counts the source
  in `reachable_nodes` but subtracts it via `reachable_nodes - 1`, and local reaching centrality
  skips paths of length `<= 0`.
- **Node identity is the predicate text** (`sha1` of the label). Two byte-identical labels are
  one node, so `decimal_threshold` changes node counts and therefore every metric in this table.
  See [/conventions/label-contract.md](/conventions/label-contract.md).
- `extract_node_metrics` is annotated `-> Any`, not `-> pd.DataFrame`, despite always returning
  a DataFrame — callers get no type checking on the result.
