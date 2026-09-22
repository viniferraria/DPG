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
`log_timer`, `calc_node_metrics`. The four centrality calculators and the NetworkX-to-igraph
conversion no longer live here — they moved to `metrics/backends/`, selectable by name; see
[/modules/metrics-backends.md](/modules/metrics-backends.md).

## Stateless contract

`NodeMetrics` has no `__init__` and no instance state. Its single public entry point is a
`@staticmethod`. The real signature is:

```python
NodeMetrics.extract_node_metrics(
    dpg_model: nx.DiGraph,
    nodes_list: list[list[str]],
    trace_lrc_by_label: dict[str, float] | None = None,
    backend: str | GraphBackend = "igraph",
) -> Any
```

Note: it takes no `target_names` parameter — that's only part of the `GraphMetrics` interface
([/modules/metrics-graph.md](/modules/metrics-graph.md)). `nodes_list` is the
`[node_id, label]` pair list returned by `DecisionPredicateGraph.to_networkx`
([/modules/dpg-core.md](/modules/dpg-core.md)). `trace_lrc_by_label`, when given, overrides the
backend-computed local reaching centrality for any label present in the mapping (used for
`context_order > 1` trace-consistent LRC). `backend` selects which `GraphBackend` computes the
four centralities — a name from `metrics.backends.BACKEND_NAMES` (default `"igraph"`) or an
already-instantiated backend object; see [/modules/metrics-backends.md](/modules/metrics-backends.md)
for the registry and the metric contract every backend implements.

## Dependency direction

`metrics/` is imported by `dpg/` — never the reverse for module-level imports. `dpg/explainer.py`
and `dpg/sklearn_dpg.py` both do `from metrics.nodes import NodeMetrics`. The weight-validation
guards that perform a *function-body* import `from dpg.exceptions import DPGMetricError` now live
in the backend modules (`metrics/backends/`), not here — see
[/modules/metrics-backends.md](/modules/metrics-backends.md) — but the same one-directional rule
applies to them.

# API

| Name | Signature | Returns | What it computes |
|---|---|---|---|
| `NodeMetrics.extract_node_metrics` | `(dpg_model: nx.DiGraph, nodes_list: list[list[str]], trace_lrc_by_label: dict[str, float] \| None = None, backend: str \| GraphBackend = "igraph") -> Any` (`@staticmethod`, `@log_timer`) | `pd.DataFrame` | Resolves `backend` (via `metrics.backends.get_backend` when given a string), calls its `node_centralities`, applies any `trace_lrc_by_label` override, and joins with degree and node labels |
| `get_logger` | `(name: str, log_file: str \| None = None) -> logging.Logger` | Logger at `INFO`, console handler always, file handler when `log_file` given | Returns early if the logger already `hasHandlers()` so handlers are never duplicated |
| `log_timer` | `(func: Callable[..., Any]) -> Callable[..., Any]` | Wrapped function | Logs `Execution time for <name>: <secs>.4f seconds`; the wrapper signature is `wrapper(self, *args, **kwargs)` |
| `calc_node_metrics` | `(dpg_model: nx.DiGraph) -> tuple[dict[str,int], dict[str,int], dict[str,int]]` | `(in_nodes, out_nodes, degree)` | `degree[node] = in_degree + out_degree`, computed on the NetworkX graph directly — this stays in `nodes.py`, not in any backend |

The four centrality calculators (`calc_betweenness_centrality`, `calc_local_reaching_centrality`,
`calc_closeness_centrality`, `calc_harmonic_centrality`) and `_nx_to_igraph` moved to
`metrics/backends/igraph_backend.py`; see [/modules/metrics-backends.md](/modules/metrics-backends.md)
for their signatures and the metric contract every backend (including igraph's) must satisfy.

## Output DataFrame

Columns, in order: `Node`, `Degree`, `In degree nodes`, `Out degree nodes`,
`Betweenness centrality`, `Local reaching centrality`, `Closeness centrality`,
`Harmonic centrality`, `Label`. Built by indexing the metric frame on `Node`, indexing
`nodes_list` (as columns `["Node", "Label"]`) on `Node`, then `pd.concat(..., axis=1,
join="inner").reset_index()`.

# Behavior

## Why igraph is still the default backend

Betweenness centrality and local reaching centrality are O(V·E)-ish and run far faster in
igraph's C core than in pure-Python NetworkX, which is why `"igraph"` stays the default
`backend` for `extract_node_metrics` and the default `dpg.metrics.backend` config value. **The
NetworkX-to-igraph conversion plus the centrality calls are the hot path of the whole metrics
layer** — see [/modules/metrics-backends.md](/modules/metrics-backends.md) for where that
conversion (`_nx_to_igraph`) and the calculators now live, and for the `networkx` and
`graph_tool` alternatives selectable via the same `backend` parameter. Each calculator stays
wrapped in `@log_timer`, so an `INFO`-level run prints a per-stage timing breakdown that makes
the hot path visible without a profiler.

## Weight-to-distance inversion

Shortest-path libraries minimize a *cost*, but DPG edge weights are directly-follows
*frequencies* — a heavier edge should be closer, not farther. Every backend therefore derives
`distance = total_weight / weight` per edge before any shortest-path call, while weight
*averages* inside local reaching centrality are taken from the original `weight` attribute. Full
details, including the `tests/test_metrics.py::TestCentralityNetworkXParity` reversal caveat, are
in [/modules/metrics-backends.md](/modules/metrics-backends.md).

## Guards

The three weight-validation guards (`non_positive_lrc_weight`, `non_positive_closeness_weight`,
`non_positive_harmonic_weight`, all `DPGMetricError` raised via a function-body
`from dpg.exceptions import DPGMetricError`) now live inside the backend implementations in
`metrics/backends/`, not in this module — see
[/modules/metrics-backends.md](/modules/metrics-backends.md).

# Gotchas

- **igraph's raw `calc_betweenness_centrality` returns integer keys**, not node ids — it is the
  one calculator in `metrics/backends/igraph_backend.py` not keyed by node id.
  `IGraphBackend.node_centralities` maps it back through `node_ids` before it ever reaches
  `NodeMetrics`, so every `NodeCentralities` dict `extract_node_metrics` sees is keyed by node id
  regardless of backend — see [/modules/metrics-backends.md](/modules/metrics-backends.md).
- **`extract_node_metrics` indexes the four centrality dicts by node id** (`[centrality[n] for n
  in nodes]`), so backend iteration order no longer matters for them. Degree, in-degree, and
  out-degree still come from `list(dict.values())` on `calc_node_metrics`'s output, which does
  depend on `calc_node_metrics` iterating `dpg_model.nodes()` in the same order as the `nodes`
  list built alongside it — both iterate the same NetworkX graph, so this holds today but is
  still an implicit ordering assumption.
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
