---
type: Python Module
title: metrics.edges — EdgeMetrics
description: Flattens a DPG's directed edges into a pandas DataFrame carrying each edge's weight plus the resolved source and target predicate labels.
resource: https://github.com/viniferraria/DPG/blob/main/metrics/edges.py
tags: [metrics, edges, weights, networkx, pandas]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`metrics/edges.py` is the smallest module in the metrics layer (55 lines). It owns exactly one
job: turn the edge set of a built DPG into a tabular, label-resolved DataFrame. It exports the
class `EdgeMetrics`, re-exported from `metrics/__init__.py` alongside `NodeMetrics` and
`GraphMetrics`.

It computes no centrality of its own. It reads the `weight` attribute that `generate_dot` /
`to_networkx` placed on each edge (the directly-follows frequency) and pairs it with the human
readable predicate labels so downstream code — `dpg/explainer.py`, `dpg/sklearn_dpg.py`,
`dpg/visualizer.py` — never has to re-resolve node ids by hand.

## Stateless contract

`EdgeMetrics` has no `__init__` and no instance state. Its single public entry point is a
`@staticmethod`. The real signature is:

```python
EdgeMetrics.extract_edge_metrics(dpg_model: nx.DiGraph, nodes_list: list[list[str]]) -> Any
```

Note: it takes **only** `(dpg_model, nodes_list)` — there is no `target_names` parameter here.
`target_names` belongs to the `GraphMetrics` interface
([/modules/metrics-graph.md](/modules/metrics-graph.md)). `nodes_list` is the
`[node_id, label]` pair list returned by `DecisionPredicateGraph.to_networkx`
([/modules/dpg-core.md](/modules/dpg-core.md)).

## Dependency direction

`metrics/` is imported by `dpg/` — never the reverse. This module imports only `typing`,
`networkx`, and `pandas`; it has no `dpg` import at all, not even a lazy one inside a function
body (unlike `metrics/nodes.py`, which lazily imports `dpg.exceptions` for its weight guards).

# API

| Name | Signature | Returns | What it computes |
|---|---|---|---|
| `EdgeMetrics.extract_edge_metrics` | `(dpg_model: nx.DiGraph, nodes_list: list[list[str]]) -> Any` (`@staticmethod`) | `pd.DataFrame`, one row per edge in `dpg_model.edges(data=True)` | Builds `node_id_to_label` from `nodes_list`, then for each edge emits the id pair, the `weight` attribute, and the two resolved labels |

There are no other public members — no classmethods, no module-level helpers, no constants.

## Output DataFrame

Columns, in this exact order:

| Column | Value | Source |
|---|---|---|
| `Edge` | `f"{u}-{v}"` | Composite string id of the edge |
| `Weight` | `data.get("weight", 0)` | Directly-follows frequency; defaults to `0` when the attribute is absent |
| `Node_u_label` | `node_id_to_label.get(u)` | Source predicate text; `None` when `u` is not in `nodes_list` |
| `Node_v_label` | `node_id_to_label.get(v)` | Target predicate text; `None` when `v` is not in `nodes_list` |
| `Source_id` | `u` | Raw NetworkX source id (`sha1`-derived) |
| `Target_id` | `v` | Raw NetworkX target id |

# Behavior

The whole implementation is a single pass. `nodes_list` is inverted once into a
`{node_id: label}` dict for O(1) lookup, edges are iterated with `data=True`, rows are appended
to a Python list, and the list is handed to `pd.DataFrame` with an explicit `columns=` list. No
grouping, no filtering, no sorting — row order is `dpg_model.edges()` order, and edge direction
is preserved (this is a `DiGraph`; `u -> v` is never emitted as `v -> u`).

Because labels are carried through verbatim, the resulting frame is directly filterable on the
label contract, which is how tests and visualizers isolate terminal edges — e.g.
`df[df["Node_v_label"].str.startswith("Class ")]` selects the edges that land on classifier
leaves. See [/conventions/label-contract.md](/conventions/label-contract.md) for the
`"<feature> <= <threshold>"` / `"<feature> > <threshold>"` / `"Class <name>"` / `"Pred <value>"`
shapes.

Real call shape, from `tests/test_metrics.py`:

```python
dot = dpg.fit(X_train)
dpg_graph, nodes_list = dpg.to_networkx(dot)
df = EdgeMetrics.extract_edge_metrics(dpg_graph, nodes_list)
```

On a 5-tree Iris RandomForest (seed `160898`) this yields 51 rows with weights in `[1.0, 70.0]`
summing to `1159.0`; on Wine (seed `42`) 126 rows with weights in `[1.0, 57.0]`.

# Gotchas

- **The docstring overstates what is computed.** It advertises "Edge Load Centrality" and
  "Trophic Differences", but neither is calculated anywhere in the module and neither appears in
  the output columns. Only `Weight` is produced. Do not build on those two names — they are
  vestigial documentation, and `tests/test_metrics.py::TestEdgeMetrics::test_expected_columns`
  asserts the six-column set above exactly.
- **Missing labels become `None`, not an error.** `node_id_to_label.get(u)` returns `None` for
  any id absent from `nodes_list`; unlike `NodeMetrics.extract_node_metrics`, which drops
  unmatched rows through an inner join, this module keeps the row with null labels. A frame with
  `NaN` in `Node_u_label` means `nodes_list` and the graph came from different `to_networkx`
  calls.
- **`Weight` defaults to `0`, not `1`.** An edge stripped of its `weight` attribute silently
  contributes zero, which will skew any downstream sum. Note `metrics/nodes.py::_nx_to_igraph`
  uses a `1.0` default for the same attribute — the two modules disagree.
- **`Edge` is not a stable key across graphs.** It is a string concatenation of two `sha1`-derived
  ids joined by `-`; the ids themselves are hashes of predicate text, so changing
  `decimal_threshold` rewrites every `Edge` value.
- **Return type is `Any`**, not `pd.DataFrame`, so callers get no type checking on the result —
  the same annotation weakness as `NodeMetrics.extract_node_metrics`.
- `nodes_list` may contain entries whose id embeds `"->"`; `metrics/graph.py` explicitly filters
  those out when building its own maps, but this module does not. In practice such rows just add
  unreachable dictionary keys and are harmless.
