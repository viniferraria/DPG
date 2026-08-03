---
type: Data Structure
title: Explanation dataclasses
description: Field-level reference for DPGExplanation, DPGLocalExplanation and DPGTreePathExplanation, the three result containers returned by DPGExplainer.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/explainer.py
tags: [dataclass, explanation, local-explanation, api, serialization]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

Three `@dataclass`es defined at the top of `dpg/explainer.py` carry every result the
[`DPGExplainer`](/modules/dpg-explainer.md) returns. All three are exported from `dpg/__init__.py`.
They are plain containers: no validation, no computed properties, only an `as_dict()` helper each.

# API

## `DPGExplanation` — global result

Returned by `DPGExplainer.explain_global()`.

| Field | Type | Meaning |
|---|---|---|
| `graph` | `Any` (an `nx.DiGraph`) | The fitted graph produced by `builder.to_networkx`. |
| `nodes` | `list[list[str]]` | The `[node_id, label]` pairs from `to_networkx`; the `nodes_list` every `metrics/` API expects. |
| `dot` | `Any` (a `graphviz.Digraph`) | The DOT object from `builder.fit`; what the plot functions render. |
| `node_metrics` | `Any` (a `pd.DataFrame`) | Output of `NodeMetrics.extract_node_metrics`; keyed by a `"Node"` column and containing `"Local reaching centrality"` and `"Betweenness centrality"`. |
| `edge_metrics` | `Any` | Output of `EdgeMetrics.extract_edge_metrics`. |
| `class_boundaries` | `dict[str, Any]` | Output of `GraphMetrics.extract_class_boundaries(graph, nodes, target_names=builder.target_names or [])`. |
| `communities` | `dict[str, Any] \| None` = `None` | Output of `GraphMetrics.extract_communities`, only when `explain_global(communities=True)`. |
| `community_threshold` | `float \| None` = `None` | The threshold actually used, set only when communities were computed (otherwise `None`, even though the parameter defaults to `0.2`). |

`as_dict()` returns all eight fields verbatim (no recursion, no copying).

## `DPGTreePathExplanation` — one traced tree

One instance per estimator in `builder.model.estimators_`, produced by `_trace_tree_path`.

| Field | Type | Meaning |
|---|---|---|
| `tree_index` | `int` | Position in `model.estimators_` (post-normalization). |
| `tree_prefix` | `str` | `f"sample{sample_id}_dt{tree_index}"` — the case-id prefix convention shared with the trace log. |
| `labels` | `list[str]` | Raw DPG labels along the executed path, e.g. `["petal width (cm) <= 0.8", "Class 0"]`. Last element is a leaf (`"Class …"` or `"Pred …"`). |
| `node_ids` | `list[str \| None]` | Graph node ids aligned with `labels`. With `validate_graph=True` these come from the fitted node lookup and are `None` when the label is absent from the graph; with `validate_graph=False` they are the locally recomputed sha1 ids (never `None`). |
| `predicate_truths` | `list[bool]` | One entry per internal split traversed. Every entry is `True` — the trace follows the branch the sample satisfies. Shorter than `labels` by one (no entry for the leaf). |
| `edge_exists` | `list[bool]` | For each consecutive label pair, whether that edge exists in the fitted graph. Length `len(labels) - 1`. |
| `starts_from_root` | `bool` | `len(labels) > 0`. |
| `ends_in_leaf` | `bool` | Last label starts with `"Class "` or `"Pred "`. |
| `graph_path_valid` | `bool` | All path node ids are present in the node-metrics lookup **and** all `edge_exists` are `True`. |
| `mean_lrc` | `float \| None` = `None` | Mean `"Local reaching centrality"` over the path nodes present in the graph; `None` when none are. |
| `mean_bc` | `float \| None` = `None` | Mean `"Betweenness centrality"`, same condition. |
| `path_confidence` | `float \| None` = `None` | `(node_coverage + edge_coverage) / 2`, where `node_coverage` = matched nodes / `len(labels)` and `edge_coverage` = `sum(edge_exists) / len(edge_exists)` (defaults to `1.0` for a single-node path). `0.0` when `labels` is empty. |

`as_dict()` returns all twelve fields.

## `DPGLocalExplanation` — one sample

Returned by `DPGExplainer.explain_local()`.

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | `int` | Identifier passed in by the caller (default `0`). |
| `sample` | `list[float]` | The flattened sample as a Python list (`np.asarray(sample).reshape(-1).tolist()`). |
| `tree_paths` | `list[DPGTreePathExplanation]` | One per estimator, in `estimators_` order. |
| `graph_validated` | `bool` | Echo of the `validate_graph` argument — whether validation was requested, not whether it passed. |
| `all_trees_valid` | `bool` | `all(path.graph_path_valid for path in tree_paths)`. |
| `majority_vote` | `str \| None` | Most common **normalized** class name, or `None` for regressors / when no leaf label starts with `"Class "`. |
| `class_votes` | `dict[str, int]` | Normalized class name → number of trees whose leaf voted for it. |
| `path_mode` | `str` | Always the literal `"execution_trace"` — local tracing never reads paths out of the mined DFG. |
| `sample_confidence` | `dict[str, Any] \| None` = `None` | Diagnostics dict from `_compute_sample_confidence`; see [faithfulness evaluation](/concepts/faithfulness-evaluation.md) for its keys. |

`as_dict()` recurses one level: `tree_paths` becomes `[path.as_dict() for path in self.tree_paths]`.
Everything else (including `sample_confidence`) is passed through by reference.

# Gotchas

## Class-label normalization is split across fields

The [label contract](/conventions/label-contract.md) says leaves are written `"Class 0"`. Only some
fields keep that raw form:

| Holds raw DPG labels (`"Class 0"`) | Holds normalized names (`"0"`) |
|---|---|
| `DPGTreePathExplanation.labels` (including the leaf) | `DPGLocalExplanation.class_votes` keys |
| `local_path_dataframe()["label"]` | `DPGLocalExplanation.majority_vote` |
| | `sample_confidence["class_scores"]`, `["class_support"]`, `["evidence_scores"]`, `["top_competitor_class_pred"]` |

Normalization is `_normalize_class_vote_label`, a plain `label.removeprefix("Class ")`. A separate
helper, `_normalize_prediction_label`, is used in `evaluate_faithfulness` to bring *model*
predictions into the same space: it maps `model.classes_` → `target_names` positionally when both
exist and have equal length, otherwise strips a `"Class "` prefix, and always returns `str`.

## Other notes

- `as_dict()` is the only serialization helper — there is no `to_json`, `from_dict`, or pickling
  hook. `graph`, `dot`, `node_metrics` and `edge_metrics` remain live objects inside the dict, so
  the result is not JSON-serializable as-is.
- The dataclasses are not frozen and have no `__post_init__`; nothing stops a caller from mutating
  a path in place, and nothing recomputes `all_trees_valid` if you do.
- `graph_validated` being `True` says nothing about success — check `all_trees_valid` for that.
- `predicate_truths` is deliberately all-`True`; it records which predicate *text* was taken, not a
  re-evaluation of the condition, so it is not a useful validity signal.

# Examples

```python
local = explainer.explain_local(X[0], sample_id=7)

local.majority_vote          # "0"        (normalized)
local.tree_paths[0].labels[-1]  # "Class 0" (raw DPG label)
local.class_votes            # {"0": 41, "1": 9}

payload = local.as_dict()    # tree_paths already flattened to dicts
payload["path_mode"]         # "execution_trace"
```
