---
type: Python Module
title: metrics.graph — GraphMetrics
description: Graph-level DPG analysis — LPA communities, absorbing-Markov-chain clustering toward class nodes, and per-class feature boundary extraction from predicate labels.
resource: https://github.com/viniferraria/DPG/blob/main/metrics/graph.py
tags: [metrics, graph, communities, clustering, markov-chain, boundaries, label-parsing]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`metrics/graph.py` (497 lines) owns everything computed over the DPG *as a whole* rather than
per node or per edge: community detection, probabilistic node-to-class assignment, and the
derivation of human-readable per-class feature intervals ("class bounds") from predicate labels.
It exports the class `GraphMetrics`, re-exported from `metrics/__init__.py` alongside
`NodeMetrics` and `EdgeMetrics`.

This is the module that **parses** the label contract rather than merely carrying it, via
`_parse_predicate` and `_normalize_class_label`.

## Stateless contract (with one caveat)

Every metric entry point is a `@classmethod` or `@staticmethod`; no method reads instance state.
The core three take `(dpg_model: nx.DiGraph, nodes_list: list[list[str]], target_names:
Sequence[str])`, where `nodes_list` is the `[node_id, label]` pair list returned by
`DecisionPredicateGraph.to_networkx` ([/modules/dpg-core.md](/modules/dpg-core.md)).

Caveat worth correcting: unlike `NodeMetrics` and `EdgeMetrics`, `GraphMetrics` **does** define
`__init__(self, target_names: list[str] | None = None) -> None`, which sets `self.target_names`.
Nothing in the module ever reads that attribute — instantiation is vestigial and all real calls
are made on the class. It also carries one class constant,
`COMMUNITY_BOUNDARY_THRESHOLD = 0.2`.

## Dependency direction

`metrics/` is imported by `dpg/` — never the reverse. This module imports only stdlib, `networkx`,
`numpy`, `pandas`, and `joblib`; it has no `dpg` import at all (not even a lazy one, unlike
`metrics/nodes.py`).

# API

| Name | Signature | Returns | What it computes |
|---|---|---|---|
| `extract_graph_metrics` | `(cls, dpg_model, nodes_list, target_names: Sequence[str]) -> dict` | `{"Communities", "Class Bounds"}` | Backwards-compatible alias; delegates verbatim to `extract_graph_metrics_lpa` |
| `extract_graph_metrics_lpa` | `(cls, dpg_model, nodes_list, target_names: Sequence[str]) -> dict` | Same dict | `nx.community.asyn_lpa_communities(weight='weight')` for communities; per-terminal-node reverse `nx.descendants` for bounds, then `calculate_boundaries` |
| `extract_class_boundaries` | `(cls, dpg_model, nodes_list, target_names: Sequence[str]) -> dict` | `{"Class Bounds": {...}}` only | Community/cluster-based bounds via `clustering(...)` at `COMMUNITY_BOUNDARY_THRESHOLD`; returns `{"Class Bounds": {}}` when no `Class ` node exists |
| `extract_communities` | `(cls, dpg_model, df_node_metrics: pd.DataFrame, nodes_list, threshold_clusters: float = 0.2) -> dict` | `{"Clusters", "Probability", "Confidence Interval"}` | Runs `clustering` then relabels ids through `df_node_metrics`'s `Node`→`Label` map. **Takes a node-metrics frame, not `target_names`** |
| `clustering` | `(cls, dpg_model, class_nodes: dict[str,str], threshold: float \| None = None) -> tuple[dict[str,list[str]], dict[str,Any], dict[str,Any]]` | `(clusters, node_probs, confidence)` | Absorbing Markov chain — see Behavior |
| `calculate_boundaries` | `(cls, class_dict: dict, class_names: Sequence[str]) -> dict` | `{class_key: [boundary strings]}` | Fans `calculate_class_boundaries` out over `joblib.Parallel(n_jobs=-1)` |
| `calculate_class_boundaries` | `(key: str, nodes: list[str], class_names: list[str]) -> tuple` (`@staticmethod`) | `(str(key), boundaries)` | Per feature, tracks `min` of `>` thresholds and `max` of `<=` thresholds, emitting `f <= u`, `f > l`, or `l < f <= u` |
| `extract_feature_intervals` | `(cls, decisions: Iterable[str]) -> tuple[dict[str,int], dict[str,dict[str,float]]]` | `(feature_count, feature_intervals)` | Regex `([a-zA-Z0-9_]+)\s*([<=\|>]+)\s*([-+]?[\d.]+)`; `>` raises `min`, `<=` lowers `max` |
| `create_dataframes` | `(cls, data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]` | `(feature_count_df, feature_intervals_df)` | Per-class counts, and a frame indexed `<feat>_min` / `<feat>_max` |
| `communities_to_csv` | `(communities: dict, file_path: str) -> None` (`@staticmethod`) | `None` | Writes long-format CSV with columns `Section, Key, Value`, coercing `np.generic` to builtins first |
| `_parse_predicate` | `(label: str) -> tuple[str, str, float] \| None` (`@staticmethod`) | `(feature, operator, threshold)` or `None` | Splits a split-node label; see Label shapes |
| `_normalize_class_label` | `(label: str) -> str` (`@staticmethod`) | Class name without prefix | Strips a single leading `"Class "`; returns the input unchanged otherwise |

# Behavior

## Label shapes the parsers expect

`_parse_predicate` matches
`^\s*(.+?)\s*(<=|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$` — i.e. exactly the two split-node
forms of the contract, `"<feature> <= <threshold>"` and `"<feature> > <threshold>"`. The feature
group is non-greedy and stripped, so feature names containing spaces and parentheses (`"petal
length (cm)"`) parse correctly; scientific notation in the threshold is accepted. Anything else —
including `"Class 0"` and `"Pred 1.5"` — returns `None` and is skipped by callers.

`_normalize_class_label` handles the classifier terminal form: `"Class 0"` → `"0"`, using
`replace("Class ", "", 1)` so only the first occurrence is removed. It does **not** understand the
regressor form `"Pred <value>"`, which passes through untouched. Full contract in
[/conventions/label-contract.md](/conventions/label-contract.md).

The two older helpers use looser parsing: `calculate_class_boundaries` does
`re.split(' <= | > ', node)` and infers the operator with `'>' in node` (a substring test on the
whole label), and `extract_feature_intervals` uses a `[a-zA-Z0-9_]+` feature regex.

## `clustering` — absorbing Markov chain

Builds a row-stochastic transition matrix `P` over `dpg_model.nodes()`: class nodes are absorbing
(`P[i,i] = 1.0`), every other node distributes probability over its out-edges proportional to
`weight` (default `1`), and a node with zero outgoing weight becomes self-absorbing. Nodes are
permuted into `transient + absorbing`, giving `Q` and `R`; the fundamental matrix is
`N = np.linalg.solve(I - Q, I)` and absorption probabilities are `B = N @ R`. Per-class
probabilities are the column sums of `B`, `np.round(...)` to 2 decimals. Each node is assigned to
its argmax class; `confidence` is the rounded margin between top and second-best probability.
When `threshold` is not `None`, an `"Ambiguous"` bucket is created and nodes whose top probability
is not `> threshold` land there.

## Two boundary paths

`extract_graph_metrics_lpa` is the legacy route: terminal nodes are those whose label contains
`'Class'` **or** `'Pred'`, ancestors are collected via `nx.descendants(dpg_model.reverse(), ...)`,
and the resulting per-class predicate lists go through the parallel `calculate_boundaries`.
`extract_class_boundaries` is the current route and deliberately bypasses LPA: it clusters, then
buckets thresholds per `(class, feature)` into `gt` / `le` / `all` lists, takes `min(gt)` as lower
and `max(le)` as upper, and falls back to `min(all)`/`max(all)` if the interval inverts. Keys are
re-prefixed to `"Class <name>"`, and when `target_names` is truthy the dict is reordered to follow
it, with any leftovers appended in sorted order.

# Gotchas

- **`GraphMetrics()` is instantiable but pointless.** `self.target_names` is written and never
  read; always call on the class, as `dpg/explainer.py`, `dpg/sklearn_dpg.py`, and the tests do.
- **`extract_communities` breaks the shared signature.** It takes `df_node_metrics` as its second
  argument and `threshold_clusters` instead of `target_names`. Passing a `nodes_list` in that slot
  raises inside `set_index('Node')`.
- **`class_names` / `target_names` are inert in the boundary helpers.**
  `calculate_class_boundaries` accepts `class_names` and never uses it; `calculate_boundaries`
  only forwards it. In `extract_class_boundaries`, `target_names` affects *ordering only*, never
  membership.
- **Terminal detection differs between the two paths.** The LPA path matches the substrings
  `'Class'`/`'Pred'` anywhere in a label — a feature literally named `Class_id` would be
  misclassified as terminal. The clustering path requires `str(label).startswith("Class ")`, so
  **regressor DPGs (`"Pred <value>"` leaves) produce no class nodes and `extract_class_boundaries`
  returns `{"Class Bounds": {}}`**.
- **`calculate_boundaries` forces `n_jobs=-1`,** ignoring the DPG config's `n_jobs`. It spawns
  joblib workers regardless of caller intent, which is wasteful for small graphs and can nest
  badly inside an already-parallel run.
- **Node id lookups are stringly typed.** `extract_class_boundaries` tries
  `node_id_to_label.get(node, node_id_to_label.get(str(node)))` and `extract_graph_metrics_lpa`
  does `node_id_to_label[str(node)]` — the latter will `KeyError` rather than skip on an unmapped
  node. Both build their maps with `if "->" not in node[0]` to exclude edge-shaped entries.
- **LPA is non-deterministic.** `asyn_lpa_communities` is a randomized algorithm; community counts
  and memberships can shift between runs even on a fixed-seed model.
- **`clustering` is dense O(n²) memory** and calls `np.linalg.solve` on a `t × t` matrix — it is
  the scaling limit of this module on large graphs, and it will raise `LinAlgError` if `I - Q` is
  singular.
- **Rounding to 2 decimals happens before assignment,** so class probabilities need not sum to
  exactly `1.0` (the tests allow `abs=0.05` drift) and ties are resolved by `classes` iteration
  order.
