---
type: Python Module
title: dpg.core — DecisionPredicateGraph
description: Core module that replays samples through a tree ensemble, mines a directly-follows graph from the resulting event log, and emits a Graphviz/NetworkX Decision Predicate Graph.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [core, graph, process-mining, ensemble, graphviz, networkx]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`dpg/core.py` owns graph construction and config resolution. It exports
`REGRESSOR_MODELS`, `DecisionPredicateGraph`, `DEFAULT_DPG_CONFIG`, and re-exports `DPGError`
(`__all__ = ["REGRESSOR_MODELS", "DPGError", "DecisionPredicateGraph"]`). It does not walk trees into a graph directly —
it produces a process-mining event log and mines a directly-follows graph from it. See
[/pipeline.md](/pipeline.md).

# API

## Module-level

| Name | Type | Purpose |
|---|---|---|
| `REGRESSOR_MODELS` | `tuple[type, ...]` | `isinstance` target that selects the regressor branch (`Pred <value>` leaves) over the classifier branch (`Class <name>` leaves) in `tracing_ensemble` / `tracing_ensemble_parallel`. Always `(RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor)`, plus `econml.grf.CausalForest` when `econml` imports successfully. See [regressor detection](/conventions/regressor-detection.md). |

## Constructor

`DecisionPredicateGraph(model, feature_names, target_names=None, config_file="config.yaml", dpg_config=None)`

| Param | Default | Purpose |
|---|---|---|
| `model` | — | Tree ensemble; must have `estimators_` or `DPGModelError.invalid_ensemble()` is raised |
| `feature_names` | — | `Sequence[str]`; empty raises `DPGValidationError.empty_feature_names()` |
| `target_names` | `None` | Optional class names used for `Class <name>` leaf labels |
| `config_file` | `"config.yaml"` | YAML path resolved relative to the CWD |
| `dpg_config` | `None` | Explicit dict / OmegaConf `DictConfig`; wins over `config_file` |

Resolved attributes: `perc_var`, `decimal_threshold`, `n_jobs`, `graph_construction_mode`,
`context_order`, `visualization_config`, plus `SUPPORTED_GRAPH_CONSTRUCTION_MODES = {"aggregated_transitions",
"execution_trace"}`. The model is passed through `SklearnEnsembleNormalizer.normalize(...)` before
being stored — see [/modules/dpg-sklearn-normalizer.md](/modules/dpg-sklearn-normalizer.md).
Details in [/conventions/config-resolution.md](/conventions/config-resolution.md).

Before normalization, if `model` is an `econml.grf.CausalForest` (only checked when `econml` is
importable), `__init__` reads `model.n_relevant_outputs_` and raises
`DPGModelError.multi_output_causal_forest(n)` unless it equals `1`. See
[causal forest support](/concepts/causal-forest-support.md).

**New in 0.3.0**, both read from `dpg_config["dpg"]["graph_construction"]`:

- `context_order` (default `1`) — DPG-k. `1` is the pre-0.3.0 graph. An explicit `int > 1` or the
  string `"auto"` requires `graph_construction_mode == "execution_trace"`; using either under
  `"aggregated_transitions"` raises a plain `DPGError` in `__init__` (not a `DPGConfigurationError`
  factory — `raise DPGError("context_order='auto' or context_order > 1 requires mode='execution_trace'")`).
  A non-`"auto"` value that is a `bool`, not an `int`, or `<= 0` raises `DPGError("context_order must
  be a positive integer or 'auto'")`. Both validation blocks are duplicated verbatim in `__init__`
  (harmless but literally repeated code, `core.py:183-199` and `:200-216`). See
  [/side-effects/context-order-validation-errors.md](/side-effects/context-order-validation-errors.md).
- `decimal_threshold` also now accepts the literal string `"auto"` (in addition to a non-negative
  `int`); anything else raises `DPGError("decimal_threshold must be a non-negative integer or 'auto'")`.
  See "`decimal_threshold="auto"`" under Behavior.

  Before normalization, if `model` is an `econml.grf.CausalForest` (only checked when `econml` is
importable), `__init__` reads `model.n_relevant_outputs_` and raises
`DPGModelError.multi_output_causal_forest(n)` unless it equals `1`. See
[causal forest support](/concepts/causal-forest-support.md).

## Public methods

| Name | Signature | Produces |
|---|---|---|
| `fit` | `fit(X_train) -> DecisionPredicateGraph` | **Signature changed.** Runs the whole pipeline, calls `build_graph` (storing the result on `self._graph`/`self._nodes_list`), and returns `self` — it no longer renders DOT. Read the graph with `to_networkx()`, render it with `to_dot()`. |
| `to_dot` | `to_dot() -> graphviz.Digraph` | **New.** Renders the fitted graph via `generate_dot`, freshly on every call — deliberately uncached, since the visualizer recolors a dot it is given in place. Raises `DPGNotFittedError` (`DPGNotFittedError.for_builder()`) if called before `fit`. |
| `tracing_ensemble` | `tracing_ensemble(case_id, sample) -> Generator[list[str]]` | Yields `[prefix, event]` pairs (sequential path, used when `n_jobs == 1`) |
| `tracing_ensemble_parallel` | `tracing_ensemble_parallel(case_id, sample) -> list[list[str]]` | Same content as a materialized list, for joblib workers |
| `filter_log` | `filter_log(log) -> pd.DataFrame` | Drops whole path *variants* below `n_cases * perc_var` |
| `discover_dfg` | `discover_dfg(log) -> dict[tuple[str, str], int]` | Directly-follows counts `{(src_label, dst_label): frequency}` |
| `discover_dfg_execution_trace` | `discover_dfg_execution_trace(log) -> dict[tuple[str, str], int]` | `discover_dfg` then drops individual *edges* below `n_cases * perc_var` |
| `discover_dfg_context` | `discover_dfg_context(log, context_order) -> dict[tuple[Any, Any], int]` | **New in 0.3.0.** Context-aware DFG: for `context_order <= 1` delegates to `discover_dfg_execution_trace`; otherwise keys non-terminal nodes by `("ctx", last-k-labels)` tuples (via `_context_node`) and terminals by `("sink", label)`, then applies the same `perc_var` edge-count filter. |
| `build_graph` | `build_graph(dfg) -> tuple[nx.DiGraph, list[list[str]]]` | **New.** Builds the NetworkX DPG directly from the DFG — no DOT involved. Nodes + weighted edges with sha1-derived ids; node keys can be the `("ctx", ...)`/`("sink", ...)` tuples from `discover_dfg_context` — `_node_id_for_key` hashes `repr(key)` for non-`str` keys. Also records `graph.graph["edge_order"]` (see Behavior). `fit()` calls this and stores the result on `self._graph`/`self._nodes_list`. |
| `generate_dot` | `generate_dot(graph) -> graphviz.Digraph` | **Signature changed** (was `generate_dot(dfg)`). Renders the already-built NetworkX graph (from `build_graph`) to DOT — a display/export step, not part of graph construction. Output is byte-identical to the pre-change DOT. |
| `to_networkx` | `to_networkx(graphviz_graph=None) -> tuple[nx.DiGraph, list[list[str]]]` | **`graphviz_graph` is now optional; no longer the construction path.** Returns the graph `fit()` already built (`self._graph`/`self._nodes_list`) when set — `graphviz_graph` is then ignored, even a stale one. If not fitted, falls back to re-parsing `graphviz_graph.body` text for a dot this instance did not build; if not fitted and `graphviz_graph` is `None`, raises `DPGNotFittedError`. Graph plus `nodes_list` of `[node_id, label]`, sorted by id. Each graph node carries `predicate`, `context`, `context_order` attributes (see Behavior). |

**New in 0.3.0, public accessors** (all read state populated by the last `fit()`):

| Name | Signature | Purpose |
|---|---|---|
| `get_context_order` | `() -> int` | The effective context order from the last `fit`; `1` unless `graph_construction_mode == "execution_trace"` resolved otherwise. |
| `get_context_order_history` | `() -> dict[int, int]` | `{k: violations}` for every order tried — from `resolve_context_order` when `context_order == "auto"`, or from `_local_context_violations` for each `k` up to an explicit order otherwise. |
| `get_node_context` | `(node) -> tuple[str, ...]` | The contextual predicate tuple backing a graph node id; empty tuple for sinks and for any node in a `k=1` graph. |
| `get_node_ids_for_trace` | `(labels: Iterable[str]) -> list[str]` | Maps one executed label sequence to the fitted graph's node ids, honoring the resolved context order (used by `DPGExplainer._trace_tree_path`, see [/modules/dpg-explainer.md](/modules/dpg-explainer.md)). |
| `get_predicate_lrc` | `(graph) -> dict[str, float]` | Sums each node's `nx.local_reaching_centrality` back onto its predicate label — at `k>1` one predicate can occupy several contextual nodes and gets credit for all of them. |
| `get_decimal_threshold` | `() -> int` | The resolved effective precision from the last `fit`/trace extraction; raises `DPGError` if called before any `fit()` when `decimal_threshold == "auto"`. |

**New in 0.2.0** (the bundle predates this; documented here because 0.3.0's `get_predicate_lrc` is
explicitly its `context_order > 1` counterpart): `get_trace_consistent_lrc()` (now emits a
`DeprecationWarning` when `get_context_order() > 1`, directing callers to `get_predicate_lrc` instead —
see [/side-effects/trace-consistent-lrc-deprecation.md](/side-effects/trace-consistent-lrc-deprecation.md)),
`get_trace_consistent_trc()`, `get_trace_signatures()`. All three are populated only in
`"execution_trace"` mode by `_build_trace_artifacts`, from the **unfiltered** raw trace log.

Private helpers: `_extract_trace_log(X_train)` (joblib fan-out + flatten into a
`DataFrame(columns=["case:concept:name", "concept:name"])`) and `_leaf_class_label(tree_index,
tree_, node_index)`.

# Behavior

Stage by stage inside `fit`:

1. `_extract_trace_log` replays every sample down every tree. Case id is `f"sample{case_id}_dt{i}"`,
   so there is one case per (sample, tree). It also resolves `decimal_threshold="auto"` once per
   `fit` via `_resolve_decimal_threshold` — see below.
2. Events use the label contract — `"<feature> <= <threshold>"`, `"<feature> > <threshold>"`,
   `"Class <name>"`, and `"Pred <value>"` for regressors (`round(value, 2)`). Thresholds are rounded
   to `self.get_decimal_threshold()` (the resolved value, not necessarily `self.decimal_threshold`
   when it is `"auto"`). See [/conventions/label-contract.md](/conventions/label-contract.md).

   **Which traversal produces those labels now depends on `graph_construction_mode`** (new in 0.3.0 —
   `tracing_ensemble`/`tracing_ensemble_parallel` pick the extractor per call):
   - `"aggregated_transitions"` → `_trace_tree_labels_legacy`: rounds each `tree_.threshold` to
     `effective_decimal` **before** comparing it to the sample and choosing left/right. Kept
     specifically so the legacy graph's edge weights stay backward-compatible with pre-0.3.0 runs.
   - `"execution_trace"` → `_trace_tree_labels`: gets the branch taken from sklearn's own
     `tree.decision_path(...)`/`tree.apply(...)`, and rounds only when *formatting* the predicate
     label afterward. Rounding can therefore never change which branch was recorded on this path,
     unlike the legacy one.

   Because the two modes now round at different points in the branch decision, the **same model and
   sample can produce a different predicate label** for a threshold-adjacent feature value depending
   on which mode built the graph — see
   [/side-effects/exact-routing-label-shift.md](/side-effects/exact-routing-label-shift.md).
3. Mode branch:
   - `"execution_trace"`: `_build_trace_artifacts(log_df)` populates the trace-consistent LRC/TRC/
     signature caches (unfiltered raw log — see the `get_trace_consistent_lrc` note above), then
     `context_order` is resolved (`resolve_context_order(traces)` if `"auto"`, else validated against
     `_local_context_violations` for every `k` up to it) and stored via `get_context_order`/
     `get_context_order_history`. `discover_dfg_execution_trace(log_df)` runs when the resolved order
     is `1`; `discover_dfg_context(log_df, resolved_order)` runs otherwise (see
     [/side-effects/context-order-pairwise-crash.md](/side-effects/context-order-pairwise-crash.md) —
     a `TypeError` regression on this call that has since been fixed on `feature/first_runs`).
   - Otherwise (`"aggregated_transitions"`): context order is forced to `1` regardless of config,
     `filter_log` runs first **only if `perc_var > 0`**, then `discover_dfg`. See
     [/conventions/graph-construction-modes.md](/conventions/graph-construction-modes.md).
4. `build_graph` iterates `sorted(dfg.items(), key=lambda item: item[1])` (ascending frequency)
   directly off the DFG. Each node is added once with `predicate`/`context`/`context_order`
   attributes; each edge is added with `weight=float(frequency)`. The same iteration order is
   recorded as `graph.graph["edge_order"]` — a list of `(source_id, target_id)` — because
   `nx.DiGraph.edges()` iterates node-major and cannot reproduce the DFG-weight order on its own.
   `generate_dot` reads it back from there so its DOT output stays byte-identical to the pre-`build_graph`
   version. `nodes_list` is returned sorted by node id.
5. `generate_dot` walks `graph.graph.get("edge_order", graph.edges())` (the fallback covers a graph
   with no `edge_order`, e.g. one not produced by `build_graph`), writing each node once — label
   escaped via `_escape_dot_label`, a `dpg_context_order` DOT attribute
   (`str(self.get_context_order())`), and a `tooltip` (contextual predicates joined by `" > "` when
   the node has a non-empty context, else the label itself) — and each edge with
   `label=str(int(weight))` when the weight is integral, else `str(weight)`, `penwidth="1"`,
   `fontsize="18"`.
6. `fit` itself stops after `build_graph` and returns `self`. `to_dot()` renders `self._graph` via
   `generate_dot` afresh on every call and raises `DPGNotFittedError.for_builder()` if `self._graph`
   is still `None`. `to_networkx` returns `self._graph`/`self._nodes_list` directly when `fit()`
   already populated them — the normal case, and `graphviz_graph` is ignored even if passed. Only for
   an unfitted instance does it look at `graphviz_graph`: if `None`, it raises
   `DPGNotFittedError.for_builder()`; otherwise it falls back to parsing `dpg_context_order` back off
   each node line of `graphviz_graph.body` (falling back further to `self.get_context_order()` if even
   that attribute is missing) and setting three attributes on every NetworkX node: `predicate` (the
   label), `context` (the contextual-predicate tuple, `()` for a sink or a `k=1` node), `context_order`
   (the parsed int/float). This is the node metadata read by
   `DecisionPredicateGraph._is_predicate_label`/`get_predicate_lrc` and by `DPGExplainer._get_node_metrics`
   — see [/modules/dpg-explainer.md](/modules/dpg-explainer.md).

## `decimal_threshold="auto"`

`_resolve_decimal_threshold`, called once per `fit()`:

1. Scans every value in `X_train` and takes the maximum number of decimal places actually present
   (`_decimal_places`, via `Decimal(str(value)).as_tuple().exponent`).
2. Resolves to `precision + 1` and caches it (`get_decimal_threshold()` reads this cache; it raises
   `DPGError` if read before any `fit()`).
3. Separately checks every tree's raw `tree_.threshold` values against that data-derived grid
   (`np.isclose(threshold, round(threshold, resolved), ...)`). If any threshold doesn't sit on the
   grid, it emits a `RuntimeWarning` naming the offending feature(s) — **routing is still exact**
   (traversal in `"execution_trace"` mode never uses the rounded value to pick a branch); only the
   label's displayed precision is approximate for those features. See
   [/side-effects/decimal-threshold-auto-warning.md](/side-effects/decimal-threshold-auto-warning.md).

Node id derivation, `_node_id_for_key` (called from `build_graph`; `dpg/explainer.py:_label_to_node_id`
now delegates to it instead of holding a second copy of the formula):

```python
"n" + hashlib.sha1(stable_key.encode()).hexdigest()[:12]
```

Truncating the digest to 12 hex chars still preserves content addressing — the same key maps to the
same id across runs and machines — and the `n` prefix keeps the token a valid DOT identifier, where a
bare leading digit is not.

Labels are escaped for DOT via a local `_escape_dot_label`, which replaces `\`, `"`, `[`, and `]`.

Regressor detection is an `isinstance` check against `REGRESSOR_MODELS`
(`RandomForestRegressor`, `ExtraTreesRegressor`, `AdaBoostRegressor`, `GradientBoostingRegressor`,
plus `econml.grf.CausalForest` when `econml` is importable). For a `CausalForest`, the leaf `Pred`
value (`round(tree_.value[node_index][0][0], 2)`) is the tree's local CATE estimate rather than a
mean target value — same code path, different quantity. See
[regressor detection](/conventions/regressor-detection.md) and
[causal forest support](/concepts/causal-forest-support.md).

# Gotchas

- **Node identity is the predicate text — except sinks, always.** Two predicates collapse into one
  node iff their label strings are byte-identical, so `decimal_threshold` directly controls graph
  merging. `context_order > 1` changes non-terminal identity to the last `context_order` labels
  (`_context_node`'s `("ctx", ...)` branch), but `"Class "`/`"Pred "` terminals always take the
  `("sink", label)` branch regardless of `context_order` — one shared node per outcome. For a
  regressor this means every path ending in the same rounded `Pred <value>` collapses into one sink
  no matter how much context precedes it, which can make unrelated decision paths converge on a
  single terminal node. See
  [/side-effects/regression-sink-collisions.md](/side-effects/regression-sink-collisions.md).
- **`discover_dfg_context` (`context_order > 1`, explicit or `"auto"`-resolved) crashed, now fixed.**
  Commit `6df6e20` ("chore: ruff fixes") replaced a working `zip(nodes, nodes[1:])` with
  `pairwise(nodes, nodes[1:])`, but `itertools.pairwise` (imported `core.py:8`) takes exactly one
  iterable argument — `TypeError: pairwise expected 1 argument, got 2` on every `fit()` that resolved
  to an order above `1` under `"execution_trace"`. fixed on `feature/first_runs`: `core.py:636` now calls
  `pairwise(nodes)` alone. Confirmed with a 3-tree RandomForest on iris and
  `dpg_config={"graph_construction": {"mode": "execution_trace", "context_order": 2}, ...}` — `fit()`
  now completes. `uv run pytest tests/test_dpg_k.py` passes all 7 tests (was 2 failures,
  including `test_execution_trace_graph_preserves_long_case_order`, whose own independent
  `pairwise(seq, seq[1:])` misuse at `test_dpg_k.py:95` was fixed alongside `core.py`). See
  [/side-effects/context-order-pairwise-crash.md](/side-effects/context-order-pairwise-crash.md).
- **DOT is a rendering output, not the construction path.** `build_graph` builds the `nx.DiGraph`
  directly from the DFG; `to_networkx` normally just returns that cached graph. The old text-parsing
  fallback — re-parsing `graphviz_graph.body` (splitting on `"->"` and regexing `label="([^"]*)"`) —
  only runs for a `graphviz.Digraph` this instance did not build itself, so label escaping or
  attribute ordering changes in `generate_dot` can no longer silently break the graph this instance
  produces.
- **`graph.graph["edge_order"]`** is a `list[tuple[str, str]]` of `(source_id, target_id)` pairs in
  DFG weight order, set by `build_graph` and read by `generate_dot` — the one piece of construction
  state carried on the graph object itself rather than on `self`.
- Edge weights are only attached when the label parses as numeric
  (`attr.replace(".", "").isdigit()`); otherwise the edge is added with no `weight`.
- `discover_dfg` raises `DPGGraphError.no_paths(perc_var, decimal_threshold)` when the log has zero
  unique cases — the usual symptom of an over-aggressive `perc_var`.
- The constructor prints its resolved settings to stdout, and `fit` prints model metadata; neither
  uses `logging`.

# Examples

```python
from dpg.core import DecisionPredicateGraph, DEFAULT_DPG_CONFIG

dpg = DecisionPredicateGraph(
    model=clf,
    feature_names=list(feature_names),
    target_names=list(target_names),
    dpg_config=DEFAULT_DPG_CONFIG,   # explicit: never depends on the CWD
)
dpg.fit(X_train)          # returns self
dot = dpg.to_dot()         # graphviz.Digraph, rendered fresh
graph, nodes_list = dpg.to_networkx()
```

Exceptions raised by this module all live in `dpg/exceptions.py` and derive from `DPGError`:
`DPGModelError.invalid_ensemble`, `DPGModelError.multi_output_causal_forest`,
`DPGValidationError.empty_feature_names`,
`DPGConfigurationError.missing_perc_var` / `.missing_decimal_threshold` / `.missing_n_jobs` /
`.unsupported_graph_mode`, `DPGGraphError.no_paths`, and `DPGNotFittedError.for_builder` (raised by
`to_dot()`/`to_networkx()` before `fit`).
