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
`DecisionPredicateGraph`, `DEFAULT_DPG_CONFIG`, and re-exports `DPGError`
(`__all__ = ["DPGError", "DecisionPredicateGraph"]`). It does not walk trees into a graph directly —
it produces a process-mining event log and mines a directly-follows graph from it. See
[/pipeline.md](/pipeline.md).

# API

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
`visualization_config`, plus `SUPPORTED_GRAPH_CONSTRUCTION_MODES = {"aggregated_transitions",
"execution_trace"}`. The model is passed through `SklearnEnsembleNormalizer.normalize(...)` before
being stored — see [/modules/dpg-sklearn-normalizer.md](/modules/dpg-sklearn-normalizer.md).
Details in [/conventions/config-resolution.md](/conventions/config-resolution.md).

## Public methods

| Name | Signature | Produces |
|---|---|---|
| `fit` | `fit(X_train) -> graphviz.Digraph` | Runs the whole pipeline and returns the DOT graph |
| `tracing_ensemble` | `tracing_ensemble(case_id, sample) -> Generator[list[str]]` | Yields `[prefix, event]` pairs (sequential path, used when `n_jobs == 1`) |
| `tracing_ensemble_parallel` | `tracing_ensemble_parallel(case_id, sample) -> list[list[str]]` | Same content as a materialized list, for joblib workers |
| `filter_log` | `filter_log(log) -> pd.DataFrame` | Drops whole path *variants* below `n_cases * perc_var` |
| `discover_dfg` | `discover_dfg(log) -> dict[tuple[str, str], int]` | Directly-follows counts `{(src_label, dst_label): frequency}` |
| `discover_dfg_execution_trace` | `discover_dfg_execution_trace(log) -> dict[tuple[str, str], int]` | `discover_dfg` then drops individual *edges* below `n_cases * perc_var` |
| `generate_dot` | `generate_dot(dfg) -> graphviz.Digraph` | Nodes + weighted edges with sha1-derived ids |
| `to_networkx` | `to_networkx(graphviz_graph) -> tuple[nx.DiGraph, list[list[str]]]` | Graph plus `nodes_list` of `[node_id, label]`, sorted by id |

Private helpers: `_extract_trace_log(X_train)` (joblib fan-out + flatten into a
`DataFrame(columns=["case:concept:name", "concept:name"])`) and `_leaf_class_label(tree_index,
tree_, node_index)`.

# Behavior

Stage by stage inside `fit`:

1. `_extract_trace_log` replays every sample down every tree. Case id is `f"sample{case_id}_dt{i}"`,
   so there is one case per (sample, tree).
2. Events use the label contract — `"<feature> <= <threshold>"`, `"<feature> > <threshold>"`,
   `"Class <name>"`, and `"Pred <value>"` for regressors (`round(value, 2)`). Thresholds are
   `round(tree_.threshold[node_index], self.decimal_threshold)`. See
   [/conventions/label-contract.md](/conventions/label-contract.md).
3. Mode branch: `"execution_trace"` calls `discover_dfg_execution_trace(log_df)` on the raw log;
   otherwise `filter_log` runs first **only if `perc_var > 0`**, then `discover_dfg`. See
   [/conventions/graph-construction-modes.md](/conventions/graph-construction-modes.md).
4. `generate_dot` iterates `sorted(dfg.items(), key=lambda item: item[1])` (ascending frequency) and
   writes each edge with `label=str(frequency)`, `penwidth="1"`, `fontsize="18"`.

Node id derivation, verbatim from `generate_dot`:

```python
str(int(hashlib.sha1(activity.encode()).hexdigest(), 16))
```

Labels are escaped for DOT via a local `_escape_dot_label`, which replaces `\`, `"`, `[`, and `]`.

Regressor detection is an `isinstance` check against `RandomForestRegressor`,
`ExtraTreesRegressor`, `AdaBoostRegressor`, `GradientBoostingRegressor`.

# Gotchas

- **Node identity is the predicate text.** Two predicates collapse into one node iff their label
  strings are byte-identical, so `decimal_threshold` directly controls graph merging.
- `to_networkx` re-parses `graphviz_graph.body` *text* (splitting on `"->"` and regexing
  `label="([^"]*)"`). Any change to label escaping or attribute ordering in `generate_dot` can
  silently break node/edge parsing.
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
dot = dpg.fit(X_train)
graph, nodes_list = dpg.to_networkx(dot)
```

Exceptions raised by this module all live in `dpg/exceptions.py` and derive from `DPGError`:
`DPGModelError.invalid_ensemble`, `DPGValidationError.empty_feature_names`,
`DPGConfigurationError.missing_perc_var` / `.missing_decimal_threshold` / `.missing_n_jobs` /
`.unsupported_graph_mode`, and `DPGGraphError.no_paths`.
