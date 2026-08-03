---
type: Convention
title: Graph construction modes
description: The two supported dpg.graph_construction.mode values — aggregated_transitions (variant-level filtering) and execution_trace (edge-level filtering) — their thresholds, failure mode, and effect on local-explanation validity flags.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [configuration, filtering, dfg, process-mining, local-explanations]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`dpg_config["dpg"]["graph_construction"]["mode"]` selects *what* the `perc_var` threshold prunes:
whole decision-path variants, or individual directly-follows edges. It is read once in
`DecisionPredicateGraph.__init__` into `self.graph_construction_mode`, and branched on in `fit`.
See [/modules/dpg-core.md](/modules/dpg-core.md).

# API

`self.SUPPORTED_GRAPH_CONSTRUCTION_MODES = {"aggregated_transitions", "execution_trace"}`

| Mode | Filter unit | Code path in `fit` |
|---|---|---|
| `"aggregated_transitions"` (default) | Path **variant** (a whole case) | `filter_log(log_df)` if `perc_var > 0`, then `discover_dfg(log_df)` |
| `"execution_trace"` | Individual **edge** | `discover_dfg_execution_trace(log_df)` on the unfiltered log |

Anything else raises at construction time:

```
DPGConfigurationError.unsupported_graph_mode(mode, sorted(SUPPORTED_GRAPH_CONSTRUCTION_MODES))
# -> "Unsupported graph construction mode '<mode>'. Supported modes: aggregated_transitions, execution_trace."
```

`DPGConfigurationError` subclasses `DPGValidationError` → `DPGError` and `ValueError`, so
`except DPGError` and `except ValueError` both catch it. The check happens in `__init__`, before any
tracing, so a typo fails immediately rather than after a long `fit`.

# Behavior

## `aggregated_transitions`

`filter_log` groups the log by `case:concept:name`, keys each case by its variant string
`"|".join(group["concept:name"].values)`, and keeps every case whose variant is frequent enough:

```python
min_count = len(log["case:concept:name"].unique()) * self.perc_var
if len(case_ids) >= min_count:      # per variant
    case_ids_to_keep.update(case_ids)
```

Surviving cases are returned as a copy of the log; `discover_dfg` then counts consecutive event
pairs per case. Because the unit is a full case, a rare path disappears entirely — including any
edge it shared with common paths, unless another kept case reproduces that edge.

Note `fit` calls `filter_log` **only when `perc_var > 0`**; with `perc_var == 0` the raw log goes
straight into `discover_dfg`.

## `execution_trace`

`discover_dfg_execution_trace` builds the full DFG first, then prunes edges:

```python
dfg = self.discover_dfg(log)
if self.perc_var <= 0:
    return dfg
min_count = log["case:concept:name"].nunique() * self.perc_var
return {edge: count for edge, count in dfg.items() if count >= min_count}
```

The threshold formula is the same `total_cases * perc_var`, and `total_cases` is the number of
unique case ids in the **raw** trace log (one case per sample-tree pair). The difference is the unit
being compared against it: variant occurrence count vs. edge frequency.

Both modes can produce an empty graph; `discover_dfg` raises
`DPGGraphError.no_paths(perc_var, decimal_threshold)` when the log contains zero unique cases.

# Gotchas

## Interaction with `explain_local`

`DPGExplainer.explain_local` never uses the built graph to derive the path. It re-traces the sample
through the raw trees and reports `path_mode="execution_trace"` regardless of the construction mode,
then checks the traced path against whatever graph was built:

- `DPGTreePathExplanation.graph_path_valid` is
  `all(node_id in node_metrics_lookup for node_id in native_node_ids) and all(edge_exists)`, where
  `edge_exists[i] = graph.has_edge(native_node_ids[i], native_node_ids[i + 1])`.
- `DPGLocalExplanation.all_trees_valid` is `all(path.graph_path_valid for path in tree_paths)`.
- `path_confidence` is the mean of node coverage and edge coverage, so partial pruning shows up as a
  fractional score rather than a hard failure.

Consequently an aggressive `perc_var` in either mode makes `graph_path_valid` / `all_trees_valid`
go `False` for samples whose traced path was pruned. That is filtering working as intended, not a
tracing bug. If validity matters more than compactness, lower `perc_var` (or set it to `0` to skip
filtering entirely) rather than changing the mode.

## Choosing a mode

- `aggregated_transitions` keeps surviving paths internally coherent — every retained edge belongs to
  a case that was traced end to end — at the cost of dropping rare-but-real branches wholesale.
- `execution_trace` keeps any sufficiently frequent transition even if no single frequent variant
  contains it, so the graph can contain edge combinations no real path took (recombination). This is
  exactly what the recombination term in `evaluate_faithfulness` measures.

Mode is orthogonal to `decimal_threshold`: rounding decides which predicates become the *same* node
before either filter sees the log. See
[/conventions/config-resolution.md](/conventions/config-resolution.md) — note that the repo's
`config.yaml` has no `graph_construction` section at all, so loading it yields the default mode.
