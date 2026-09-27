---
type: Architecture
title: The DPG pipeline
description: How DPG turns a fitted sklearn ensemble into a decision predicate graph via process mining, stage by stage.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [architecture, pipeline, process-mining, core]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# The one thing to understand first

DPG does **not** walk trees into a graph directly. It replays samples through the ensemble to produce a
process-mining *event log*, then mines a directly-follows graph (DFG) from that log. Every design
consequence in this codebase follows from that choice.

# Stages

```
sklearn model + X
  ↓ SklearnEnsembleNormalizer        normalize heterogeneous ensembles to a uniform .estimators_ / .tree_ shape
  ↓ tracing_ensemble(_parallel)      replay each sample down every tree → [case_id, event] pairs
  ↓ DataFrame                        columns: "case:concept:name", "concept:name"; one case per (sample, tree)
  ↓ filter_log / discover_dfg        count directly-follows pairs → {(src_label, dst_label): frequency}
  ↓ build_graph                      build nx.DiGraph + nodes_list directly from the DFG; node id = "n" + sha1(key)[:12]
                                      fit() ends here, caches graph+nodes_list, and returns self (not a dot)
  ↓ to_dot() / to_networkx()         called after fit(): to_dot() renders graphviz.Digraph afresh every call
                                      (display only); to_networkx() returns the cached graph/nodes_list,
                                      re-parsing dot.body TEXT only as a fallback when not fitted
  ↓ NodeMetrics / EdgeMetrics / GraphMetrics
  ↓ DPGExplanation / DPGLocalExplanation
  ↓ visualizer.py                    render to graphviz / matplotlib
```

| Stage | Owner | Input | Output |
|---|---|---|---|
| Normalize | [SklearnEnsembleNormalizer](/modules/dpg-sklearn-normalizer.md) | any supported sklearn ensemble | uniform tree-path interface |
| Trace | [dpg.core](/modules/dpg-core.md) | model + `X` | `[case_id, event_label]` pairs |
| Filter | [dpg.core](/modules/dpg-core.md) | trace log | log with rare variants (or edges) dropped |
| Mine | [dpg.core](/modules/dpg-core.md) | trace log | DFG: `{(src, dst): frequency}` |
| Build graph | [dpg.core](/modules/dpg-core.md) | DFG | `nx.DiGraph` + `nodes_list` (`build_graph`) |
| Render to DOT | [dpg.core](/modules/dpg-core.md) | `nx.DiGraph` (via `to_dot()`) | `graphviz.Digraph`, for display/export only |
| Measure | [metrics](/modules/metrics-graph.md) | `nx.DiGraph`, `nodes_list`, `target_names` | metric frames/dicts |
| Explain | [DPGExplainer](/modules/dpg-explainer.md) | all of the above | explanation dataclasses |
| Visualize | [dpg.visualizer](/modules/dpg-visualizer.md) | explanation | figures / DOT renders |

# Consequences worth knowing

1. **Node identity is the predicate text.** Ids are `"n" + sha1(label)[:12]`, so merging is
   byte-equality of label strings, and `decimal_threshold` therefore directly controls how much the
   graph merges. See [the label contract](/conventions/label-contract.md).
2. **Label formats are a cross-module contract.** `dpg/core.py`, `metrics/graph.py`, `dpg/explainer.py`
   and `dpg/visualizer.py` all parse the same string shapes.
3. **DOT is a rendering output, not a construction step.** `build_graph` builds the `nx.DiGraph`
   directly from the DFG; `fit(X)` returns `self`, not a dot. `to_dot()` renders that graph to DOT
   afresh on every call — deliberately uncached, since the visualizer recolors a dot it is given in
   place — and raises `DPGNotFittedError` before `fit`. `to_networkx(graphviz_graph=None)` returns the
   graph `fit()` already built via a cached reference; the old text-parsing path (splitting `dot.body`
   on `"->"` and regexing `label="..."`) only runs when this instance isn't fitted and a
   `graphviz.Digraph` it did not build is passed in — with neither, it raises `DPGNotFittedError` too.
4. **Filtering is a modelling choice, not noise removal.** The two
   [graph-construction modes](/conventions/graph-construction-modes.md) drop whole path *variants* or
   individual *edges*. Aggressive `perc_var` legitimately makes local-explanation validity flags go
   `False` — that is filtering working, not a tracing bug.
5. **Local explanations re-trace.** `explain_local` always re-runs the sample through the raw trees
   (reported as `path_mode="execution_trace"`) and then checks each traced node/edge against the built
   graph.

# Related

- [Project overview](/index.md)
- [Config resolution](/conventions/config-resolution.md)
- [Faithfulness evaluation](/concepts/faithfulness-evaluation.md)
