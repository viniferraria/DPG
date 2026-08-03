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
  ↓ generate_dot                     graphviz.Digraph; node id = str(int(sha1(label), 16))
  ↓ to_networkx                      parses the dot.body TEXT back into nx.DiGraph + nodes_list
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
| Render to DOT | [dpg.core](/modules/dpg-core.md) | DFG | `graphviz.Digraph` |
| Re-parse | [dpg.core](/modules/dpg-core.md) | DOT source text | `nx.DiGraph` + `nodes_list` |
| Measure | [metrics](/modules/metrics-graph.md) | `nx.DiGraph`, `nodes_list`, `target_names` | metric frames/dicts |
| Explain | [DPGExplainer](/modules/dpg-explainer.md) | all of the above | explanation dataclasses |
| Visualize | [dpg.visualizer](/modules/dpg-visualizer.md) | explanation | figures / DOT renders |

# Consequences worth knowing

1. **Node identity is the predicate text.** Ids are SHA-1 of the label, so merging is byte-equality of
   label strings, and `decimal_threshold` therefore directly controls how much the graph merges. See
   [the label contract](/conventions/label-contract.md).
2. **Label formats are a cross-module contract.** `dpg/core.py`, `metrics/graph.py`, `dpg/explainer.py`
   and `dpg/visualizer.py` all parse the same string shapes.
3. **The DOT round-trip is load-bearing.** `to_networkx` reads the DOT *source text* produced by
   `generate_dot`. Changing label escaping or attribute ordering can silently break node/edge parsing.
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
