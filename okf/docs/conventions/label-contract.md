---
type: Convention
title: Event label contract
description: The exact string formats DPG uses for graph node labels, and every module that parses them.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [contract, labels, parsing, node-identity]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Why this is a contract

DPG has no separate node-identity model. A node's identity **is** its label text: the node id is the
SHA-1 of the label string, rendered as a decimal integer.

```python
# dpg/core.py — id derivation, mirrored verbatim in dpg/explainer.py:_label_to_node_id
str(int(hashlib.sha1(activity.encode()).hexdigest(), 16))
```

Two predicates collapse into one node **iff their label strings are byte-identical**. Everything
downstream — metrics, explanations, plots — re-derives meaning by parsing those strings. So the label
formats below are a cross-module contract, not an implementation detail.

# Formats

| Kind | Format | Produced at | Notes |
|---|---|---|---|
| Split, left branch | `<feature> <= <threshold>` | `dpg/core.py` (`tracing_ensemble`, `tracing_ensemble_parallel`) | `threshold = round(tree_.threshold[i], decimal_threshold)` |
| Split, right branch | `<feature> > <threshold>` | same | same rounding |
| Classifier leaf | `Class <name>` | `dpg/core.py:_leaf_class_label` | `<name>` is `target_names[k]` if given, else `model.classes_[k]`, else the raw index |
| Regressor leaf | `Pred <value>` | `dpg/core.py` tracing | `value = round(tree_.value[i][0][0], 2)` |
| Trace case id | `sample<case_id>_dt<tree_index>` | `dpg/core.py` tracing | one case per (sample, tree) |

The trace log is a `DataFrame` with exactly two columns, using PM4Py-style names:
`"case:concept:name"` (the case id above) and `"concept:name"` (the label above).

# Consumers that parse these strings

| Location | Helper | Expects |
|---|---|---|
| `metrics/graph.py` | `GraphMetrics._parse_predicate` | regex `^\s*(.+?)\s*(<=\|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$` → `(feature, operator, threshold)`, else `None` |
| `metrics/graph.py` | `GraphMetrics._normalize_class_label` | strips a single leading `"Class "` prefix |
| `dpg/visualizer.py` | `_normalize_class_label` | independent copy of the same rule |
| `dpg/explainer.py` | `_label_to_node_id` | re-hashes a label to find its node in the built graph |

Note the duplication: `_normalize_class_label` exists separately in `metrics/graph.py` and
`dpg/visualizer.py`. Changing the `Class ` prefix means changing both.

# Gotchas

- **Rounding controls merging.** `decimal_threshold` is applied to split thresholds only. Lower it and
  distinct thresholds collapse into one node; raise it and the graph fragments. See
  [config resolution](/conventions/config-resolution.md).
- **Regressor leaves ignore `decimal_threshold`.** `Pred` values are hard-rounded to 2 decimals, so
  regression leaf granularity is not configurable through the same knob.
- **Feature names with `<=` or `>` in them would break parsing.** `_parse_predicate` is non-greedy on the
  feature group, so a feature named e.g. `a > b` parses ambiguously.
- **`to_networkx` round-trips through DOT source text.** The graph is rebuilt by parsing the text emitted
  by `generate_dot`, so any change to label escaping or attribute ordering in `generate_dot` can silently
  break node/edge parsing. See [dpg.core](/modules/dpg-core.md).
- **Normalized vs raw labels differ by field.** Local explanations use normalized class names (`"0"`) in
  `class_votes` / `majority_vote`, but keep raw DPG labels (`"Class 0"`) inside `tree_paths[*].labels`.
  See [explanation dataclasses](/concepts/explanation-dataclasses.md).

# Related

- [The DPG pipeline](/pipeline.md)
- [dpg.core](/modules/dpg-core.md)
- [metrics.graph](/modules/metrics-graph.md)
