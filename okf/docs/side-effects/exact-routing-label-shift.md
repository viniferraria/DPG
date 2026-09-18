---
type: Side Effect
title: Exact routing shifts weights and labels at threshold boundaries
description: 0.3.0 changed execution_trace tracing and local explanation to route samples through sklearn's own decision_path/apply, using decimal_threshold only to format labels. aggregated_transitions still routes on the rounded threshold. The two modes can disagree on which branch a sample took, and therefore on predicted class, for samples between the true and rounded threshold.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [regression, routing, decimal_threshold, execution_trace, aggregated_transitions, node-identity]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed

Before 0.3.0, every tracing path compared a sample's raw feature value against a **rounded** split
threshold to decide which child to descend into. 0.3.0 added a second traversal,
`_trace_tree_labels` (used by `execution_trace` mode and by `DPGExplainer._trace_tree_path` for local
explanations), which routes using sklearn's own `tree.decision_path()` / `tree.apply()` — the exact
routing the fitted model itself uses — and applies `decimal_threshold` rounding **only** when
formatting the predicate label, never when choosing the branch:

```python
# dpg/core.py:491-520 — _trace_tree_labels (execution_trace, and mirrored in
# dpg/explainer.py:645-696 DPGExplainer._trace_tree_path for local explanations)
decision_path = tree.decision_path(sample_array)
...
went_left = int(path[position + 1]) == int(tree_.children_left[node_index])
operator = "<=" if went_left else ">"
labels.append(f"{self.feature_names[feature_index]} {operator} {threshold}")  # threshold is rounded, routing is not
```

`aggregated_transitions` still uses the old behavior, kept verbatim as `_trace_tree_labels_legacy`
(`dpg/core.py:453-489`) specifically so its graph weights stay backward-compatible:

```python
# dpg/core.py:482-489 — _trace_tree_labels_legacy (aggregated_transitions)
threshold = round(float(tree_.threshold[node_index]), effective_decimal)
...
if sample_array[feature_index] <= threshold:      # compares against the ROUNDED threshold
    ...
```

`dpg.tracing_ensemble` / `tracing_ensemble_parallel` pick the extractor per mode
(`dpg/core.py:419-423`, `441-445`).

# Why

Rounding a threshold before comparing against it can move the decision boundary. If the true
threshold is `2.4999` and `decimal_threshold=3` rounds it to `2.5`, any sample in `(2.4999, 2.5]`
takes the legacy code's left branch (`<= 2.5`) even though the fitted tree's real boundary is
`2.4999` and would send that same sample right. 0.3.0 fixes this for `execution_trace` and local
explanations, but `aggregated_transitions` — the **default** mode — keeps the old, boundary-shifting
comparison on purpose, to avoid changing existing graphs' weights.

# Who is affected

Anyone using the default `aggregated_transitions` mode with samples that land in the (typically tiny)
window between a tree's true threshold and its rounded label. The effect is two-fold:

- **Weight/label mismatch with the model.** A sample can be pooled into the graph under a predicate
  path whose rounded label disagrees with the branch the fitted model actually took for that sample,
  and thus can be pooled toward the wrong leaf/class label.
- **Cross-mode disagreement.** The same sample, same tree, same `decimal_threshold` can produce a
  different traced label sequence depending solely on `graph_construction.mode`.

The window shrinks as `decimal_threshold` grows, but is never zero for any finite precision — and
`decimal_threshold="auto"`'s own off-grid warning (see
[decimal-threshold-auto-warning](decimal-threshold-auto-warning.md)) is evidence some real datasets
land thresholds off any exact decimal grid.

# Reproduction

Craft training data so a `RandomForestClassifier`'s single-tree threshold sits at `2.4999`, then trace
a sample exactly between that threshold and its 3-decimal rounding (`2.5`):

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dpg.core import DecisionPredicateGraph

X = np.array([[2.4988], [2.5010], [0.0], [5.0]])
y = np.array([0, 1, 0, 1])
model = RandomForestClassifier(n_estimators=1, bootstrap=False, max_depth=1, random_state=0)
model.fit(X, y)

threshold = float(model.estimators_[0].tree_.threshold[0])
rounded = round(threshold, 3)
x = (threshold + rounded) / 2  # strictly between true and rounded threshold
print("model.predict:", model.predict([[x]]))

legacy_config = {"dpg": {"default": {"perc_var": 1e-9, "decimal_threshold": 3, "n_jobs": 1},
                          "graph_construction": {"mode": "aggregated_transitions"}, "visualization": {}}}
b_legacy = DecisionPredicateGraph(model, feature_names=["f0"], target_names=["0", "1"], dpg_config=legacy_config)
print("aggregated_transitions labels:", b_legacy._trace_tree_labels_legacy(0, model.estimators_[0], np.array([x])))

exec_config = {"dpg": {"default": {"perc_var": 1e-9, "decimal_threshold": 3, "n_jobs": 1},
                        "graph_construction": {"mode": "execution_trace"}, "visualization": {}}}
b_exec = DecisionPredicateGraph(model, feature_names=["f0"], target_names=["0", "1"], dpg_config=exec_config)
b_exec._resolved_decimal_threshold = 3
print("execution_trace labels:        ", b_exec._trace_tree_labels(0, model.estimators_[0], np.array([x])))
```

Run with `uv run python` from the repo root. Real output:

```text
true threshold: 2.499899983406067 rounded (decimal_threshold=3): 2.5 sample: 2.4999499917030334
model.predict: [1]
aggregated_transitions (_trace_tree_labels_legacy) labels: ['f0 <= 2.5', 'Class 0']
execution_trace (_trace_tree_labels) labels:           ['f0 > 2.5', 'Class 1']
```

The fitted model predicts class `1` for this sample — matching `execution_trace`'s
`'f0 > 2.5' → 'Class 1'`. `aggregated_transitions` pools this sample's path under `'f0 <= 2.5' →
'Class 0'`, the opposite branch and the opposite class label, purely because it compares the raw
value against the **rounded** threshold instead of the tree's real one.

# Workaround

Use `graph_construction.mode = "execution_trace"` when exact fidelity to the fitted model's routing
matters more than backward-compatible pooled-graph weights. There is no workaround for
`aggregated_transitions` itself — the rounded comparison is deliberate legacy behavior, not a bug to
patch around.

# File:line citations (HEAD `276a503`)

- `dpg/core.py:419-423`, `441-445` — `tracing_ensemble` / `tracing_ensemble_parallel` select
  `_trace_tree_labels_legacy` vs `_trace_tree_labels` by `graph_construction_mode`.
- `dpg/core.py:453-489` — `_trace_tree_labels_legacy` (rounded-threshold routing, `aggregated_transitions`).
- `dpg/core.py:491-520` — `_trace_tree_labels` (exact `decision_path` routing, `execution_trace`).
- `dpg/explainer.py:645-696` — `DPGExplainer._trace_tree_path`, the local-explanation mirror of the
  exact-routing traversal (always used by `explain_local`, regardless of `graph_construction_mode` —
  see [explain-local-node-lookup-crash](explain-local-node-lookup-crash.md) for why it currently
  cannot run at all).
- CHANGELOG.md 0.3.0 "Added": "Added exact sklearn `decision_path` routing. Threshold rounding now
  formats predicate labels without changing the branch selected by the model." and "Compatibility and
  limitations": "The routing correction can change graph weights and labels at floating-point
  boundaries."

# Related

- [Event label contract](/conventions/label-contract.md)
- [Graph construction modes](/conventions/graph-construction-modes.md)
- [decimal-threshold-auto-warning](decimal-threshold-auto-warning.md)
- [explain-local-node-lookup-crash](explain-local-node-lookup-crash.md)
