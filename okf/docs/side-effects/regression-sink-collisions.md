---
type: Side Effect
title: Regression sinks collide by rounding, not by output identity
description: A regression leaf's DPG sink label is "Pred <value rounded to 2 decimals>", hard-coded independently of decimal_threshold. Two distinct leaves whose true predictions differ can round to the identical label and merge into one graph sink purely by coincidence of rounding.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [regression, node-identity, rounding, sinks, limitation]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed / pre-existing limitation

Node identity in DPG is the label's exact text (see
[Event label contract](/conventions/label-contract.md)): a regression leaf becomes `f"Pred {round(value, 2)}"`.
That 2-decimal rounding is hard-coded at every tracing call site, independent of `decimal_threshold`
(which only controls split-predicate rounding):

```python
# dpg/core.py:475-476 (_trace_tree_labels_legacy)
pred = round(float(tree_.value[node_index][0][0]), 2)
labels.append(f"Pred {pred}")
```

```python
# dpg/core.py:516-517 (_trace_tree_labels, execution_trace)
pred = round(float(tree_.value[leaf_id][0][0]), 2)
labels.append(f"Pred {pred}")
```

0.3.0 doesn't change this rounding; it explicitly declares it out of scope and pins it down with a
regression test (`tests/test_e6_regression_scope.py::test_regressor_sink_count_is_not_a_fixed_output_count`,
added by commit `f2113e3`) that asserts the DPG's distinct `"Pred "` sinks equal exactly the set of
`round(leaf_value, 2)` strings — i.e. it locks in the collision-by-rounding behavior rather than fixing
it, as a guard against silently changing the rounding precision.

# Why

Classification has a natural, small, enumerable sink set: one `"Class <name>"` per class. Regression
has no equivalent bounded output space — a regression tree can have arbitrarily many distinct leaf
values. DPG needs *some* finite label to hash into a node id, so it rounds to 2 decimals. That rounding
is a display/identity convenience, not a semantically meaningful clustering of "similar" outputs: two
leaves separated by less than 0.005 collide into one sink; two leaves separated by more than 0.005
stay distinct, regardless of whether they come from the same tree, different trees, or represent
meaningfully different or meaningfully similar predictions.

# Who is affected

Anyone building a DPG from a regressor (`RandomForestRegressor`, `GradientBoostingRegressor`,
`ExtraTreesRegressor`, `AdaBoostRegressor`) and reading sink-level structure (sink count, edges into a
sink, weight pooled at a sink) as if it corresponded to distinct model outputs. It does not: sink count
tracks rounding collisions, not "number of outputs" (see the CHANGELOG limitation quoted below).
Community/class-boundary extraction is unaffected by this specific issue since those features already
reject regression DPGs outright — see
[communities-regressor-valueerror](communities-regressor-valueerror.md).

# Reproduction

Force two distinct leaves of the same tree to have raw values that round to the identical label:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from dpg.core import DecisionPredicateGraph

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
y = np.array([3.001, 3.001, 3.004, 3.004, 5.0, 5.0, 7.0, 7.0])
model = RandomForestRegressor(n_estimators=1, bootstrap=False, max_depth=2, random_state=0).fit(X, y)
tree = model.estimators_[0]
tree_ = tree.tree_
for leaf in (i for i in range(tree_.node_count) if tree_.children_left[i] == -1):
    val = float(tree_.value[leaf][0][0])
    print("leaf", leaf, "raw value:", val, "-> label: Pred", round(val, 2))

dpg_config = {"dpg": {"default": {"perc_var": 1e-9, "decimal_threshold": 3, "n_jobs": 1},
                       "graph_construction": {"mode": "aggregated_transitions"}, "visualization": {}}}
builder = DecisionPredicateGraph(model, feature_names=["f0"], dpg_config=dpg_config)
for x in (0.0, 2.0):
    labels = builder._trace_tree_labels_legacy(0, tree, np.array([x]))
    print("sample f0=", x, "-> path:", labels, "| model.predict:", model.predict([[x]]))
```

Run with `uv run python` from the repo root. Real output:

```text
leaf 2 raw value: 3.001 -> label: Pred 3.0
leaf 3 raw value: 3.004 -> label: Pred 3.0
leaf 5 raw value: 5.0 -> label: Pred 5.0
leaf 6 raw value: 7.0 -> label: Pred 7.0
sample f0= 0.0 -> path: ['f0 <= 3.5', 'f0 <= 1.5', 'Pred 3.0']
sample f0= 2.0 -> path: ['f0 <= 3.5', 'f0 > 1.5', 'Pred 3.0']
model.predict: [3.001]
model.predict: [3.004]
```

`f0=0.0` and `f0=2.0` are genuinely different leaves with genuinely different model predictions
(`3.001` vs `3.004`), yet both trace to the identical sink node `"Pred 3.0"` in the pooled graph — by
coincidence of 2-decimal rounding, not because the model treats them as the same output.

# Workaround

None built in — this is a documented, deferred limitation (CHANGELOG: "A principled regression sink
policy is deferred to a future release"). Treat regression sink structure as approximate; if exact
per-leaf identity matters, trace samples directly via `tree.apply()` / `tree.tree_.value` rather than
relying on the DPG sink label.

# File:line citations (HEAD `276a503`)

- `dpg/core.py:475-476` — `_trace_tree_labels_legacy` regressor leaf label (`aggregated_transitions`).
- `dpg/core.py:516-517` — `_trace_tree_labels` regressor leaf label (`execution_trace`).
- `dpg/explainer.py` — `DPGExplainer._trace_tree_path` mirrors the same `round(value, 2)` for local
  explanations.
- `tests/test_e6_regression_scope.py::test_regressor_sink_count_is_not_a_fixed_output_count` — pins
  sink identity to exactly `round(leaf_value, 2)`, added by commit `f2113e3`.
- CHANGELOG.md 0.3.0 "Compatibility and limitations": "**Regression sink semantics are out of scope for
  0.3.0.** ... there is no 'one sink per output' guarantee: a regression sink is only as unique as the
  2-decimal rounded leaf value, so two leaves collide into one sink by coincidence of rounding, not by
  any modeled notion of 'output'."

# Related

- [communities-regressor-valueerror](communities-regressor-valueerror.md)
- [Event label contract](/conventions/label-contract.md)
