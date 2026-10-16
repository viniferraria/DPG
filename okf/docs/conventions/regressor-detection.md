---
type: Convention
title: Regressor detection
description: is_regression_model as the single predicate that routes tracing into Pred leaves instead of Class leaves, shared by dpg/core.py and dpg/explainer.py.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [contract, regressor, is_regression_model, econml, causal-forest, label-contract]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-09T00:00:00Z }
status: stable
---

# Responsibility

Every tracing loop in DPG has to decide, per leaf, whether to emit a classifier leaf label
(`"Class <name>"`) or a regressor leaf label (`"Pred <value>"`) — see
[the label contract](/conventions/label-contract.md). That decision is one function,
`is_regression_model(model)`, defined in `dpg/core.py` and imported by `dpg/explainer.py`:

```python
# dpg/core.py
def is_regression_model(model: Any) -> bool:
    return is_regressor(model) or isinstance(model, _CAUSAL_FOREST_MODELS)
```

`is_regressor` is sklearn's own, so every sklearn regressor (including `GradientBoostingRegressor`)
is covered without a hand-maintained list. econml's `CausalForest` sets no `_estimator_type`, so it
is added explicitly through `_CAUSAL_FOREST_MODELS`, which is `(econml.grf.CausalForest,)` when
`from econml.grf import CausalForest` succeeds at import time, else `()` (`isinstance(x, ())` is
always `False`). `is_regression_model` is in `dpg/core.py`'s `__all__`.

An older design used a module-level tuple of regressor classes; it no longer exists in the source.

# API

| Call site | Function | Uses it to |
|---|---|---|
| `dpg/core.py:532` | `DecisionPredicateGraph._trace_tree_labels_legacy` (aggregated-transitions leaf) | pick `Pred`-leaf vs. `Class`-leaf label |
| `dpg/core.py:573` | `DecisionPredicateGraph._trace_tree_labels` (execution-trace leaf) | same |
| `dpg/explainer.py:649` | `DPGExplainer._trace_tree_path` | same, when building a `DPGTreePathExplanation` for local explanation |
| `dpg/explainer.py:1020` | `DPGExplainer._trace_execution_labels_for_tree` | same, when re-deriving the raw execution trace |

All four do the identical leaf-label branch: `round(float(tree_.value[node][0][0]), 2)` →
`f"Pred {pred}"` for a regressor, else `self._leaf_class_label(...)`.

## Optional-import mechanism

```python
try:
    from econml.grf import CausalForest
    _CAUSAL_FOREST_MODELS: tuple[type, ...] = (CausalForest,)
except ImportError:
    _CAUSAL_FOREST_MODELS = ()
```

`econml` is an optional dependency, installed via the `causal` extra (`econml>=0.17`). When it is
absent, `_CAUSAL_FOREST_MODELS` is empty, `is_regression_model` reduces to sklearn's `is_regressor`,
and nothing else in DPG changes behavior.

## The `n_relevant_outputs_` guard

`is_regression_model` only decides the tracing *branch*; it does not decide whether a `CausalForest`
is safe to build a DPG from. That check is separate, in `DecisionPredicateGraph.__init__`
(`dpg/core.py:158-161`), guarded by the same optional-import flag:

```python
if _CAUSAL_FOREST_MODELS and isinstance(model, _CAUSAL_FOREST_MODELS):
    n_relevant_outputs = int(getattr(model, "n_relevant_outputs_", 1))
    if n_relevant_outputs != 1:
        raise DPGModelError.multi_output_causal_forest(n_relevant_outputs)
```

See [causal forest support](/concepts/causal-forest-support.md) for why a multi-output forest can't
be traced correctly even though `is_regression_model` would route it into the `Pred` branch, and for
the `CausalForestDML` case (no `estimators_`, rejected before this guard).

# Side effect: this closed a known bug

Before commit `8295760`, `dpg/explainer.py` had its **own** tuple of regressor classes —
`(RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor)` — missing
`GradientBoostingRegressor`, which `dpg/core.py`'s tuple already included. The two tracing sides
disagreed on GB regressors specifically:

- `dpg/core.py` (global graph construction) took the regressor branch → leaves labeled
  `"Pred <value>"`.
- `dpg/explainer.py` (`explain_local`) took the *classifier* branch → leaves labeled
  `"Class <target_names[0]>"` (or `"Class 0"` with no `target_names`).

Because node identity is the label text (see
[the label contract](/conventions/label-contract.md)), a local explanation's leaf label could never
match any node in the graph core.py had built.

**Before** (illustrative — old explainer tuple, `target_names=["price"]`):

```python
tree_paths[0].labels[-1]   # "Class price"
```

**After** (current — both modules call `is_regression_model`):

```python
tree_paths[0].labels[-1]   # "Pred 4.87"
```

This is a **behavior change** relative to 0.1.6: any saved logs, notebooks, or regression tests that
captured `explain_local` output for a `GradientBoostingRegressor` and asserted on `"Class ..."` leaf
labels will now see `"Pred ..."` instead. Global (`core.py`) output for GB regressors is unchanged.

# Remaining gap: leaf values are not predictions for GB regressors

Routing is now consistent, but a `Pred` leaf is the tree's own `tree_.value`, which is not the
ensemble prediction for gradient boosting (`predict` adds the init estimator and sums
learning-rate-scaled trees) or for `CausalForest` (see
[the leaf-value limitation](/concepts/causal-forest-support.md)).

# Related

- [Event label contract](/conventions/label-contract.md)
- [dpg.core](/modules/dpg-core.md)
- [dpg.sklearn_normalizer](/modules/dpg-sklearn-normalizer.md)
- [Causal forest support](/concepts/causal-forest-support.md)
