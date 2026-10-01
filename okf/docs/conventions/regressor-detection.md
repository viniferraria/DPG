---
type: Convention
title: Regressor detection
description: REGRESSOR_MODELS as the single isinstance check that routes tracing into Pred leaves instead of Class leaves, shared by dpg/core.py and dpg/explainer.py.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [contract, regressor, isinstance, econml, causal-forest, label-contract]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-09T00:00:00Z }
status: stable
---

# Responsibility

Every tracing loop in DPG has to decide, per leaf, whether to emit a classifier leaf label
(`"Class <name>"`) or a regressor leaf label (`"Pred <value>"`) — see
[the label contract](/conventions/label-contract.md). That decision is a single `isinstance` check
against a module-level tuple, `REGRESSOR_MODELS`, defined once in `dpg/core.py` and imported by
`dpg/explainer.py` rather than redefined.

```python
# dpg/core.py
REGRESSOR_MODELS: tuple[type, ...] = (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    *_CAUSAL_FOREST_MODELS,
)
```

`_CAUSAL_FOREST_MODELS` is `(econml.grf.CausalForest,)` when `from econml.grf import CausalForest`
succeeds at import time, else `()`. `REGRESSOR_MODELS` is part of `dpg/core.py`'s `__all__`.

# API

| Consumer | Function | Uses it to |
|---|---|---|
| `dpg/core.py` | `DecisionPredicateGraph.tracing_ensemble` | pick `Pred`-leaf vs. `Class`-leaf branch during the sequential global trace |
| `dpg/core.py` | `DecisionPredicateGraph.tracing_ensemble_parallel` | same, for the joblib-parallel global trace |
| `dpg/explainer.py` | `DPGExplainer._trace_tree_path` | same, when building a `DPGTreePathExplanation` for local explanation |
| `dpg/explainer.py` | `DPGExplainer._trace_execution_labels_for_tree` | same, when re-deriving the raw execution trace used by `evaluate_faithfulness`'s trace-diagnostics |

All four sites do the identical `is_regressor = isinstance(self.model, REGRESSOR_MODELS)` (or
`self._builder.model` in `explainer.py`) followed by the identical leaf-label branch:
`round(tree_.value[node_index][0][0], 2)` → `f"Pred {pred}"`.

## Optional-import mechanism

```python
try:
    from econml.grf import CausalForest
    _CAUSAL_FOREST_MODELS: tuple[type, ...] = (CausalForest,)
except ImportError:
    _CAUSAL_FOREST_MODELS = ()
```

`econml` is an optional dependency, installed via the `causal` extra (`econml>=0.17`) declared in
`pyproject.toml`. When it is absent, `_CAUSAL_FOREST_MODELS` is the empty tuple, `REGRESSOR_MODELS`
is unaffected (just the four sklearn regressor classes), and nothing else in DPG changes behavior —
there is no code path that requires `econml` to be installed for non-causal-forest models.

## The `n_relevant_outputs_` guard

`REGRESSOR_MODELS` only decides the tracing *branch*; it does not decide whether a `CausalForest` is
safe to build a DPG from. That check is separate and happens once, in
`DecisionPredicateGraph.__init__`, guarded by the same optional-import flag:

```python
if _CAUSAL_FOREST_MODELS and isinstance(model, _CAUSAL_FOREST_MODELS):
    n_relevant_outputs = int(getattr(model, "n_relevant_outputs_", 1))
    if n_relevant_outputs != 1:
        raise DPGModelError.multi_output_causal_forest(n_relevant_outputs)
```

See [causal forest support](/concepts/causal-forest-support.md) for why a multi-output forest can't
be traced correctly even though `REGRESSOR_MODELS` would happily route it into the `Pred` branch.

# Side effect: this closed a known bug

Before this change, `dpg/explainer.py` had its **own** 3-tuple —
`(RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor)` — missing
`GradientBoostingRegressor`, which `dpg/core.py`'s tuple already included. The two tracing sides
disagreed on GB regressors specifically:

- `dpg/core.py` (global graph construction) took the regressor branch → leaves labeled
  `"Pred <value>"`.
- `dpg/explainer.py` (`explain_local`, via `_trace_tree_path` / `_trace_execution_labels_for_tree`)
  took the *classifier* branch → leaves labeled `"Class <target_names[0]>"` (or `"Class 0"` with no
  `target_names`).

Because node identity is the label text (see
[the label contract](/conventions/label-contract.md)), a local explanation's leaf label could never
match any node in the graph core.py had built. `explain_local` on a `GradientBoostingRegressor`
therefore produced tree paths whose leaves were structurally unfindable in the graph
(`graph_path_valid` false at the leaf, `native_node_id not in node_metrics_lookup`).

**Before** (illustrative — old explainer tuple, `target_names=["price"]`):

```python
tree_paths[0].labels[-1]   # "Class price"
```

**After** (current — `REGRESSOR_MODELS` shared with `core.py`):

```python
tree_paths[0].labels[-1]   # "Pred 4.87"
```

This is a **behavior change** relative to 0.1.6: any saved logs, notebooks, or regression tests that
captured `explain_local` output for a `GradientBoostingRegressor` and asserted on `"Class ..."` leaf
labels will now see `"Pred ..."` instead. Global (`core.py`) output for GB regressors is unchanged —
only the local/explainer side moved to match it.

# Related

- [Event label contract](/conventions/label-contract.md)
- [dpg.core](/modules/dpg-core.md)
- [dpg.sklearn_normalizer](/modules/dpg-sklearn-normalizer.md)
- [Causal forest support](/concepts/causal-forest-support.md)
