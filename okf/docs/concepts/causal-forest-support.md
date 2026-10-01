---
type: Concept
title: Causal forest support
description: How econml.grf.CausalForest is traced into a Decision Predicate Graph, why only single-treatment single-outcome forests (n_relevant_outputs_ == 1) are supported, and worked examples of the supported and rejected cases.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [concept, causal-forest, econml, cate, regressor, grf]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-09T00:00:00Z }
status: stable
---

# Responsibility

DPG can build a Decision Predicate Graph from a fitted `econml.grf.CausalForest`, treating it as a
regressor whose leaves hold CATE (conditional average treatment effect) estimates instead of mean
target values. Support is opt-in at the dependency level (the `causal` extra, `econml>=0.17`) and
restricted at the model level to forests with a single treatment and a single outcome. This page is
the in-depth reference; [regressor detection](/conventions/regressor-detection.md) covers the
`REGRESSOR_MODELS` mechanism that routes a `CausalForest` into the same tracing branch as any other
sklearn regressor, and [faithfulness on regressors](/concepts/faithfulness-regressors.md) covers a
separate, unrelated limitation in `evaluate_faithfulness`.

Everything below was verified by fitting real `CausalForest` models with econml `0.17.0` and reading
`econml/grf/_base_grftree.py` and the compiled `econml/tree/_tree.pyx` (via its generated `_tree.c`).

# Behavior

## (i) What DPG traverses

`cf.estimators_` is a 1D list of `econml.grf._base_grftree.GRFTree` objects — already flat, so
`SklearnEnsembleNormalizer.needs_normalization(cf)` is `False` and `normalize` returns it unchanged
(see [dpg.sklearn_normalizer](/modules/dpg-sklearn-normalizer.md)). Each `GRFTree.tree_` is an
`econml.tree._tree.Tree`, exposing the exact same array attributes DPG's tracing loop already relies
on for sklearn trees: `children_left`, `children_right`, `feature`, `threshold`, `value`. DPG's
tracing code (`dpg/core.py:tracing_ensemble`, and its parallel/local-explanation counterparts in
`dpg/explainer.py`) never distinguishes a `GRFTree` from a sklearn `DecisionTreeRegressor` — it walks
whichever `.tree_` it is handed with the same loop:

```python
left = tree_.children_left[node_index]
right = tree_.children_right[node_index]
if left == right:            # leaf
    ...
feature_index = tree_.feature[node_index]
threshold = round(tree_.threshold[node_index], self.decimal_threshold)
```

Splits are on **`X` columns only** — `tree_.feature` indexes into the same feature matrix `X` passed
to `fit`/`predict`, never into `T` or `y`. This means DPG's `feature_names` must line up with `X`'s
columns exactly as for any other ensemble; treatment and outcome never appear as predicate text.

Causal forests are *honest*: half the leaf's samples build the tree structure (splits), the other
half estimate the leaf's parameter. DPG is indifferent to this — it only walks the structure
(`children_left/right`, `feature`, `threshold`) and reads whatever is in `value` at the leaf, which
econml has already set to the (out-of-structure-sample) estimated CATE. No DPG code needs to know
about honesty; it falls out of how econml populates `tree_.value`.

## (ii) Why only `n_relevant_outputs_ == 1` is supported

Node identity in DPG is the label text (see [the label contract](/conventions/label-contract.md)). A
leaf's label is:

```python
pred = round(tree_.value[node_index][0][0], 2)
f"Pred {pred}"
```

For a general GRF, a leaf holds a parameter vector `theta(x)` of length `n_relevant_outputs_` (one
CATE coordinate per treatment × outcome pair) — `tree_.value[node_index]` has shape
`(n_relevant_outputs_, 1)`. DPG's tracer reads only coordinate `[0][0]`. If `n_relevant_outputs_ > 1`,
this would silently label every leaf using only the *first* treatment/outcome pair's CATE, and two
leaves that agree on that first coordinate but disagree on every other coordinate would collapse into
the same graph node — a silent, wrong merge, not a crash. `DecisionPredicateGraph.__init__` therefore
raises `DPGModelError.multi_output_causal_forest(n)` instead of building a graph that misrepresents
the model:

```python
if _CAUSAL_FOREST_MODELS and isinstance(model, _CAUSAL_FOREST_MODELS):
    n_relevant_outputs = int(getattr(model, "n_relevant_outputs_", 1))
    if n_relevant_outputs != 1:
        raise DPGModelError.multi_output_causal_forest(n_relevant_outputs)
```

The check runs before `SklearnEnsembleNormalizer.normalize`, so it fails fast, before any tracing.

## (iii) How `n_relevant_outputs_` is derived, and why `fit_intercept=True` is not a problem

Reading `econml/grf/_base_grf.py` and the compiled `econml/tree/_tree.pyx`: `CausalForest.fit(X, T,
y)` reshapes `T` and `y` to 2-D, and `n_relevant_outputs_ = T.shape[1] * y.shape[1]` — one CATE
parameter per treatment/outcome pair. `n_outputs_` can be **larger** than `n_relevant_outputs_`: with
`fit_intercept=True` (the default), econml estimates extra nuisance intercept parameters alongside
the CATE parameters, so `n_outputs_ == n_relevant_outputs_ + (number of intercept columns)`.

Observed on a single-treatment, single-outcome fit (`fit_intercept=True`, the default):

```python
>>> cf.n_relevant_outputs_, cf.n_outputs_
(1, 2)
```

Crucially, the `Tree.value` property is **already sliced** to just the relevant outputs —
`_get_value_ndarray()[:node_count, :n_relevant_outputs]` — while `Tree.full_value` is the
**unsliced** array, `_get_value_ndarray()[:node_count]`, which includes the intercept column(s). DPG
reads `tree_.value`, never `tree_.full_value`, so the intercept never leaks into a `Pred` label.
Observed directly on a leaf of a `fit_intercept=True` forest:

```python
>>> tree_.value.shape, tree_.full_value.shape
((13, 1, 1), (13, 2, 1))
>>> tree_.value[leaf]
[[-1.79833959]]
>>> tree_.full_value[leaf]     # second row is the intercept nuisance parameter
[[-1.79833959]
 [ 0.03713487]]
```

So `fit_intercept=True` changes `n_outputs_` but not `n_relevant_outputs_`, and therefore does not
trip the guard and does not change what a `Pred` label contains — it is always the CATE, never a
mixture with the intercept.

## (iv) Worked examples

All three were run with econml `0.17.0` against `dpg/core.py` after the `REGRESSOR_MODELS` /
`multi_output_causal_forest` change landed. Outputs below are **observed**, not predicted.

### (a) Binary treatment, scalar outcome — builds

```python
import numpy as np
from econml.grf import CausalForest
from dpg.core import DecisionPredicateGraph, DEFAULT_DPG_CONFIG

rng = np.random.RandomState(0)
n = 200
X = rng.normal(size=(n, 3))
T = rng.binomial(1, 0.5, size=n)          # scalar (binary) treatment
y = X[:, 0] * T + rng.normal(scale=0.1, size=n)   # scalar outcome

cf = CausalForest(n_estimators=20, random_state=0)
cf.fit(X, T, y)
cf.n_relevant_outputs_        # 1 -- check this before building

dpg = DecisionPredicateGraph(
    model=cf,
    feature_names=["x0", "x1", "x2"],
    dpg_config=DEFAULT_DPG_CONFIG,
)
dot = dpg.fit(X)
graph, nodes = dpg.to_networkx(dot)
```

Observed:

```
cf.n_relevant_outputs_ == 1
Total of paths: 4000            # 200 samples x 20 estimators
num Pred nodes: 104
first 5 Pred labels (sorted): ['Pred -0.0', 'Pred -0.04', 'Pred -0.08', 'Pred -0.12', 'Pred -0.16']
```

Each `Pred <value>` label is that leaf's local CATE estimate for this treatment/outcome pair — read
it the same way you would read a `RandomForestRegressor` leaf, except the target being predicted is
the treatment effect, not the outcome itself.

### (b) Two treatment columns, scalar outcome — raises

```python
T2 = rng.binomial(1, 0.5, size=(n, 2))    # two treatment columns
cf_b = CausalForest(n_estimators=20, random_state=0)
cf_b.fit(X, T2, y)
cf_b.n_relevant_outputs_      # 2

DecisionPredicateGraph(
    model=cf_b,
    feature_names=["x0", "x1", "x2"],
    dpg_config=DEFAULT_DPG_CONFIG,
)
```

Observed:

```
cf_b.n_relevant_outputs_ == 2
dpg.exceptions.DPGModelError: CausalForest models are supported only with a single treatment and a
single outcome (n_relevant_outputs_ == 1); got n_relevant_outputs_=2. Fit one CausalForest per
treatment/outcome and build a DPG for each.
```

### (c) Scalar treatment, two outcome columns — rejected before DPG is even reached

```python
y2 = np.column_stack([y, y * 2])          # two outcome columns
cf_c = CausalForest(n_estimators=20, random_state=0)
cf_c.fit(X, T, y2)
```

Observed: this raises **inside econml's own `.fit()`**, not inside DPG —

```
AttributeError: LinearMomentGRFCriterion currently only supports a scalar y
```

In econml `0.17.0`, `CausalForest`'s default splitting criterion only accepts a scalar `y`
regardless of `T`'s shape, so a `CausalForest` with `n_relevant_outputs_ == 2` arising from a
multi-column outcome cannot even be constructed with this criterion — econml's own constructor-time
restriction pre-empts DPG's `n_relevant_outputs_` guard for this specific combination. The guard in
`dpg/core.py` still exists and still fires for any `CausalForest` instance that *does* reach `fit()`
with `n_relevant_outputs_ != 1` (as example (b) demonstrates via multi-column `T`), but a multi-column
`y` with scalar `T` is not, in practice, a reachable path with econml's default criterion.

## (v) Workarounds for a multi-output forest

- **One `CausalForest` per treatment column and/or outcome column, one DPG each.** Fit and explain
  each treatment/outcome pair independently; this is exactly what the error message recommends
  ("Fit one CausalForest per treatment/outcome and build a DPG for each.").
- **One-hot a multi-valued treatment into several binary forests** rather than passing it as a single
  multi-column `T` — e.g. for a 3-arm trial, fit `arm_A_vs_control`, `arm_B_vs_control` as separate
  binary-treatment `CausalForest`s instead of one forest with `T.shape[1] == 2`.

# Related

- [Regressor detection](/conventions/regressor-detection.md) — the `REGRESSOR_MODELS` mechanism that
  routes `CausalForest` into the `Pred`-leaf tracing branch.
- [Faithfulness on regressors](/concepts/faithfulness-regressors.md) — a separate limitation:
  `evaluate_faithfulness`'s voting/scoring is inert for any regressor, `CausalForest` included.
- [Event label contract](/conventions/label-contract.md)
- [dpg.core](/modules/dpg-core.md)
- [dpg.sklearn_normalizer](/modules/dpg-sklearn-normalizer.md)
