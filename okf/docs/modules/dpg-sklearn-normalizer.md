---
type: Python Module
title: dpg.sklearn_normalizer
description: Normalizes heterogeneous scikit-learn ensembles to a uniform 1D `.estimators_` list of objects exposing `.tree_`, preserving the GradientBoosting class-slot index.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/sklearn_normalizer.py
tags: [python, module, sklearn, ensemble, gradient-boosting, normalization, dpg]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

DPG's tracing loop (see `/modules/dpg-core.md`) walks every tree in an ensemble with a single
uniform expression:

```python
for i, tree in enumerate(self.model.estimators_):
    tree_ = tree.tree_
    ...
```

That loop only works if `model.estimators_` is a flat, iterable sequence whose items each expose a
scikit-learn `.tree_` (with `children_left`, `children_right`, `feature`, `threshold`, `value`).
Most sklearn ensembles already satisfy this. `GradientBoosting*` does not — its `estimators_` is a
2D `ndarray` of shape `(n_estimators, n_trees_per_iteration)`.

`SklearnEnsembleNormalizer` is the single adapter that closes that gap. It is a stateless class of
three `@staticmethod`s; it holds no instance state and is never instantiated.

# API

| Member | Signature | Purpose |
|---|---|---|
| `GB_MODELS` | `tuple` class attribute | `(GradientBoostingClassifier, GradientBoostingRegressor)` — the only families that trigger normalization. |
| `needs_normalization` | `(model: Any) -> bool` | `isinstance(model, GB_MODELS)`. Nothing else returns True. |
| `normalize` | `(model: Any) -> Any` | Returns `model` unchanged when it does not need normalization; otherwise a `copy.copy` shallow clone with a flattened `estimators_`. |
| `get_tree_class_index` | `(model: Any, tree_index: int) -> int \| None` | Reads back the preserved class slot for flattened GB tree `tree_index`. Returns `None` for non-normalized models and for out-of-range indices. |

## Uniform interface produced

After `normalize`, every model DPG consumes satisfies:

| Attribute | Shape after normalization |
|---|---|
| `model.estimators_` | 1D `list` of tree-bearing estimators |
| `estimators_[i].tree_` | sklearn `Tree` — `children_left/right`, `feature`, `threshold`, `value` |
| `model._normalized_for_dpg` | `True` (only set on normalized GB copies) |
| `model._original_estimators_shape` | the original `(n_estimators, n_trees_per_iteration)` tuple |
| `model._dpg_tree_class_indices` | `list[int \| None]`, one entry per flattened tree |

# Behavior

## Supported estimator matrix

| Family | Native `estimators_` | Normalizer action | Special handling downstream |
|---|---|---|---|
| `RandomForestClassifier` | 1D list of trees | pass-through | leaf class = `argmax(tree_.value[node])` |
| `RandomForestRegressor` | 1D list of trees | pass-through | regressor branch in `tracing_ensemble`; leaf = `Pred <round(value,2)>` |
| `ExtraTreesClassifier` | 1D list of trees | pass-through | same as RandomForest classifier |
| `ExtraTreesRegressor` | 1D list of trees | pass-through | listed in `core.py`'s regressor tuple → `Pred` leaves |
| `AdaBoostClassifier` | 1D list of trees | pass-through | same as RandomForest classifier |
| `AdaBoostRegressor` | 1D list of trees | pass-through | listed in `core.py`'s regressor tuple → `Pred` leaves |
| `BaggingClassifier` | 1D list of trees | pass-through | offered by `dpg/sklearn_dpg.py`'s model registry; no normalizer code path |
| `GradientBoostingClassifier` (multiclass) | 2D `(n_estimators, n_classes)` | flatten row-major; record column index as the class slot | `core._leaf_class_label` uses the recorded slot as the predicted class, ignoring `tree_.value` |
| `GradientBoostingClassifier` (binary) | 2D `(n_estimators, 1)` | flatten; slot recorded as `None` because `len(row) == 1` | sign-of-leaf-score rule: `1 if float(tree_.value[node][0][0]) > 0 else 0` |
| `GradientBoostingRegressor` | 2D `(n_estimators, 1)` | flatten; slots all `None` | regressor branch → `Pred` leaves in `core.py` |

The two GB cases are the whole reason the module exists. A multiclass GB classifier trains one
regression tree *per class per boosting round*; each such tree's leaf values are gradient residuals,
not class scores, so `argmax` over `tree_.value` is meaningless — the class is determined solely by
which column the tree came from. Binary GB trains a single column, so the column index carries no
information and the class comes from the *sign* of the leaf score instead.

## Non-mutation guarantee

`normalize` never writes to the caller's model. It takes `copy.copy(model)` and rebinds
`estimators_` on the clone only, so the original keeps its 2D array and `model.predict()` still
works after building a DPG. `tests/test_sklearn_models.py::test_gb_original_model_predict_still_works_after_explainer`
and `::test_gb_normalizer_flattens_estimators` pin this.

## Idempotence

If `model.estimators_` is already a `list`, `normalize` returns the model untouched — even for a GB
model. This makes repeated normalization safe, but also means a previously normalized GB copy will
not have its `_dpg_tree_class_indices` recomputed.

## Unsupported estimators

The normalizer itself never rejects anything: `needs_normalization` returns `False` and the model
is returned as-is. Rejection happens one level up, in `DecisionPredicateGraph.__init__`, which
raises `DPGModelError.invalid_ensemble()` ("Model must be a tree-based ensemble with fitted
estimators.") when the object has no `estimators_` attribute. A model that *has* `estimators_` but
whose entries lack `.tree_` (e.g. a `BaggingClassifier` built on non-tree base estimators) passes
validation and then fails later with an `AttributeError` during tracing.

# Gotchas

- **The module normalizes GradientBoosting only.** Despite the docstring naming RandomForest and
  AdaBoost, those families take the identity path — they were already flat. Do not read the matrix
  above as "six code paths"; there is one branch plus pass-through.
- **`get_tree_class_index` returning `None` is overloaded.** It means *any* of: model not
  normalized, model not GB, binary GB, GB regressor, or index out of range. Callers must therefore
  re-check `isinstance(model, GradientBoostingClassifier) and n_classes_ == 2` to reach the
  sign-of-leaf-score rule — which both `core._leaf_class_label` and `DPGExplainer._leaf_class_label`
  do, as two independent copies of the same logic.
- **Leaf-label duplication.** `_leaf_class_label` exists verbatim in `dpg/core.py` (global tracing)
  and `dpg/explainer.py` (local re-tracing). Any change to the GB class-resolution rule must be
  applied to both or global and local explanations will disagree on `"Class …"` strings — and node
  identity is the label text, see `/conventions/label-contract.md`.
- **`GradientBoostingRegressor` is missing from the explainer's regressor tuple.**
  `dpg/core.py` tests against `(RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
  GradientBoostingRegressor)`, but `dpg/explainer.py` uses only the first three. Local explanation
  of a GB regressor therefore takes the classifier branch and emits `"Class <target_names[0]>"`
  where the graph holds `"Pred <value>"` nodes, so lookups miss.
- **Private attributes are part of the contract.** `_dpg_tree_class_indices`,
  `_normalized_for_dpg` and `_original_estimators_shape` are read by other DPG modules despite the
  underscore. They also travel with pickled explainer state.

# Examples

```python
from sklearn.ensemble import GradientBoostingClassifier
from dpg.sklearn_normalizer import SklearnEnsembleNormalizer

gb = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X, y)
gb.estimators_.shape                      # (5, 3) for 3 classes

norm = SklearnEnsembleNormalizer.normalize(gb)
len(norm.estimators_)                     # 15, a flat list
gb.estimators_.shape                      # (5, 3) — original untouched

SklearnEnsembleNormalizer.get_tree_class_index(norm, 0)   # 0
SklearnEnsembleNormalizer.get_tree_class_index(norm, 1)   # 1
SklearnEnsembleNormalizer.get_tree_class_index(norm, 99)  # None (out of range)
```

Binary GB — every slot is `None`, so the caller falls through to the sign rule
(`1 if float(tree_.value[node_index][0][0]) > 0 else 0`). An already-flat ensemble such as
`RandomForestClassifier` short-circuits: `needs_normalization(rf)` is `False` and
`normalize(rf) is rf`.

See also `/modules/dpg-core.md` for the tracing loop that consumes this interface and
`/conventions/label-contract.md` for the `"Class <name>"` / `"Pred <value>"` event grammar the
leaf labels must satisfy.
