---
type: Side Effect
title: GradientBoosting faithfulness crashed until _original_model was added
description: DPGExplainer used to call .predict() on the SklearnEnsembleNormalizer's shallow copy of the model, whose estimators_ is flattened to a 1D list for GradientBoosting; sklearn's own predict() indexes that same attribute as 2D and crashed with a confusing TypeError. Fixed pre-release by keeping a reference to the original, un-normalized model.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/explainer.py
tags: [gradient-boosting, faithfulness, evaluate_faithfulness, sklearn-normalizer, fixed-pre-release]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed

This was a real crash during 0.3.0 development, fixed **before** the 0.3.0 release
(`f16a977`) by commit `be5389a` ("Preserve original model for sklearn predict"). It is documented here
because the failure mode is instructive and the fix is easy to accidentally regress: at HEAD
`276a503` it remains fixed, but nothing prevents a future edit from calling `self._builder.model` in
place of `self._original_model` again.

`SklearnEnsembleNormalizer.normalize` (see [dpg.sklearn_normalizer](/modules/dpg-sklearn-normalizer.md))
shallow-copies a `GradientBoostingClassifier`/`Regressor` and flattens its `estimators_` from
sklearn's native `(n_stages, n_classes)` 2D ndarray to a flat 1D list, so DPG's own tree-traversal code
can iterate it uniformly with other ensembles. Before the fix, `DPGExplainer.evaluate_faithfulness`
called `.predict()` on that *same normalized copy* (`self._builder.model`) to get the black-box
model's prediction for the fidelity check. But sklearn's `GradientBoostingClassifier.predict()` →
`decision_function()` → `_raw_predict()` → `_raw_predict_init()` indexes `self.estimators_[0, 0]` — 2D
tuple indexing that only works on the original ndarray, not the flattened list:

```python
# dpg/explainer.py:147 (current, post-fix)
self._original_model = model
```

```python
# dpg/explainer.py:516, 518 (current, post-fix)
model_pred_raw = self._original_model.predict(row_for_predict.to_frame().T)[0]
...
model_pred_raw = self._original_model.predict(np.asarray(row_values).reshape(1, -1))[0]
```

Before `be5389a`, both call sites read `self._builder.model.predict(...)` instead.

# Why

`DecisionPredicateGraph` needs the normalized, flattened `estimators_` for its own tree-mining
traversal. sklearn's `GradientBoostingClassifier.predict()` needs the *original* 2D `estimators_` for
its own internal indexing. One model object cannot serve both call paths simultaneously once
normalized — the fix is to keep a second, un-normalized reference (`self._original_model`) specifically
for calls that go through sklearn's own `predict()`.

# Who is affected

Anyone calling `DPGExplainer.evaluate_faithfulness()` on a `GradientBoostingClassifier` (binary or
multiclass) at a 0.3.0-development revision before `be5389a` / at the 0.2.0 baseline (`11decd3`), which
also has this bug — `output_fidelity` (and everything the faithfulness composite depends on) could
never be computed for GB models. Non-GradientBoosting ensembles (`RandomForest`, `ExtraTrees`,
`Bagging`, `AdaBoost`) are unaffected — `SklearnEnsembleNormalizer` is an identity pass-through for
every model except GB (see [dpg.sklearn_normalizer](/modules/dpg-sklearn-normalizer.md)).

# Reproduction

Reproduced by fitting `evaluate_faithfulness` against a worktree checked out at the 0.2.0 baseline
(`11decd3`, pre-dating the fix) and comparing to the same call against `f16a977` (0.3.0 release,
post-fix). Both ran with `uv run python` and identical `dpg_config`.

At `11decd3` (0.2.0 baseline — bug present):

```text
=== GradientBoostingClassifier (binary, breast_cancer) ===
...
Traceback (most recent call last):
  File ".../repro.py", line 37, in run_case
    result = explainer.evaluate_faithfulness(X[:5], y_true=y[:5])
  File ".../dpg/explainer.py", line 491, in evaluate_faithfulness
    model_pred_raw = self._builder.model.predict(np.asarray(row_values).reshape(1, -1))[0]
  File ".../sklearn/ensemble/_gb.py", line 970, in _raw_predict_init
    X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
TypeError: list indices must be integers or slices, not tuple
FAILED: TypeError: list indices must be integers or slices, not tuple

=== GradientBoostingClassifier (multiclass, iris) ===
...
FAILED: TypeError: list indices must be integers or slices, not tuple

=== RandomForestClassifier (multiclass, iris) [control] ===
...
SUCCESS: 0.9999999999999999

=== SUMMARY ===
gb_binary: FAILED
gb_multiclass: FAILED
rf_multiclass: OK
```

At `f16a977` (0.3.0 release — fixed):

```text
=== GradientBoostingClassifier (binary, breast_cancer) ===
...
SUCCESS: 0.9999999999999999

=== GradientBoostingClassifier (multiclass, iris) ===
...
SUCCESS: 0.8499999999999999

=== RandomForestClassifier (multiclass, iris) [control] ===
...
SUCCESS: 0.9999999999999999

=== SUMMARY ===
gb_binary: OK
gb_multiclass: OK
rf_multiclass: OK
```

The same `evaluate_faithfulness(X[:5], y_true=y[:5])` call goes from an unconditional `TypeError` for
every GB model to a successful faithfulness score, with `RandomForestClassifier` behaving identically
in both revisions (control).

# Workaround

None needed at HEAD `276a503` — this is fixed. If a future change reintroduces
`self._builder.model.predict(...)` for the fidelity check, the fix is to route that call through
`self._original_model` instead.

# File:line citations

- HEAD `276a503`: `dpg/explainer.py:147` (`self._original_model = model`), `dpg/explainer.py:516,518`
  (both `evaluate_faithfulness` predict call sites).
- Fixed by commit `be5389a` ("Preserve original model for sklearn predict"), before the 0.3.0 release
  merge `f16a977`.
- Bug present at 0.2.0 baseline `11decd3`: `dpg/explainer.py:491` in that revision
  (`self._builder.model.predict(...)`).
- `dpg/sklearn_normalizer.py` — `SklearnEnsembleNormalizer`, `GB_MODELS`, the estimators_-flattening
  behavior that makes the two `predict()` call paths incompatible.

# Related

- [dpg.sklearn_normalizer](/modules/dpg-sklearn-normalizer.md)
- [Faithfulness evaluation](/concepts/faithfulness-evaluation.md)
