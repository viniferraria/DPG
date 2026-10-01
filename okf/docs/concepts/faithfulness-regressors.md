---
type: Metric
title: Faithfulness evaluation on regressors
description: Why DPGExplainer.evaluate_faithfulness is classifier-shaped and structurally cannot vote, score, or compare predictions for regressor models (RandomForestRegressor, GradientBoostingRegressor, econml CausalForest).
resource: https://github.com/viniferraria/DPG/blob/main/dpg/explainer.py
tags: [metric, faithfulness, regressor, causal-forest, limitation, known-limitation]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-09T00:00:00Z }
status: stable
---

# Responsibility

This page documents a limitation of
[`evaluate_faithfulness`](/concepts/faithfulness-evaluation.md): it was written assuming a
classifier — a discrete label space voted on by tree leaves — and every vote/evidence code path is
gated on the leaf label starting with `"Class "`. For a pure regressor (all leaves are
`"Pred <value>"`), that gate is never satisfied, so voting and evidence scoring are not merely
imprecise — they are structurally inert. This includes `econml.grf.CausalForest`
(see [causal forest support](/concepts/causal-forest-support.md)). The
[regressor detection](/conventions/regressor-detection.md) change that made `explain_local` trace
`CausalForest` and `GradientBoostingRegressor` consistently with the global graph does **not** touch
any of this — it is orthogonal. This page is the known-limitation record, verified by running a
fitted `RandomForestRegressor` through `explain_local` / `evaluate_faithfulness` directly.

# Behavior

## `class_votes` never accumulates regressor leaves

`DPGExplainer.explain_local` only records a vote when the leaf label starts with `"Class "`
(`dpg/explainer.py:270-271`):

```python
if path.labels and path.labels[-1].startswith("Class "):
    class_votes[self._normalize_class_vote_label(path.labels[-1])] += 1
```

A regressor's leaf labels are `"Pred <round(value, 2)>"` (see
[the label contract](/conventions/label-contract.md)), which never satisfies `startswith("Class ")`.
For a model that is entirely a regressor, `class_votes` stays an **empty** `Counter` for every
sample, so `majority_vote` (`dpg/explainer.py:273-275`, `class_votes.most_common(1)[0][0]` guarded by
`if class_votes:`) stays `None`. Observed directly:

```python
>>> local = explainer.explain_local(sample=X[0], sample_id=0)   # RandomForestRegressor
>>> local.majority_vote, local.class_votes
(None, {})
>>> [p.labels[-1] for p in local.tree_paths]
['Pred 1.68', 'Pred 1.92', 'Pred 1.61', 'Pred 1.63', 'Pred 1.52']
```

## `output_fidelity` is identically `0.0` for a pure regressor, not just unreliable

`evaluate_faithfulness` (`dpg/explainer.py:502-506`) computes:

```python
model_pred_raw = self._builder.model.predict(row_for_predict.to_frame().T)[0]   # or reshape(1, -1)
model_pred = self._normalize_prediction_label(model_pred_raw)
```

`_normalize_prediction_label` (`dpg/explainer.py:1044-1054`) special-cases matching against
`target_names`/`classes_` and a leading `"Class "` prefix; regressors have neither, so it falls
through to `return str(value)`. For a plain sklearn regressor `model_pred` is the full-precision
string form of a `numpy.float64` (e.g. `"1.6734732418416916"`); observed for `CausalForest`
(`predict(X)` returns shape `(n, n_relevant_outputs_)`, so `predict(row)[0]` is a 1-element
`ndarray`, not a scalar), `model_pred` is `str(ndarray)`, i.e. `"[1.53729555]"`.

`record["matches_model"] = bool(local_pred == model_pred)` (`dpg/explainer.py:517`) then compares
`local_pred = local.majority_vote` against that string. Since `majority_vote` is `None` for a pure
regressor, this is `bool(None == "1.6734732418416916")` — always `False`, unconditionally, not merely
unlikely to match. Observed end to end on a fitted `RandomForestRegressor`:

```python
>>> details = explainer.evaluate_faithfulness(X[:5], max_samples=5, return_details=True)
>>> details["per_sample"][["sample_id", "model_pred", "local_pred", "matches_model"]]
   sample_id           model_pred local_pred  matches_model
0          0   1.6734732418416916       None          False
1          1    2.210814247405204       None          False
2          2    1.027750515461933       None          False
3          3  0.39564923474736585       None          False
4          4   0.7728062234327904       None          False
```

`output_fidelity = mean(local_pred == model_pred)` over successful samples is therefore `0.0` for
every dataset of a pure regressor — a hard zero baked into the composite, not a fidelity signal.

## Vote-confidence and evidence metrics are hard zero too

`_compute_sample_confidence` (`dpg/explainer.py:809-825`) derives `class_scores` from `class_votes`;
with `class_votes == {}`, `total_votes == 0`, so `class_scores == {}` and both `vote_confidence` and
`score_margin` are set to `0.0` via the `if not sorted_scores:` branch — always, for every regressor
sample.

`_compute_evidence_scores` (`dpg/explainer.py:883-908`) independently walks `tree_paths` and skips
any leaf label that does not start with `"Class "` (`dpg/explainer.py:892-894`), so `class_support`
stays `{}`. With `total_support == 0` it falls back to `dict(class_scores)` — also `{}`. So
`evidence_scores == {}`, `evidence_score_pred` stays `None` (`if class_votes:` guard at
`dpg/explainer.py:833` never fires), and `evidence_score_margin` is `None`
(`sorted_evidence` empty). In the aggregate details dict, `_mean_records` treats every `None` as
excluded and defaults the mean to `0.0` when nothing remains, so
`mean_vote_confidence == mean_evidence_score_pred == mean_evidence_score_margin == 0.0` for a pure
regressor — again unconditionally, not approximately.

## What is NOT affected

The structural diagnostics — `node_recall`, `edge_recall`, `trace_coverage_score`,
`recombination_rate` (`_compute_trace_diagnostics`) — compare **node and edge identity** (predicate
label sets and label-pair sets), never leaf *values*, so they remain meaningful for regressors: they
still measure how much of the raw per-sample trace survived into the mined graph. `graph_path_valid`
/ `graph_path_valid_rate` are also value-independent (they check node/edge membership, not label
content) and remain meaningful.

# Known limitation — exact code paths

- `dpg/explainer.py:270-271` — the `startswith("Class ")` gate in `explain_local` that keeps
  `class_votes` empty for regressor leaves.
- `dpg/explainer.py:273-275` — `majority_vote` selection, `None` when `class_votes` is empty.
- `dpg/explainer.py:502-506` — `model_pred_raw` / `model_pred` computation inside
  `evaluate_faithfulness`.
- `dpg/explainer.py:517` — `matches_model = bool(local_pred == model_pred)`.
- `dpg/explainer.py:1044-1054` — `_normalize_prediction_label`, which has no numeric branch.
- `dpg/explainer.py:809-825` — `vote_confidence` / `score_margin` derivation from `class_votes`.
- `dpg/explainer.py:883-908` — `_compute_evidence_scores`, gated the same way on `"Class "`.

This is **not** fixed by the `REGRESSOR_MODELS` / `CausalForest` change in
[regressor detection](/conventions/regressor-detection.md) — that change only made `explain_local`'s
leaf labels agree with the graph's leaf labels (both `"Pred <value>"`); it did not touch any of the
`"Class "`-gated voting or scoring logic above, nor `_normalize_prediction_label`.

For a **mixed** ensemble where `REGRESSOR_MODELS` is somehow inapplicable but individual trees still
emit `"Class "` labels this section would not apply as written — but no code path in `dpg/core.py` or
`dpg/explainer.py` currently produces that mix; the regressor/classifier branch is chosen once per
model via a single `isinstance` check, so leaves are uniformly `"Pred "` or uniformly `"Class "` for
any one model.

# What a regression-aware variant would need

- **A numeric local prediction**, since `majority_vote` (a mode over discrete class labels) has no
  regressor analogue. A mean (or `path_confidence`-weighted mean) of parsed leaf values across tree
  paths is the natural substitute for "local prediction."
- **Numeric tolerance instead of string/`None` equality** for `output_fidelity` / `matches_model`:
  compare the numeric local prediction against `model_pred_raw` (unwrapped from its length-1 array
  for `CausalForest`) within an absolute or target-scale-relative tolerance.
- **A value-based evidence/confidence measure** to replace the `"Class "`-gated
  `_compute_evidence_scores` / vote-confidence logic — e.g. weighting by inverse distance to the
  aggregated prediction, or by leaf value variance, instead of counting label matches.

# Related

- [Faithfulness evaluation](/concepts/faithfulness-evaluation.md)
- [Causal forest support](/concepts/causal-forest-support.md)
- [Regressor detection](/conventions/regressor-detection.md)
- [Event label contract](/conventions/label-contract.md)
