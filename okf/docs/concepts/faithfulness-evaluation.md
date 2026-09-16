---
type: Metric
title: Faithfulness evaluation
description: How DPGExplainer.evaluate_faithfulness scores local DPG explanations against the fitted ensemble, and how the sample-confidence diagnostics that feed it are computed.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/explainer.py
tags: [metric, faithfulness, fidelity, local-explanation, evaluation]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`DPGExplainer.evaluate_faithfulness` measures how well the local DPG explanations reproduce the
fitted black-box ensemble, plus structural diagnostics about how much of the raw execution trace
survives into the mined graph. Its docstring is explicit that it does **not** measure ground-truth
correctness unless `y_true` is supplied, and that the composite is "a heuristic summary, not a
calibrated probability".

Everything it aggregates is produced per-sample by `_compute_sample_confidence` (which fills
`DPGLocalExplanation.sample_confidence`) and its two helpers `_compute_evidence_scores` and
`_compute_trace_diagnostics`.

# API

```python
evaluate_faithfulness(
    X,
    y_true=None,
    max_samples=None,
    weights=None,
    return_details=False,
    sample_ids=None,
) -> float | dict[str, Any]
```

| Argument | Behavior |
|---|---|
| `X` | `pd.DataFrame` (rows predicted via `row.to_frame().T`) or array-like (rows reshaped to `(1, -1)`). |
| `y_true` | Optional labels; normalized with `_normalize_prediction_label`, enabling `local_accuracy` and a per-sample `correct` column. |
| `max_samples` | Slices `X` (and `y_true` / `sample_ids`) to the first N rows. `<= 0` raises `DPGValidationError.invalid_max_samples()`. |
| `weights` | Composite weights; see below. |
| `return_details` | `False` → returns the composite `float`. `True` → returns the details dict. |
| `sample_ids` | Ids attached to each record; defaults to `range(n_samples)`. |

Errors: not fitted → `DPGNotFittedError.for_faithfulness()`; empty input →
`DPGValidationError.empty_evaluation_input()`; length mismatches →
`DPGValidationError.mismatched_y_true_length()` / `.mismatched_sample_ids_length()`; every sample
failing → `DPGExplanationError.all_local_explanations_failed()`.

Per-sample `explain_local` failures are caught individually (`except Exception`), recorded with
`error=str(exc)` and all-`None` metrics, counted in `n_local_failures`, and excluded from the means.

# Behavior

## Component metrics

| Metric | Where computed | Formula |
|---|---|---|
| `output_fidelity` | `evaluate_faithfulness` | `mean(local_pred == model_pred)` over successful samples, where `local_pred` is `DPGLocalExplanation.majority_vote` and `model_pred` is `model.predict(row)` run through `_normalize_prediction_label`. |
| `trace_coverage_score` | `_compute_trace_diagnostics` | `(node_recall + edge_recall) / 2`. |
| `node_recall` / `node_precision` | `_compute_trace_diagnostics` | Size of `trace_nodes & explanation_nodes` divided by the size of `trace_nodes` (recall) or `explanation_nodes` (precision), over sets of non-leaf predicate labels. Both default to `1.0` when the denominator set is empty. |
| `edge_recall` / `edge_precision` | `_compute_trace_diagnostics` | Same, over sets of `(label[i], label[i+1])` pairs. Explanation edges are only counted when both node ids resolved and `edge_exists[i]` is `True`. |
| `recombination_rate` | `_compute_trace_diagnostics` | Size of `explanation_edges - trace_edges` divided by size of `explanation_edges`, i.e. the fraction of graph edges the explanation walked that the raw ensemble never executed for this sample. `0.0` if there are no explanation edges. |
| `evidence_score_margin` | `_compute_sample_confidence` | Top `evidence_scores` value minus the runner-up; equals the top value outright when only one class has evidence. `None` when `evidence_scores` is empty. |
| `vote_confidence` | `_compute_sample_confidence` | Largest normalized vote share, `class_votes[c] / total_votes`. |
| `score_margin` | `_compute_sample_confidence` | Same, top vote share minus second (or the top value alone if only one class). |
| `graph_path_valid_rate` | `_compute_sample_confidence` | `num_valid_paths / num_paths`, over `DPGTreePathExplanation.graph_path_valid`. |

The reference trace used by `_compute_trace_diagnostics` is *re-derived independently* by
`_extract_execution_trace_labels` / `_trace_execution_labels_for_tree`, walking the raw trees again
without any graph lookup. The explanation side comes from the already-built `tree_paths`. So these
are genuinely "raw ensemble behavior vs. what the mined graph supports".

## Evidence scores

`_compute_evidence_scores` accumulates, per normalized class name, the `path_confidence` of every
tree path whose leaf label starts with `"Class "` (`class_support`), then normalizes:
`evidence_scores[c] = class_support[c] / sum(class_support)`. If total support is zero it falls back
to a copy of `class_scores` (the plain vote shares). `evidence_score_pred` is the evidence score of
the majority-vote class; `evidence_margin_pred_vs_competitor` is that minus the runner-up's score.

## Composite score

```python
composite = (
    weights["output_fidelity"]   * output_fidelity
  + weights["trace_coverage"]    * mean_trace_coverage_score
  + weights["anti_recombination"] * (1.0 - mean_recombination_rate)
  + weights["evidence_margin"]   * mean_evidence_score_margin
)
```

Defaults (`_validate_faithfulness_weights`):

| Key | Default |
|---|---|
| `output_fidelity` | `0.35` |
| `trace_coverage` | `0.30` |
| `anti_recombination` | `0.20` |
| `evidence_margin` | `0.15` |

Custom weights must supply **exactly** these four keys and sum to `1.0` within `atol=1e-6`, else
`DPGExplanationError.unsupported_weight_keys` / `.missing_weight_keys` / `.weights_do_not_sum_to_one`.

All `mean_*` values use `_mean_records`, which skips `None`/`NaN` and returns `0.0` when nothing
remains — so a metric that is never populated silently contributes `0.0` rather than erroring.

## Return shape (`return_details=True`)

| Key | Type |
|---|---|
| `faithfulness_score` | `float` (the composite, identical to the scalar return) |
| `weights` | `dict[str, float]` actually used |
| `n_samples`, `n_successful`, `n_local_failures` | `int` |
| `output_fidelity` | `float` |
| `mean_node_recall`, `mean_node_precision`, `mean_edge_recall`, `mean_edge_precision` | `float` |
| `mean_trace_coverage_score`, `mean_recombination_rate` | `float` |
| `mean_vote_confidence`, `mean_evidence_score_pred`, `mean_evidence_score_margin`, `mean_evidence_margin_pred_vs_competitor` | `float` |
| `mean_path_purity`, `mean_competitor_exposure`, `mean_explanation_confidence` | `float` (see gotcha) |
| `per_sample` | `pd.DataFrame` of every record, including failures |
| `local_accuracy` | `float`, present only when `y_true` was given |

# Gotchas

- **Three details keys are always `0.0`.** `path_purity`, `competitor_exposure` and
  `explanation_confidence` are read out of `sample_confidence` in the per-sample record, but
  `_compute_sample_confidence` never writes them. The `.get()` returns `None`, `_mean_records`
  falls back to `0.0`, so `mean_path_purity`, `mean_competitor_exposure` and
  `mean_explanation_confidence` are constant zeros. They do not affect the composite.
- **`perc_var` filtering drives the structural metrics.** `explain_local` always re-traces the raw
  trees, then checks each node/edge against the built graph. Filtering removes low-frequency
  variants or edges, so those checks fail: `graph_path_valid` goes `False` per tree,
  `all_trees_valid` goes `False`, `path_confidence` drops, `node_recall` / `edge_recall` drop and
  therefore `trace_coverage_score` drops. This is filtering working as intended, not a bug.
- **Recombination is the opposite failure mode.** Because node identity is the predicate text
  (sha1 of the label), paths from different trees merge in the graph, and the explanation can walk
  an edge no single tree ever executed. That inflates `recombination_rate`, which the composite
  penalizes via `1 - mean_recombination_rate`.
- `decimal_threshold` interacts with both: coarser rounding merges more labels (more recombination,
  higher coverage), finer rounding splits them (less recombination, more missing nodes).
- Precision metrics (`node_precision`, `edge_precision`) are reported but **not** in the composite.
- `output_fidelity` compares normalized strings, so `target_names` must line up positionally with
  `model.classes_` for the mapping in `_normalize_prediction_label` to apply; otherwise both sides
  fall back to `str(value)`.
- Cost: each sample triggers a full `explain_local` (one walk per estimator) plus a second full
  re-trace inside `_compute_trace_diagnostics`. Use `max_samples` on large datasets.

# Examples

```python
score = explainer.evaluate_faithfulness(X_test, max_samples=100)

details = explainer.evaluate_faithfulness(
    X_test, y_test, max_samples=100, return_details=True,
    weights={
        "output_fidelity": 0.50,
        "trace_coverage": 0.20,
        "anti_recombination": 0.20,
        "evidence_margin": 0.10,
    },
)
details["output_fidelity"], details["mean_recombination_rate"]
details["per_sample"].query("not matches_model")
```
