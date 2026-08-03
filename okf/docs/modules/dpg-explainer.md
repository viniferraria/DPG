---
type: Python Module
title: dpg.explainer
description: High-level DPGExplainer API that builds a DPG, produces global and local explanations, evaluates faithfulness, and wraps the visualizer plot functions.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/explainer.py
tags: [python, module, explainer, api, local-explanation, global-explanation]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`dpg/explainer.py` is the user-facing entry point of the library. It wraps
[`DecisionPredicateGraph`](/modules/dpg-core.md) (graph construction), the `metrics/` package
(`NodeMetrics`, `EdgeMetrics`, `GraphMetrics`), and [the visualizer](/modules/dpg-visualizer.md)
into a single object with a fit/explain/plot workflow.

The module also defines the three result dataclasses documented in
[explanation dataclasses](/concepts/explanation-dataclasses.md) and the
[faithfulness evaluation](/concepts/faithfulness-evaluation.md) routine.

All four public names (`DPGExplainer`, `DPGExplanation`, `DPGLocalExplanation`,
`DPGTreePathExplanation`) are re-exported from `dpg/__init__.py`.

# API

## Construction and state

| Member | Signature / default | Meaning |
|---|---|---|
| `__init__` | `(model, feature_names, target_names=None, config_file="config.yaml", dpg_config=None)` | Forwards everything to `DecisionPredicateGraph`; `feature_names`/`target_names` are copied to `list`. |
| `builder` | property → `DecisionPredicateGraph` | The wrapped builder (holds `model`, `feature_names`, `target_names`, `decimal_threshold`, `perc_var`). |
| `fit(X)` | → `DPGExplainer` | Calls `builder.fit(X)` to get the `dot`, then `builder.to_networkx(dot)` for `(graph, nodes)`. Clears cached node/edge metrics and sets `_is_fitted`. |

Internal caches (`_get_node_metrics`, `_get_node_metrics_lookup`, `_get_edge_metrics`) are lazy and
invalidated on every `fit`. `_get_node_metrics_lookup` keys node-metric rows by the `"Node"` column
(the sha1-derived node id).

## Explanation entry points

| Method | Signature | Returns |
|---|---|---|
| `explain_global` | `(X=None, communities=False, community_threshold=0.2)` | `DPGExplanation`. Fits first if `X` is given. Computes node metrics, `EdgeMetrics.extract_edge_metrics`, `GraphMetrics.extract_class_boundaries` and, when `communities=True`, `GraphMetrics.extract_communities`. `community_threshold` is stored on the result only when communities were computed. |
| `explain_local` | `(sample, sample_id=0, X=None, validate_graph=True)` | `DPGLocalExplanation`. |
| `local_path_dataframe` | `(local_explanation)` | `pd.DataFrame`, one row per path step, columns: `sample_id, tree_index, step_index, label, node_id, is_leaf, predicate_true, edge_exists_from_prev, starts_from_root, ends_in_leaf, graph_path_valid, mean_lrc, mean_bc, path_confidence`. Paths sorted by `tree_index`; `edge_exists_from_prev` is `True` at `step_index == 0`. |
| `evaluate_faithfulness` | `(X, y_true=None, max_samples=None, weights=None, return_details=False, sample_ids=None)` | `float` composite, or a details dict — see [faithfulness evaluation](/concepts/faithfulness-evaluation.md). |

Not-fitted calls raise the corresponding `DPGNotFittedError` factory
(`for_graph`, `for_nodes`, `for_global_explanation`, `for_local_explanation`, `for_local_plot`,
`for_faithfulness`).

## Plot wrappers (delegate to `dpg/visualizer.py`)

Each of these resolves a `DPGExplanation` first (calling `explain_global()` itself when
`explanation is None`) and then forwards to the matching visualizer function.

| Explainer method | Delegates to |
|---|---|
| `plot` | `plot_dpg` |
| `plot_communities` | `plot_dpg_communities` (re-runs `explain_global(communities=True, ...)` if `explanation.communities is None`) |
| `plot_local_on_dpg` | `plot_dpg_local_paths_aggregate` |
| `plot_lrc_importance` | `plot_lrc_vs_rf_importance` |
| `plot_top_lrc_splits` | `plot_top_lrc_predicate_splits` |
| `plot_class_feature_complexity` | `class_feature_predicate_counts` → `plot_class_feature_complexity` |
| `plot_sample_using_bc_weights` | `plot_sample_using_bc_weights` |
| `plot_class_bounds_vs_dataset_ranges` | `plot_dpg_class_bounds_vs_dataset_feature_ranges` |
| `class_feature_predicate_counts` | `class_feature_predicate_counts` (returns the matrix, no plot) |
| `sample_bc_weights` | `sample_bc_weights` (returns weights, no plot) |

Shared styling defaults across the graph plotters: `save_dir="results/"`, `fig_size=(16, 8)`,
`dpi=300`, `pdf_dpi=600`, `show=True`, `export_pdf=False`, `theme="dpg"`, `palette="default"`.
`plot` uses `label_mode="full"`, `readability="normal"`, `class_flag=False`; `plot_communities` and
`plot_local_on_dpg` use `label_mode="wrapped"`, `readability="presentation"`, `class_flag=True`.

# Behavior

## `explain_local`

1. Optionally `fit(X)`; raises `DPGValidationError.sample_feature_count` if the flattened sample
   length does not match `len(builder.feature_names)`.
2. Builds `node_lookup = {label: node_id}` from the fitted node list.
3. For every `tree` in `builder.model.estimators_` (already flattened by
   `SklearnEnsembleNormalizer` inside the builder), `_trace_tree_path` walks the raw `tree_` arrays
   following the sample, appending predicate labels and finishing with a leaf label.
4. Leaf labels starting with `"Class "` are stripped of that prefix and counted into `class_votes`;
   `majority_vote` is `class_votes.most_common(1)[0][0]`.
5. `_compute_sample_confidence` produces the `sample_confidence` dict.
6. `path_mode` on the result is always the literal `"execution_trace"`.

`_trace_tree_path` details:

- Regressor leaves (`RandomForestRegressor`, `ExtraTreesRegressor`, `AdaBoostRegressor`) produce
  `f"Pred {round(value, 2)}"`; classifier leaves go through `_leaf_class_label`, which honors the
  GradientBoosting per-class column layout (`SklearnEnsembleNormalizer.get_tree_class_index`) and
  the binary sign-of-leaf-score case, then maps the index through `target_names` or `model.classes_`.
- Thresholds are rounded with `builder.decimal_threshold`, matching the graph's label contract.
- `node_ids` on the returned path are the *graph* ids looked up by label when
  `validate_graph=True`, and the locally recomputed sha1 ids otherwise. Validity checks
  (`edge_exists`, `graph_path_valid`) always use the natively recomputed ids.
- `path_confidence = (node_coverage + edge_coverage) / 2`, where `node_coverage` is the fraction of
  path labels present in the node-metrics lookup and `edge_coverage` is the fraction of consecutive
  pairs present as graph edges (`1.0` when the path has no edges).
- `predicate_truths` collects one `True` per internal split taken — by construction the traced
  branch is the true one, so this list contains only `True` values.

# Gotchas

- **Label formats are load-bearing.** `_label_to_node_id` is
  `str(int(hashlib.sha1(label.encode()).hexdigest(), 16))`, so a traced label only maps onto a graph
  node when the string is byte-identical to the one `generate_dot` emitted. See
  [the label contract](/conventions/label-contract.md).
- `explain_local` re-traces raw trees; it does not read paths out of the DFG. Aggressive `perc_var`
  filtering therefore makes `graph_path_valid` / `all_trees_valid` go `False` legitimately.
- `class_votes` and `majority_vote` hold *normalized* names (`"0"`), while
  `tree_paths[*].labels` keep the raw DPG labels (`"Class 0"`).
- `plot_local_on_dpg` will build a local explanation itself if given only `sample`; passing neither
  `local_explanation` nor `sample` raises `DPGValidationError.missing_local_input()`, and an
  out-of-range entry in `path_indices` raises `DPGValidationError.invalid_path_indices()`.
- `_get_node_metrics` uses `self._graph` / `self._nodes` directly rather than the `_require_*`
  guards, so it relies on callers having checked `_is_fitted` first.

# Examples

```python
from dpg import DPGExplainer

explainer = DPGExplainer(model, feature_names, target_names, dpg_config=cfg).fit(X)

expl = explainer.explain_global(communities=True, community_threshold=0.2)
explainer.plot("iris", explanation=expl, show=False)

local = explainer.explain_local(X[0], sample_id=0)
steps = explainer.local_path_dataframe(local)

report = explainer.evaluate_faithfulness(X, y, max_samples=50, return_details=True)
```
