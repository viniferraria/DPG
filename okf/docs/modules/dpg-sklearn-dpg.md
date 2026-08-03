---
type: Python Module
title: dpg.sklearn_dpg — dataset selection and train/evaluate/report glue
description: Convenience layer that loads a standard sklearn dataset or a CSV, trains a tree ensemble, writes an evaluation report, and returns node/edge/graph metrics for the resulting DPG.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/sklearn_dpg.py
tags: [cli, glue, datasets, sklearn, metrics, reporting]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`dpg/sklearn_dpg.py` (249 lines) is the script-facing glue behind the two CLIs in `examples/`. It
does one end-to-end run: pick a dataset, split it, fit an ensemble, write a text evaluation report,
build a `DecisionPredicateGraph`, and return the metric frames. It is *not* re-exported from
`dpg/__init__.py`; callers import the module directly (`import dpg.sklearn_dpg as test`).

It sits above [/modules/dpg-core.md](/modules/dpg-core.md) and pulls `NodeMetrics`, `EdgeMetrics`,
`GraphMetrics` from `metrics/`, plus `plot_dpg` / `plot_dpg_communities` from `dpg.visualizer`. The
CLI wrappers that call it are documented in [/workflows/cli-entrypoints.md](/workflows/cli-entrypoints.md).

# API

Two public functions, both module-level.

| Function | Signature | Returns |
|---|---|---|
| `select_dataset` | `select_dataset(source: str, target_column: str \| None = None) -> tuple[Any, Any, Any]` | `(data, features, target)` |
| `test_dpg` | `test_dpg(datasets, target_column=None, n_learners=5, perc_var=0.00000001, decimal_threshold=3, n_jobs=-1, model_name='RandomForestClassifier', file_name=None, plot=False, save_plot_dir="examples/", attribute=None, communities=False, clusters_flag=False, threshold_clusters=None, class_flag=False, seed=160898) -> tuple[Any, ...]` | 6-tuple, see below |

## `select_dataset`

`source` is matched first against a dict of eagerly-loaded sklearn bunches:

| Key | Loader |
|---|---|
| `iris` | `load_iris()` |
| `diabetes` | `load_diabetes()` |
| `digits` | `load_digits()` |
| `wine` | `load_wine()` |
| `cancer` | `load_breast_cancer()` |

On a hit it returns `dataset.data, dataset.feature_names, dataset.target`. Otherwise `source` is
read as a comma-separated CSV via `pd.read_csv(source, sep=',')`; failure raises
`DPGDatasetError.load_failed(source, e)`. When `target_column` is `None` the **last column** is used
(printed as `[INFO] Using last column as target: ...`); a name not present in the frame raises
`DPGDatasetError.missing_target_column`. Remaining columns are coerced with
`pd.to_numeric(errors='coerce')`, `±inf → NaN`, `fillna(df.mean())`, then
`np.round(df.values, 2).astype(np.float64)`.

## `test_dpg`

Returns `(df, df_edges, df_dpg, clusters, node_prob, confidence)`:

| Position | Value |
|---|---|
| `df` | `NodeMetrics.extract_node_metrics(dpg_model, nodes_list)` |
| `df_edges` | `EdgeMetrics.extract_edge_metrics(dpg_model, nodes_list)` |
| `df_dpg` | `GraphMetrics.extract_graph_metrics_lpa(...)` with `target_names=np.unique(y_train).astype(str).tolist()` |
| `clusters`, `node_prob`, `confidence` | `GraphMetrics.clustering(dpg_model, class_nodes, threshold_clusters)` when `clusters_flag=True`, else all `None` |

Supported `model_name` strings (anything else raises `DPGModelError.unsupported_model`):
`RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`,
`GradientBoostingRegressor`, `ExtraTreesClassifier`, `AdaBoostClassifier`, `AdaBoostRegressor`,
`BaggingClassifier`.

Flow: validate `n_learners > 0` (`DPGValidationError.positive_learner_count`) → `select_dataset` →
`train_test_split(test_size=0.3, random_state=seed)` → instantiate
`model(n_estimators=n_learners, random_state=seed, n_jobs=n_jobs)` → `fit` / `predict` → report →
`DecisionPredicateGraph(...).fit(X_train)` → `to_networkx` → metrics → optional plot.

The report at `file_name` (parent dirs are created) is classifier- or regressor-shaped:

- `is_classifier(model)`: `accuracy_score`, weighted `f1_score`, `confusion_matrix` (written with
  `np.savetxt(..., fmt='%d')`), and the full `classification_report`.
- otherwise: `mean_squared_error` only, written as `MSE: {:.2f}`.

# Gotchas

- **`perc_var`, `decimal_threshold` and `n_jobs` are not forwarded to the graph.**
  `DecisionPredicateGraph` is constructed with only `model`, `feature_names`, `target_names`, so it
  resolves those values itself from `config.yaml` relative to the CWD (see
  [/modules/dpg-core.md](/modules/dpg-core.md)). Inside `test_dpg` the three arguments only reach the
  **plot filename**: `f"{plot_name}_{model_name}_l{n_learners}_pv{perc_var}_t{decimal_threshold}_{seed}"`.
  Passing `--pv`/`--t` therefore renames the PNG without changing the graph.
- **`n_jobs` breaks half the model table.** `GradientBoostingClassifier`, `GradientBoostingRegressor`,
  `AdaBoostClassifier` and `AdaBoostRegressor` have no `n_jobs` parameter, so the unconditional
  `model_classes[model_name](n_estimators=..., random_state=..., n_jobs=n_jobs)` raises `TypeError`
  for those four names. Only the RandomForest, ExtraTrees and Bagging entries are actually reachable.
- **Two different return arities.** If `len(nodes_list) < 2` the function prints
  `"Warning: Insufficient nodes for DPG analysis"` and returns `None, None` — a 2-tuple where the
  happy path returns 6. Callers that unpack unconditionally will crash on one branch or the other.
- The five sklearn loaders in `select_dataset` are executed on **every call**, because the dict
  literal calls `load_iris()`, `load_digits()`, … before the membership test.
- `class_nodes` is derived with a substring test (`'Class' in i[1]`), so any predicate label
  containing the word "Class" is treated as a leaf node.
- Regression datasets (`diabetes`) still default to `RandomForestClassifier`; pass
  `model_name='RandomForestRegressor'` explicitly.
- `df_dpg` uses `y_train`-derived target names while the graph itself was built with
  `np.unique(target)` over the *full* dataset — they diverge if a class is absent from the split.
- `save_plot_dir` defaults to the relative string `"examples/"`, so plots land relative to the CWD.

# Examples

```python
import dpg.sklearn_dpg as test

df, df_edges, df_dpg, clusters, node_prob, confidence = test.test_dpg(
    datasets="iris",
    n_learners=5,
    model_name="RandomForestClassifier",
    file_name="results/iris_stats.txt",
    plot=True,
    save_plot_dir="results",
    seed=42,
)
```

```python
from dpg.sklearn_dpg import select_dataset

data, features, target = select_dataset("datasets/custom.csv", target_column="label")
```
