---
type: Playbook
title: Running the DPG CLI entrypoints
description: How to run examples/run_dpg_standard.py and examples/run_dpg_custom.py, what flags they accept, what they write to disk, and why the declared dpg console script does not work.
resource: https://github.com/viniferraria/DPG/blob/main/examples/run_dpg_standard.py
tags: [cli, playbook, examples, argparse, packaging]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# When to use

Use these two scripts for a one-shot batch run: train an ensemble, dump an evaluation report and
node/graph metric files, and optionally render the DPG. Both are thin argparse wrappers around
`test_dpg` — see [/modules/dpg-sklearn-dpg.md](/modules/dpg-sklearn-dpg.md). For programmatic or
notebook use go through `DPGExplainer` instead (`examples/quickstart.py`).

Both scripts load `config.yaml` from the **repo root** (resolved from `__file__`, not the CWD) and
read `dpg.default.perc_var`, `dpg.default.decimal_threshold`, `dpg.default.n_jobs`. A missing file
raises `FileNotFoundError`, invalid YAML raises `yaml.YAMLError`.

# Steps

## 1. `examples/run_dpg_standard.py` — sklearn datasets or any CSV

| Flag | Type | Default | Meaning (verbatim help) |
|---|---|---|---|
| `--ds` / `--dataset` | str | `iris` | Basic dataset to be analyzed |
| `--l` / `--n_learners` | int | `5` | Number of learners for the Random Forest |
| `--model_name` | str | `RandomForestClassifier` | Chosen tree-based ensemble model |
| `--dir` | str | `examples/results` | Directory to save results |
| `--plot` | flag | `False` | Plot the DPG, add the argument to use it as True |
| `--save_plot_dir` | str | `examples/results` | Directory to save the plot image |
| `--attribute` | str | `None` | A specific node attribute to visualize |
| `--communities` | flag | `False` | Boolean indicating whether to visualize communities |
| `--clusters` | flag | `False` | Boolean indicating whether to visualize clusters |
| `--threshold_clusters` | float | `None` | Threshold for detecting ambiguous nodes in clusters |
| `--t` | int | `None` | Override decimal_threshold from config |
| `--class_flag` | flag | `False` | Boolean indicating whether to highlight class nodes |
| `--seed` | int | *(none — resolves to `None`)* | Randomicity control |
| `--pv` | float | `None` | Override perc_var from config |

The `--dir` / `--save_plot_dir` defaults are `os.path.join(SCRIPT_DIR, "results")`, i.e. absolute
paths under `examples/`, independent of the CWD.

```bash
# smallest useful run
uv run python examples/run_dpg_standard.py --ds iris --l 5 --seed 42

# with a plot, class highlighting and explicit output dirs
uv run python examples/run_dpg_standard.py \
  --ds wine --l 10 --model_name RandomForestClassifier \
  --dir results/wine --save_plot_dir results/wine \
  --plot --class_flag --pv 0.001 --t 2 --seed 160898

# clustering extras
uv run python examples/run_dpg_standard.py \
  --ds iris --l 5 --dir results/iris \
  --clusters --threshold_clusters 0.2 --seed 160898
```

Writes into `--dir` (created with `os.makedirs(exist_ok=True)`):

- `{ds}_l{l}_seed{seed}_stats.txt` — accuracy / F1 / confusion matrix / classification report
- `{ds}_l{l}_seed{seed}_node_metrics.csv`
- `{ds}_l{l}_seed{seed}_dpg_metrics.txt` — one `key: value` line per graph metric

With `--clusters`, three more files keyed by `t{threshold_clusters}`:
`..._dpg_clusters_count.csv`, `..._dpg_clusters_intervals.csv`, `..._dpg_clusters.txt` (clusters,
probabilities, and rounded confidence intervals). Feature intervals are parsed out of the predicate
labels with the regex `([a-zA-Z0-9_]+)\s*([<=|>]+)\s*([-+]?[\d.]+)`.

With `--plot`, the image goes to `--save_plot_dir` as
`{ds}_{model_name}_l{l}_pv{pv}_t{t}_{seed}` (e.g. `iris_RandomForestClassifier_l5_pv0.001_t2_None.png`).

## 2. `examples/run_dpg_custom.py` — CSV with an explicit target column

| Flag | Type | Default | Meaning (verbatim help) |
|---|---|---|---|
| `--dataset` | str | **required** | Basic dataset to be analyzed |
| `--target_column` | str | `None` | Name of the column to be used as the target variable |
| `--n_learners` | int | `5` | Number of learners for the Ensemble model |
| `--model_name` | str | `RandomForestClassifier` | Chosen tree-based ensemble model |
| `--dir` | str | `examples/results` | Directory to save results |
| `--plot` | flag | `False` | Plot the DPG, add the argument to use it as True |
| `--save_plot_dir` | str | `examples/results` | Directory to save the plot image |
| `--attribute` | str | `None` | A specific node attribute to visualize |
| `--communities` | flag | `False` | Boolean indicating whether to visualize communities |
| `--class_flag` | flag | `False` | Boolean indicating whether to highlight class nodes |

There is no `--pv`, `--t`, `--seed`, `--clusters` or `--threshold_clusters` here; `perc_var` /
`decimal_threshold` / `n_jobs` come from `config.yaml` only.

```bash
uv run python examples/run_dpg_custom.py \
  --dataset datasets/custom.csv --target_column label \
  --n_learners 5 --dir results/custom --plot
```

Intended outputs in `--dir`: `custom_l{n}_pv{pv}_t{t}_stats.txt`,
`custom_l{n}_pv{pv}_t{t}_node_metrics.csv`, `custom_l{n}_pv{pv}_t{t}_dpg_metrics.txt`.

## 3. Other runnable scripts in `examples/`

| Script | One-line purpose |
|---|---|
| `quickstart.py` | 5-fold CV RandomForest on `datasets/custom.csv`, then `DPGExplainer.explain_global` + plots. |
| `quickstart_iris.py` | Same flow but downloads Iris from HuggingFace into `datasets/iris/` and caches it. |
| `local_explanation_iris.py` | Trains a 5-tree forest, calls `explain_local` on sample 0, prints votes/confidence, renders `iris_local_sample0.png`. |
| `generate_doc_images.py` | Regenerates `docs/_static/quickstart/` and `docs/_static/visualization/` images. |
| `generate_all_vis_images.py` | Regenerates the full `docs/_static/visualization/` image set. |
| `render_plot_gallery.py` | Renders every plot function into the docs plot gallery. |

```bash
uv run python examples/quickstart_iris.py
uv run python examples/local_explanation_iris.py
```

# Gotchas

- **The declared console script is broken — verified.** `pyproject.toml` contains
  `scripts = { "dpg" = "scripts.run_dpg_standard:main" }`, but `ls scripts` returns
  *No such file or directory*: there is no `scripts/` package (`[tool.poetry].packages` only includes
  `dpg` and `metrics`). Even if the path resolved, `examples/run_dpg_standard.py` defines **no
  `main()`** — all of its logic lives inline under `if __name__ == "__main__":`. Always invoke the
  scripts by path with `uv run python examples/...`; never rely on a `dpg` executable.
- **`run_dpg_custom.py` currently crashes.** It does `df, df_dpg_metrics = test.test_dpg(...)` while
  `test_dpg` returns a 6-tuple. Reproduced end-to-end: the run trains, builds the graph, writes
  `*_stats.txt`, then dies with `ValueError: too many values to unpack (expected 2)` at line 47 —
  before any CSV/metrics file is written. `run_dpg_standard.py` unpacks all six and works.
- **`--pv` and `--t` only rename the plot.** `test_dpg` never forwards them to
  `DecisionPredicateGraph`, which re-reads `config.yaml` relative to the CWD. To actually change the
  graph, edit `config.yaml` or use `DPGExplainer(dpg_config=...)`.
- `--seed` has no default, so omitting it yields `random_state=None` and literal `seedNone` filenames
  (`examples/results/iris_l5_seedNone_node_metrics.csv` is a committed example of this).
- `--model_name GradientBoosting*` / `AdaBoost*` raise `TypeError`: `test_dpg` always passes
  `n_jobs`, which those estimators do not accept.
- `--communities` and `--clusters` are mutually exclusive in effect — `communities=True` routes to
  `plot_dpg_communities` and the `clusters` arguments are ignored by that renderer.
- Plotting needs the Graphviz `dot` binary on `PATH` (`brew install graphviz`).
