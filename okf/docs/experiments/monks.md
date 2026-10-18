---
type: Experiment Suite
title: MONK's problems
description: Runs the DPG causal-accuracy protocol against the UCI MONK's benchmark, whose target concepts are exactly known, using the benchmark's canonical train/test split.
resource: https://github.com/viniferraria/DPG/tree/main/experiments/monks
tags: [experiments, monks, uci, benchmark, causal, one-hot, centrality]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Purpose

The MONK's problems are the classic "known concept" benchmark: six nominal attributes `a1..a6` and a
boolean target defined by a written-down rule, so the causally relevant attributes are ground truth
rather than an estimate. This suite reuses the causal-accuracy protocol of the
[synthetic scenarios](/experiments/causal-synthetic-scenarios.md) on real benchmark data — rank DPG
predicate nodes by centrality, take the top 3 attributes, score precision and recall against the rule.

`SCENARIO_GROUND_TRUTH` in `run_monk.py` enumerates only two of the three problems:

| Train / test | Concept (from the runner's comments) | Ground-truth attributes |
|---|---|---|
| `datasets/monks-1.train` / `datasets/monks-1.test` | `a1 = a2 OR a5 = 1` | `a1`, `a2`, `a5` |
| `datasets/monks-3.train` / `datasets/monks-3.test` | `(a5 = 3 AND a4 = 1) OR (a5 != 4 AND a2 != 3)` | `a5`, `a4`, `a2` |

The runner deliberately uses the benchmark's canonical split rather than resampling: each `.test` file
is the complete 432-row attribute space and each `.train` file is a labelled subset of it. Accuracy over
the full space therefore measures concept recovery, and it preserves the monks-3 design where roughly 5%
label noise lives in `.train` only. There is exactly one split per scenario, recorded as `split_idx = 0`.

# Layout

```
experiments/monks/
├── run_monk.py                    # the only runner; no argparse, main() at the bottom
├── datasets/                      # UCI MONK's distribution, copied verbatim
│   ├── monks-1.train  (124 rows)  ├── monks-1.test  (432 rows)
│   ├── monks-2.train  (169 rows)  ├── monks-2.test  (432 rows)   # present, never used
│   ├── monks-3.train  (122 rows)  ├── monks-3.test  (432 rows)
│   ├── monks.names                # attribute domains and the three target concepts
│   ├── Index                      # UCI directory listing
│   ├── update                     # UCI changelog
│   ├── thrun.comparison.dat       # the Thrun et al. algorithm comparison
│   └── thrun.comparison.ps.Z      # same, compressed PostScript
├── results/                       # empty in the tree; run output lands here
├── states/                        # created by main(); pickled explainers/explanations
└── dpg_explainer_<timestamp>.log  # emitted per run into the CWD
```

Rows are whitespace-delimited with a leading space; `load_dataset` reads them with
`sep=r"\s+", header=None, names=["class","a1"..."a6","id"]`, drops `id`, and splits `class` off as `y`.
A literal `" "` separator would turn the leading field into an all-NaN index — the regex avoids that.

# How to run

Dataset paths are `datasets/...` relative to the process CWD, so run from the suite directory:

```bash
cd experiments/monks
uv run python run_monk.py
```

No CLI flags exist. The knobs are module constants: `RANDOM_STATE = 42`,
`ATTRIBUTES = ["a1".."a6"]`, and `MODEL_FACTORIES = {"RandomForest", "ExtraTrees"}` (both
`random_state=42`, otherwise sklearn defaults). `TOP_K = 3` and `METRICS` are not defined here any more:
they live in the shared [experiments.causal_accuracy](/modules/experiments-causal-accuracy.md) module.
`main()` creates `states/` and calls `run_experiments_with_ground_truth(SCENARIO_GROUND_TRUTH,
OUTPUT_PATH, logger)`.

Like the causal suite, `DPGExplainer` is constructed with
`config_file=str(CONFIG_PATH.resolve(strict=True))` where `CONFIG_PATH = <suite>/../../config.yaml` —
the repo-root file, located from `__file__` rather than the CWD, so the run gets `perc_var=0.0001`,
`decimal_threshold=3`, `n_jobs=-1`. See [/conventions/config-resolution.md](/conventions/config-resolution.md);
`strict=True` makes a missing `config.yaml` a hard failure at explainer construction.

`pre_process_dataset` one-hot encodes `a1..a6` across the train and test frames **in one
`pd.get_dummies` call** (`dtype=bool`, `drop_first=False`), so both end up with an identical column set.
Only the category vocabulary is shared — the target never enters, so this leaks nothing.

# Scoring

The reference protocol, through the shared module: `causal_accuracy_rows(explanation.node_metrics,
gt_features, run_key, split_idx, feature_name=base_feature_name, drop_duplicates=True)`.
Named deviations (both forced by the data, both call-site keywords):

- `feature_name=base_feature_name`: one-hot columns are mapped back to their source attribute
  (`a1_2` to `a1`, only a trailing `_<digits>` is stripped).
- `drop_duplicates=True`: features are de-duplicated *before* `head(TOP_K)`, so the result is three
  distinct attributes. The reference keeps repeats (`F2;F2;F2`); MONK's does not, which makes its
  precision/recall not directly comparable with the causal-synthetic history.

`Class` and `Pred` labels are both excluded (MONK's is a classifier, so `Pred` never occurs; before
2026-10-17 only `Class` was excluded here).

# Outputs

Both under `results/`, written by `append_csv` (which creates the directory):

- `results/node_metrics_<timestamp>.csv` — the reference schema (`experiment, split, node, degree,
  in_degree, out_degree`, every `NodeMetrics` column snake-cased in frame order, `node_idx, label,
  processing_time`). A new `NodeMetrics` column flows through with no edit. Appended per (scenario, model),
  header only when the file is new, so the four runs share one file. The columns used to be `node_index`,
  `in_degree_nodes`, `out_degree_nodes`, `split_idx` with `experiment`/`split_idx` after `label`; old files
  keep that layout, see [node_metrics schema change](/side-effects/node-metrics-schema-unification.md).
- `results/causal_accuracy_<timestamp>.csv` — `experiment, split, k, ground_truth,
  explanation_top_features, metric, intersection, precision, recall`, list fields `;`-joined and
  `intersection` sorted. One path per run, sharing its timestamp with `node_metrics_<timestamp>.csv`
  (before 2026-10-13 each call recomputed `timestamp()` and scattered a run across several tiny files).
- Both files are written with LF line endings ([why](/side-effects/csv-line-endings-lf.md)).
- `states/<monks-N>_<Model>_split0_dpg_explainer_<timestamp>.pkl` and
  `states/<monks-N>_<Model>_split0_explanation_<timestamp>.pkl` — e.g.
  `monks-1_RandomForest_split0_explanation_2026-08-02T19-22-44.pkl`. `run_key` is
  `f"{Path(train_file).stem}_{model_name}"`, so the stem keeps the hyphen.
- `dpg_explainer_<timestamp>.log` in the CWD, timestamps UTC `%Y-%m-%dT%H-%M-%S`.

Everything in `results/`, `states/`, and the `dpg_explainer_*.log` files is accumulated run output, not
source. Nothing is cleaned or overwritten between runs, so both directories grow monotonically.

# Test coverage

`tests/test_run_monk.py` is minimal. The suite has no `__init__.py`, so the test loads `run_monk.py`
by file path (`importlib.util.spec_from_file_location`) in a module-scoped fixture, after `chdir` to a
temp dir because importing the module opens a log file in the CWD. It checks that every shared `METRICS`
name is a column of the real `NodeMetrics.extract_node_metrics` output (a missing one would raise a
`KeyError` swallowed by the per-run `except Exception`), and, with `causal_accuracy_rows` and the dataset
helpers monkeypatched, that `run_experiments_with_ground_truth` passes `feature_name=base_feature_name` and
`drop_duplicates=True` and that `base_feature_name("a5_2") == "a5"`. Scoring and CSV schema are covered once
in `tests/test_causal_accuracy.py`. `load_dataset` and `pre_process_dataset` are untested. `run_monk.py` is
tracked in git.

# Gotchas

- `Collective influence` is always computed at radius ℓ = 2: `DPGExplainer` calls
  `NodeMetrics.extract_node_metrics` without `ci_radius` and no config key sets it. It is
  integer-valued with many ties at `0`, so its top-k order among tied nodes is arbitrary. `Local
  clustering coefficient` is saved in the node CSV but deliberately not ranked (same tie problem).
- **monks-2 ships but is never run.** `monks-2.train`/`.test` sit in `datasets/` and are absent from
  `SCENARIO_GROUND_TRUTH`. `monks.names` gives its concept as "EXACTLY TWO of {a1 = 1, ..., a6 = 1}",
  which implicates all six attributes — so a top-3 ground-truth set would be ill-defined.
- **Predicate labels are one-hot column names.** After encoding, a node reads `a1_2 <= 0.5`.
  `extract_top_k_features` takes the token before the first space, maps it back with
  `feature_name=base_feature_name`, then (`drop_duplicates=True`) de-duplicates *before* `head(top_k)` —
  so the result is 3 distinct source attributes, comparable to a ground truth written in raw `aN` names.
  The reference runner passes neither keyword.
- **The module-level `logger` shadows the parameter.** `logger` is assigned at import time (line ~123),
  which means importing `run_monk` immediately creates a `dpg_explainer_<timestamp>.log` file as a side
  effect. `_build_explanation` uses that module-level logger; `run_experiments_with_ground_truth`
  takes a `logger` argument that shadows it locally, and `main()` passes a second, separately created
  one. Two log files per invocation is the normal outcome.
- **Failures are swallowed per run.** The explanation block is wrapped in `try/except Exception`, which
  logs and prints a traceback but continues — and the accuracy line is logged *after* the `except`, so
  a scenario that failed to explain still reports its model accuracy.
- **`y_train` is used for `target_names`** (`np.unique(y_train)`), unlike the causal runner which uses
  the full `y`. With MONK's balanced binary labels this is equivalent in practice, but the two runners
  do differ.
- The node-metrics CSV schema follows whatever [`DPGExplainer.explain_global`](/modules/dpg-explainer.md)
  currently emits: `node_metrics_rows` writes every non-fixed `NodeMetrics` column in frame order, with no
  dataclass pinning it (the causal suite's `NodeMetricRecord` is gone). A new metric is exported
  automatically but scored only once added to `METRICS`.
