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

No CLI flags exist. The knobs are module constants: `RANDOM_STATE = 42`, `TOP_K = 3`,
`ATTRIBUTES = ["a1".."a6"]`, `MODEL_FACTORIES = {"RandomForest", "ExtraTrees"}` (both
`random_state=42`, otherwise sklearn defaults), and
`METRICS = ["Local reaching centrality", "Closeness centrality", "Harmonic centrality"]`.
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

# Outputs

Both under `results/`, created by `write_records_to_csv` / `write_causal_accuracy_to_csv`:

- `results/node_metrics_<timestamp>.csv` — one path per run; every column of
  `explanation.node_metrics` is carried through verbatim, plus `node_index` and the fixed columns
  `experiment`, `split_idx`, `processing_time`. Column names are snake-cased by `to_snake_case`
  ("In degree nodes" → `in_degree_nodes`). Written with `mode="a"` and
  `header=not output_path.exists()`, so the four (scenario × model) runs append into one file.
- `results/causal_accuracy_<timestamp>.csv` — `experiment, split, k, ground_truth,
  explanation_top_features, metric, intersection, precision, recall`, list fields `;`-joined.
  `write_causal_accuracy_to_csv` recomputes `timestamp()` on every call, so one run scatters its rows
  across several tiny files rather than one.
- `states/<monks-N>_<Model>_split0_dpg_explainer_<timestamp>.pkl` and
  `states/<monks-N>_<Model>_split0_explanation_<timestamp>.pkl` — e.g.
  `monks-1_RandomForest_split0_explanation_2026-08-02T19-22-44.pkl`. `run_key` is
  `f"{Path(train_file).stem}_{model_name}"`, so the stem keeps the hyphen.
- `dpg_explainer_<timestamp>.log` in the CWD, timestamps UTC `%Y-%m-%dT%H-%M-%S`.

Everything in `results/`, `states/`, and the `dpg_explainer_*.log` files is accumulated run output, not
source. Nothing is cleaned or overwritten between runs, so both directories grow monotonically.

# Test coverage

**None.** No file under `tests/` imports `experiments.monks` or `run_monk`; the suite has no
`__init__.py`, so it is not importable as a package the way
[local_explanation](/experiments/local-explanation.md) is. The pure helpers are written to be testable
in isolation — `to_snake_case`, `timestamp`, `load_dataset`, `pre_process_dataset`,
`base_feature_name`, `extract_top_k_features`, `calculate_causal_accuracy`, `records_from_explanation`
— but nothing exercises them. Note also that the suite is **entirely untracked in git**
(`git ls-files experiments/monks` returns nothing), unlike the other two.

The closest analogue is `tests/test_run_dpg_causal_synthetic.py`, which covers the near-identical
helpers of the causal suite; any change shared between the two runners is only guarded there.

# Gotchas

- **monks-2 ships but is never run.** `monks-2.train`/`.test` sit in `datasets/` and are absent from
  `SCENARIO_GROUND_TRUTH`. `monks.names` gives its concept as "EXACTLY TWO of {a1 = 1, ..., a6 = 1}",
  which implicates all six attributes — so a top-3 ground-truth set would be ill-defined.
- **Predicate labels are one-hot column names.** After encoding, a node reads `a1_2 <= 0.5`.
  `extract_top_k_features` takes the token before the first space, maps it back with
  `base_feature_name` (strips a trailing `_<digits>` only), then `drop_duplicates(subset="Label")`
  *before* `head(top_k)` — so the result is 3 distinct source attributes, comparable to a ground truth
  written in raw `aN` names. This mapping step does not exist in the causal-synthetic runner.
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
- `explanation.node_metrics` columns are passed through untyped, so the node-metrics CSV schema follows
  whatever [`DPGExplainer.explain_global`](/modules/dpg-explainer.md) currently emits — there is no
  dataclass pinning it, in contrast to `NodeMetricRecord` in the causal suite.
