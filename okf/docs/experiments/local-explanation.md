---
type: Experiment Suite
title: Local explanation experiments
description: Parameter sweep that measures how often a DPG local explanation reproduces the model's prediction on built-in sklearn datasets, plus an analysis pass that aggregates the resulting CSVs.
resource: https://github.com/viniferraria/DPG/tree/main/experiments/local_explanation
tags: [experiments, local-explanation, sweep, fidelity, sklearn, cli]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Purpose

Quantifies local-explanation fidelity: for each test sample, does
[`DPGExplainer.explain_local`](/modules/dpg-explainer.md)'s `majority_vote` agree with the model's own
prediction (`local_matches_model`) and with the true label (`local_correct`)? Confidence signals from
`local.sample_confidence` — `vote_confidence`, `evidence_score_pred`, `evidence_score_margin`,
`trace_coverage_score`, `recombination_rate`, `num_paths`, `num_valid_paths` — are recorded alongside,
so the analysis pass can ask whether low confidence predicts disagreement. The sweep is a Cartesian
product over dataset × `n_estimators` × `max_depth` × `perc_var` × `decimal_threshold` ×
`graph_construction_mode` × `seed` (`iter_configs` uses `itertools.product`), with a fixed
`train_test_split(test_size=0.3, stratify=y)` and `RandomForestClassifier(n_jobs=1)`. From the source:

- `SUPPORTED_DATASETS = {"iris", "wine", "breast_cancer", "digits"}` (sklearn `load_*(as_frame=True)`).
- `SUPPORTED_GRAPH_CONSTRUCTION_MODES = {"aggregated_transitions", "execution_trace"}`, validated by
  `parse_graph_mode_values` up front — see
  [/conventions/graph-construction-modes.md](/conventions/graph-construction-modes.md).

# Layout

This is the only experiment suite that is a real Python package — the only one with an `__init__.py`,
and the only one whose contents are tracked in git in full:

```
experiments/local_explanation/
├── __init__.py                 # re-exports run_local_explanation_experiments;
│                               # __all__ = ["run_local_explanation_experiments"]
├── run_local_explanations.py   # sweep runner + argparse CLI (main(argv) -> int)
└── analyze_results.py          # aggregation CLI over summary.csv / per_sample.csv
```

No `datasets/`, `results/`, or `states/` is committed — datasets come from sklearn and `out_dir` is
created on demand by `os.makedirs(out_dir, exist_ok=True)`.

# How to run

Run the sweep (module form works because the package has `__init__.py`):

```bash
uv run python -m experiments.local_explanation.run_local_explanations \
  --datasets iris,wine --out_dir experiments/local_explanation/results \
  --n_estimators 5,10 --max_depth None,3 \
  --perc_var 1e-9 --decimal_threshold 6 \
  --graph_construction_mode execution_trace,aggregated_transitions \
  --seed 42 --max_test_samples 10
```

Real flags and defaults from `build_arg_parser()`:

| Flag | Default | Notes |
|---|---|---|
| `--datasets` | `iris` | comma-separated; validated against `SUPPORTED_DATASETS` in `main()` |
| `--out_dir` | `experiments/local_explanation/results` | relative to the CWD |
| `--n_estimators` | `5` | comma-separated ints |
| `--max_depth` | `None` | comma-separated; the literal string `None` maps to `None` |
| `--perc_var` | `1e-9` | comma-separated floats |
| `--decimal_threshold` | `6` | comma-separated ints |
| `--graph_construction_mode` | `execution_trace` | comma-separated; validated |
| `--seed` | `42` | comma-separated ints |
| `--max_test_samples` | `10` | `type=int`, single value, capped at `len(X_test)` |

Then aggregate — `--results_dir` and `--out_dir` are `required=True`, `--group_by` defaults to
`dataset,graph_construction_mode`:

```bash
uv run python -m experiments.local_explanation.analyze_results \
  --results_dir experiments/local_explanation/results \
  --out_dir experiments/local_explanation/analysis --group_by dataset,graph_construction_mode
```

Library use bypasses argparse — `run_local_explanation_experiments` accepts scalars or sequences for
every sweep axis (same defaults, unstringified) and returns `(summary_df, per_sample_df)`.

# Outputs

`run_local_explanation_experiments` always writes exactly two files into `out_dir`, overwriting them:

- `summary.csv` — one row per sweep config: `dataset, n_train, n_test_explained, n_estimators,
  max_depth, perc_var, decimal_threshold, graph_construction_mode, seed, model_accuracy,
  local_matches_model_rate, local_accuracy, avg_vote_confidence, avg_evidence_score_pred,
  avg_trace_coverage_score, avg_recombination_rate, avg_num_paths`.
- `per_sample.csv` — one row per explained sample: the config columns plus `sample_index, true_label,
  model_pred, local_pred, local_matches_model, local_correct, vote_confidence, evidence_score_pred,
  evidence_score_margin, trace_coverage_score, recombination_rate, num_paths, num_valid_paths`.

`analyze_results.py` writes `aggregate_summary.csv` (per-group `mean/std/count` over `SUMMARY_METRICS`,
columns flattened to `metric_mean`/`metric_std`/`metric_count`) and `cohort_summary.csv`, which buckets
samples into `correct_and_matches_model`, `correct_but_disagrees_model`, `wrong_but_matches_model`,
`wrong_and_disagrees_model` and reports `mean_*` metrics plus `n_samples`. `load_results` fails loudly
on a missing file, an empty frame, or a missing required column.

Like the other suites, running `DPGExplainer` emits a timestamped `dpg_explainer_<timestamp>.log` into
the CWD, and anything under `out_dir` is accumulated run output, not source.

# Test coverage

`tests/test_smoke.py` is the only test file touching this suite:

- `test_local_experiment_runner_smoke` — `from experiments.local_explanation import
  run_local_explanation_experiments`; runs iris with `n_estimators=3, max_depth=3, perc_var=1e-9,
  decimal_threshold=6, graph_construction_mode="execution_trace", seed=42, max_test_samples=3` into
  `tmp_path`, asserting both CSVs exist with the required column sets.
- `test_local_experiment_runner_parse_helpers` — `from
  experiments.local_explanation.run_local_explanations import parse_csv_values,
  parse_graph_mode_values, parse_optional_int_values`.
- `test_local_experiment_runner_sweep_smoke` — same entry point, list arguments; asserts
  `len(summary_df) == 4` for a 2×1×1×1×2×1 grid.
- `test_local_experiment_analysis_smoke` and
  `test_local_experiment_analysis_missing_columns_raise_clear_error` — `from
  experiments.local_explanation.analyze_results import aggregate_summary, build_cohort_summary,
  load_results`, with hand-built frames rather than runner output.

Not covered: either module's `main()`/`build_arg_parser()`, `run_single_dataset` in isolation,
`load_builtin_dataset` for `wine`/`breast_cancer`/`digits`, `build_explainer`. Only iris is ever run.

# Gotchas

- **`build_explainer` passes `dpg_config` explicitly**, hard-coding `n_jobs: 1` alongside the swept
  `perc_var` and `decimal_threshold`. That short-circuits the
  [config resolution chain](/conventions/config-resolution.md) — `config.yaml` is never read, so this
  suite alone behaves identically regardless of where it is launched.
- **The runner's default mode is `execution_trace`**, not the library default
  `aggregated_transitions` — a run left at defaults measures non-default library behavior.
- **`--max_test_samples` takes the *first* N test rows** (`X_test.iloc[:max_samples]`), not a random
  subset, so the cohort is whatever `train_test_split(..., stratify=y)` put first. `digits` is
  supported but expensive: 64 features, 1797 samples, `n_jobs=1`.
- **`out_dir` files are overwritten, not appended.** Two sweeps into the same directory leave only the
  second one's rows — unlike the causal and MONK's runners, which accumulate.
- `evidence_score_pred`/`evidence_score_margin` pass through `_maybe_float` and can be empty in the CSV;
  the summary means them with `.fillna(0.0)`, which is not the same as skipping them.
