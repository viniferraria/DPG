---
type: Experiment Suite
title: Causal synthetic scenarios
description: Measures whether DPG node-centrality rankings recover the known causal features of four synthetic datasets whose data-generating process is fixed by generator scripts.
resource: https://github.com/viniferraria/DPG/tree/main/experiments/causal_synthetic_scenarios
tags: [experiments, causal, synthetic-data, centrality, node-metrics, ground-truth]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Purpose

Each scenario CSV is produced by a generator script with a written-down data-generating process, so the
set of causal features is known by construction. The runner fits RandomForest and ExtraTrees on the
scenario, builds a [`DPGExplainer`](/modules/dpg-explainer.md) global explanation, ranks predicate nodes
by graph centrality, and scores the top-k features against that ground truth.

Ground truth is hard-coded in `SCENARIO_GROUND_TRUTH` in `run_dpg_causal_synthetic.py`:

| Scenario file | Causal features | Mechanism (from the generator docstring) |
|---|---|---|
| `datasets/scenario_1.csv` | `F1` | `Y = F1`; `F2`, `F3` (one-hot `F3_A/B/C`) irrelevant — sanity check |
| `datasets/scenario_2_updated.csv` | `F1` | `Y = F1 + ε`, σ=1.5 (SNR 4); `F2`/`F3` weakly correlated (ρ=0.30), `F4`/`F5` irrelevant categoricals |
| `datasets/scenario_3.csv` | `F1`, `F2` | `Y ← 0.6·F1 + 0.4·F2 + ε`; two additive causes with `F1 > F2` |
| `datasets/scenario_5.csv` | `F1`, `F2` | `s = sin(F1) + exp(F2) + ε`, `Y = 1{s > median(s)}`; non-linear additive, balanced binary label |

Scoring is set-based (`evaluate_causal_accuracy`): precision `|E ∩ GT| / |E|`, recall `|E ∩ GT| / |GT|`
with `TOP_K = 3`, repeated for each of `METRICS` — `Local reaching centrality`, `Closeness centrality`,
`Harmonic centrality`.

# Layout

```
experiments/causal_synthetic_scenarios/
├── run_dpg_causal_synthetic.py   # the maintained runner (pure helpers + main())
├── run_dpg_synthetic.py          # older top-level-script variant, no main()
├── datasets/
│   ├── scenario_1.csv            # 1000 rows: F1,F2,F3_A,F3_B,F3_C,Y
│   ├── scenario_2_updated.csv    # 1000 rows: F1,F2,F3,F4,F5_A,F5_B,F5_C,Y
│   ├── scenario_3.csv            # 1000 rows: F1,F2,F3,F4,F5_A,F5_B,F5_C,Y
│   └── scenario_5.csv            # 1000 rows: F1..F6,F7_A/B/C,F8_X/Y/Z,Y
├── generator/
│   ├── scenario_1_generator.py   # -> test_datasets/scenario_1.csv
│   ├── scenario_2_generator.py   # -> <suite>/test_datasets/scenario_2_updated.csv
│   ├── scenario_3_generator.py   # -> test_datasets/scenario_3.csv
│   ├── scenario_5_generator.py   # -> <suite>/test_datasets/scenario_5.csv
│   ├── dataset_scenario1.py      # -> scenario_01_sanity_check.csv  (exploratory)
│   ├── dataset_scenario1_bin.py  # -> scenario_01b_binary_y.csv     (exploratory)
│   ├── dataset_scenario2.py      # -> scenario_02_moderate_noise_weak_corr.csv
│   ├── dataset_scenario2_bin.py  # -> scenario_02b_binary_y.csv     (exploratory)
│   └── Untitled-1.ipynb          # plus dataset_scenario5.py (exploratory)
├── results/  states/             # accumulated CSV / .pkl artifacts (not source)
└── dpg_explainer_<timestamp>.log # emitted per run into the CWD
```

# How to run

The runner reads its scenario paths as `datasets/...` relative to the process CWD, so it must be run
from the suite directory:

```bash
cd experiments/causal_synthetic_scenarios
uv run python run_dpg_causal_synthetic.py
```

There is no argparse and no CLI flag — everything is module-level constants: `RANDOM_STATE = 42`,
`N_SPLITS = 2`, `TEST_SIZE = 0.2`, `TOP_K = 3`, `MODEL_FACTORIES = {"RandomForest", "ExtraTrees"}`
(both `random_state=42`, otherwise sklearn defaults). `main()` calls
`run_experiments_with_ground_truth(SCENARIO_GROUND_TRUTH, OUTPUT_PATH, logger)`.
Splits come from `StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)`; the DPG
explainer is fitted on `X_train.values` only, then `explain_global(communities=True)` is called.
The runner passes `config_file=str(CONFIG_PATH.resolve(strict=True))` where
`CONFIG_PATH = <suite>/../../config.yaml` — the repo-root `config.yaml`, located from the script rather
than the CWD, side-stepping the usual [config resolution](/conventions/config-resolution.md) CWD
dependence: `perc_var=0.0001`, `decimal_threshold=3`, `n_jobs=-1`. `strict=True` makes a missing
`config.yaml` a hard failure.

Regenerating a dataset (writes to `test_datasets/`, not `datasets/` — see Gotchas):

```bash
cd experiments/causal_synthetic_scenarios && uv run python generator/scenario_3_generator.py
```

# Outputs

Written under `results/` and `states/`, both created on demand:

- `results/node_metrics_<timestamp>.csv` — one row per graph node per (scenario, model, split), columns
  from `NodeMetricRecord`: `experiment, split, node, degree, in_degree, out_degree,
  betweenness_centrality, local_reaching_centrality, node_idx, label, processing_time`. Header written
  only when the file does not already exist (`write_records_to_csv` opens in `"a+"`).
- `results/causal_accuracy_<timestamp>.csv` — `experiment, split, k, ground_truth,
  explanation_top_features, metric, intersection, precision, recall`; list fields `;`-joined.
  `write_causal_accuracy_to_csv` recomputes `timestamp()` on **every call**, so one run scatters rows
  across many tiny files rather than one.
- `states/<scenario>_<Model>_split<N>_dpg_explainer_<timestamp>.pkl` (pickled `DPGExplainer`) and
  `states/<scenario>_<Model>_split<N>_explanation_<timestamp>.pkl` (pickled `DPGExplanation`), sharing
  one `ts` per split — e.g. `scenario_5_ExtraTrees_split1_explanation_2026-07-26T23-13-17.pkl`.
- `dpg_explainer_<timestamp>.log` — created by `get_logger(..., log_file=...)` in the CWD, one per run.

Timestamps are UTC `%Y-%m-%dT%H-%M-%S`. Everything under `results/`, `states/`, and the
`dpg_explainer_*.log` files is accumulated run output, not source — some of it is committed, and it
grows monotonically because nothing is overwritten or cleaned.

# Test coverage

`tests/test_run_dpg_causal_synthetic.py` imports
`experiments.causal_synthetic_scenarios.run_dpg_causal_synthetic` and exercises the pure helpers only:
`timestamp` (fixed datetime + filesystem safety), `load_dataset` (last column is the target),
`make_splitter` (reproducibility, `N_SPLITS`), `evaluate_accuracy`, `records_from_explanation` (column
mapping, types, 4-dp rounding, empty frame), `write_records_to_csv` (roundtrip and no duplicate header),
`save_pickle`, `mean_or_nan`, and `iter_splits`. Two hygiene tests assert importing the module has no
side effects and that `MODEL_FACTORIES` yields fresh instances.

`test_run_experiments_wiring_with_stubs` drives `run_experiments` end-to-end with `explain_split`
monkeypatched out, so no DPG pipeline runs. Nothing covers `run_experiments_with_ground_truth`,
`extract_top_k_features`, `evaluate_causal_accuracy`, `write_causal_accuracy_to_csv`,
`_build_explanation`, the generator scripts, or `run_dpg_synthetic.py`.

# Gotchas

- **Generators do not write where the runner reads.** All four `scenario_*_generator.py` scripts write
  to `test_datasets/` (CWD-relative for scenarios 1 and 3, `<suite>/test_datasets/` for 2 and 5) while
  the runner reads `datasets/`, and `test_datasets/` does not exist in the tree. Regenerating a
  scenario therefore does not update the file the runner uses — copy it across by hand.
- **`run_dpg_synthetic.py` is dead as-committed.** No `main()` guard (importing it runs the experiment),
  its only enabled input `test_datasets/scenario_2.csv` does not exist, it writes node metrics to
  `datasets/node_metrics_3.csv` in `"w+"` (overwriting), and it constructs `DPGExplainer` with no
  `config_file`, falling back to CWD-relative resolution. Prefer `run_dpg_causal_synthetic.py`.
- **The `dataset_scenario*.py` generators are exploratory.** They print diagnostics (permutation
  importance, PCA, cross-validated scores) and write differently named CSVs
  (`scenario_01_sanity_check.csv`, `scenario_02b_binary_y.csv`, …) absent from `datasets/`.
- **Per-split failures are swallowed.** `run_experiments_with_ground_truth` wraps each split in
  `try/except Exception`, logging a traceback but continuing, so a partially failed run still produces
  CSVs — check the log before trusting a results file.
- **Categorical features arrive pre-one-hot.** `scenario_1.csv` already carries `F3_A/F3_B/F3_C`, so
  predicate labels read `F3_B <= 0.5`; `extract_top_k_features` takes the token before the first space
  and does *not* map one-hot columns back to a source attribute (unlike the MONK's runner). `y` is also
  passed whole into `_build_explanation`, so `target_names` come from the full dataset, not the split.
- Node IDs in `node_metrics_*.csv` are sha1-derived integers, stable only for byte-identical labels.
