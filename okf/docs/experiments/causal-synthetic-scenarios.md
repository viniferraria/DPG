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
`Harmonic centrality`, `Betweenness centrality`, `Collective influence`, `Percolation centrality`.
Betweenness is the control for percolation: with equal node states percolation reduces to normalized
betweenness, so any rank difference comes from the traffic-based states.

# SCM scenarios (S1-S16)

Sixteen further scenarios come from an external generator (`generator/causal_dataset_generators/
dpg_causal_dataset_generators/`, `causal/src/synthetic_scm.py`). Pre-generated data is copied to
`datasets/scm/<scenario>/seed_<n>/{X.csv, y.csv, ground_truth.yaml, metadata.json}` (29 MB, 16
scenarios x seeds 42-51, 1000 rows, 8 standard-normal features `X1..X8` plus scenario extras).
They are opt-in: `uv run python run_dpg_causal_synthetic.py --scm` adds them to the four CSV scenarios
(160 scenario dirs x 2 models x 2 splits, a long sweep). `scm_ground_truth(root, scenarios, seeds)`
builds the `{seed_dir: causal_features}` dict from each `ground_truth.yaml` (nothing hard-coded), and
`load_dataset` reads a directory as `X.csv` + `y.csv`. A subset run:

```python
gt = scm_ground_truth(scenarios=["s13_mediator"], seeds=[42])
run_experiments_with_ground_truth(gt, Path("out"), logger)
```

Run keys are `<scenario>_seed_<n>_<Model>` (e.g. `s13_mediator_seed_42_RandomForest`).
`SCENARIO_GROUND_TRUTH` and the four CSV scenarios are unchanged. `yaml` is imported from PyYAML,
which arrives transitively via OmegaConf, not as a direct dependency.

All 16 are classification: 15 binary (balanced 500/500 except `s04_xor` 504/496 and `s09_class_imbalance`
689/311 at seed 42) and `s16` 4-class. None is regression. `s15` and `s16` run as plain single-table
runs (see limitations); none are excluded.

## What "causal" means in the ground truth

`causal_features` lists variables that enter the score directly, not everything upstream of Y.

| Scenario | `causal_features` | Notes |
|---|---|---|
| s01 linear_additive | X1, X2, X3 | |
| s02 nonlinear_main | X1, X2, X3 | `X2^2` is non-monotone |
| s03 redundant_correlated | X1, X2 | `X4 = X1 + noise` is a proxy, not causal |
| s04 xor | X1, X2 | no marginal effect; 5% label flips |
| s05 multiplicative | X1, X2, X3 | |
| s06 threshold_rule | X1, X2, X3 | |
| s07 sparse_high_dim | X2, X11, X17 | 20 features, 17 distractors |
| s08 weak_effects | X1, X2, X3 | noise 0.9 vs coefficients 0.3-0.45 |
| s09 class_imbalance | X1, X2 | Bernoulli(sigmoid) target |
| s10 nonactionable | age_like, X1, X2 | `age_like` is truly predictive but non-actionable |
| s11 chain | X2 | X1 -> X2 -> Y; **X1 is not causal here** |
| s12 hidden_fork | (empty) | latent Z -> X_proxy, Z -> Y; Z unobserved |
| s13 mediator | M | X1 -> M -> Y; score is `1.4 M + 0.2 X1`, **X1 not in GT** |
| s14 proxy | X1, X2 | `P = X1 + noise` is a proxy |
| s15 distribution_shift | X1, X2, X3 | X3 predictive only in the second half |
| s16 multi_outcome_shared | X1, X2, X3 | 4-class code `2*a + b` |

Consequences for judging rankings:

- Mediator/chain: a DPG that ranks X1 highly is penalised (precision) although X1 drives M. In s13 the
  generator even adds `0.2 * X1` directly to the score, yet X1 stays out of GT.
- Proxy (s03, s14): a proxy ranked above its source costs recall; the GT does not credit it.
- s12: GT is empty, so see the limitation below.

## Limitations and side effects

- **Empty ground truth (s12).** `evaluate_causal_accuracy` returns precision 0.0 and recall 0.0
  whenever GT is empty, and the CSV `ground_truth`/`intersection` fields are blank. Example from the
  smoke run: `s12_hidden_fork_seed_42_RandomForest,0,3,,X3;X5;X6,Local reaching centrality,,0.0,0.0`.
  Those zeros mean "undefined", not "failed"; averaging them into other scenarios deflates means. Judge
  s12 by whether `X_proxy` (the observed stand-in for Z) tops the ranking instead.
- **Duplicate feature names in top-k** (existing behaviour) also applies: `X3;X5;X3` keeps repeats.
- **s15 pooled, not shifted.** Rows 0-499 and 500-999 are two environments (X3 shifted by +1.5 in the
  second), but `StratifiedShuffleSplit` mixes them across train/test, so no out-of-distribution test is
  run. GT lists X3 although it matters only in half the rows, so its recovery is not evidence of
  robustness to the shift.
- **s16 is one 4-class target.** The label encodes two binary outcomes (`2*a + b`); DPG explains the
  joint class. Classes 0/3 are 327 rows and 1/2 are 173 rows at seed 42; per-outcome attribution is not
  separable, so X2 (only in outcome a) and X3 (only in b) share one GT.
- **Result files unseparated.** SCM rows land in the same `results/causal_accuracy_*.csv` as the four
  CSV scenarios, distinguished only by the `s..._seed_..` run key.
- **Gaps not covered.** A high-cardinality-noise scenario and a shuffled-label null scenario cannot be
  produced from `synthetic_scm.py` by configuration: the CLI only exposes `--scenarios`, `--n-samples`
  and `--seeds`; the feature count is a hard-coded argument (`n_features=20` only in `s07`) and no
  scenario permutes labels or has an empty-signal mechanism or categorical high-cardinality features.
  They need two new scenario functions (e.g. `_package(..., causal_features=[])` over pure noise with
  `y` from `rng.permutation`, and a wide/categorical `_base_features`). Not written here.

# Layout

```
experiments/causal_synthetic_scenarios/
├── run_dpg_causal_synthetic.py   # the maintained runner (pure helpers + main())
├── run_dpg_synthetic.py          # older variant; reads test_datasets/
├── datasets/
│   ├── scenario_1.csv            # 1000 rows: F1,F2,F3_A,F3_B,F3_C,Y
│   ├── scenario_2_updated.csv    # 1000 rows: F1,F2,F3,F4,F5_A,F5_B,F5_C,Y
│   ├── scenario_3.csv            # 1000 rows: F1,F2,F3,F4,F5_A,F5_B,F5_C,Y
│   ├── scenario_5.csv            # 1000 rows: F1..F6,F7_A/B/C,F8_X/Y/Z,Y
│   └── scm/<scenario>/seed_<n>/  # X.csv, y.csv, ground_truth.yaml, metadata.json (S1-S16)
├── generator/
│   ├── scenario_1_generator.py   # -> <suite>/test_datasets/scenario_1.csv
│   ├── scenario_2_generator.py   # -> <suite>/test_datasets/scenario_2_updated.csv
│   ├── scenario_3_generator.py   # -> <suite>/test_datasets/scenario_3.csv
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
uv run python experiments/causal_synthetic_scenarios/generator/scenario_3_generator.py
```

# Outputs

Written under `results/` and `states/`, both created by `main()` before any experiment runs.
`write_causal_accuracy_to_csv` does not create its folder, so a caller that bypasses `main()` must
create `results/` itself; without it every split fails with `FileNotFoundError`, which the per-run
`except Exception` logs as `Failed:` while the run carries on (that was the case before 2026-09-30):

- `results/node_metrics_<timestamp>.csv` — one row per graph node per (scenario, model, split), columns
  from `NodeMetricRecord`: `experiment, split, node, degree, in_degree, out_degree,
  betweenness_centrality, local_reaching_centrality, closeness_centrality, harmonic_centrality,
  collective_influence, local_clustering_coefficient, percolation_centrality, node_idx, label,
  processing_time`. Header written
  only when the file does not already exist (`write_records_to_csv` opens in `"a+"`).
- `results/causal_accuracy_<timestamp>.csv` — `experiment, split, k, ground_truth,
  explanation_top_features, metric, intersection, precision, recall`; list fields `;`-joined.
  `run_experiments_with_ground_truth` takes one `timestamp()` per run and passes the path down as
  `file_path`, so every (scenario, model, split, metric) row of a run lands in this one file, which
  shares its timestamp with that run's `node_metrics_<timestamp>.csv` (before 2026-10-13 each call
  recomputed `timestamp()` and scattered a run across many tiny files).
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
`save_pickle`, `mean_or_nan`, `iter_splits`, and `extract_top_k_features` (with and without
`drop_duplicates`), and `load_dataset` / `scm_ground_truth` on the SCM directory layout.
`test_records_from_real_node_metrics_have_all_runner_columns` runs the real
`NodeMetrics.extract_node_metrics` on a tiny graph and checks every `METRICS` name and every
`NodeMetricRecord` source column exists — a missing column would otherwise raise a `KeyError` that
the per-run `except Exception` swallows, silently dropping rows. Two hygiene tests assert importing the module has no
side effects and that `MODEL_FACTORIES` yields fresh instances.

`test_run_experiments_wiring_with_stubs` drives `run_experiments` end-to-end with `explain_split`
monkeypatched out, so no DPG pipeline runs. Nothing covers `run_experiments_with_ground_truth`, `evaluate_causal_accuracy`, `write_causal_accuracy_to_csv`,
`_build_explanation`, the generator scripts, or `run_dpg_synthetic.py`.

# Gotchas

- `Collective influence` is always computed at radius ℓ = 2: `DPGExplainer` calls
  `NodeMetrics.extract_node_metrics` without `ci_radius` and no config key sets it. It is
  integer-valued with many ties at `0`, so its top-k order among tied nodes is arbitrary. `Local
  clustering coefficient` is saved in the node CSV but deliberately not ranked (same tie problem).
- **`extract_top_k_features` keeps repeated features by default — deliberately.** Several predicates
  on one feature can fill the whole top-k (e.g. `F2;F2;F2`), which caps precision/recall for every
  metric. Repeats are kept so new `causal_accuracy_*.csv` rows stay comparable with the accumulated
  history in `results/`, all of which was produced this way. Pass `drop_duplicates=True` to keep only
  each feature's best-ranked predicate; the runner's call in `save_feature_extraction_results` does
  not pass it, so a default run still writes repeats. Don't compare a de-duplicated run against the
  history without saying so. MONK's copy of the function always de-duplicates.
- **Generators do not write where the runner reads.** All four `scenario_*_generator.py` scripts write
  to `<suite>/test_datasets/` while the runner reads `datasets/`. Regenerating a scenario therefore
  does not update the file the runner uses — copy it across by hand.
- **`run_dpg_synthetic.py` is a legacy runner.** It reads `test_datasets/scenario_2_updated.csv`
  (run the generators first to create it), writes node metrics to
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
- Node IDs in `node_metrics_*.csv` are sha1-derived id strings (`"n" + sha1(label)[:12]`), stable only
  for byte-identical labels.
