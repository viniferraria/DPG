# Experiment suites

Research runners under `experiments/`. None of them are type checked by mypy, and each writes
timestamped logs and result artifacts next to itself — treat `results/`, `states/`, and `*.log` as
output, not source.

* [Causal synthetic scenarios](causal-synthetic-scenarios.md) - Measures whether DPG node-centrality rankings recover the known causal features of four synthetic datasets whose data-generating process is fixed by generator scripts.
* [Local explanation experiments](local-explanation.md) - Parameter sweep that measures how often a DPG local explanation reproduces the model's prediction on built-in sklearn datasets, plus an analysis pass that aggregates the resulting CSVs.
* [MONK's problems](monks.md) - Runs the DPG causal-accuracy protocol against the UCI MONK's benchmark, whose target concepts are exactly known, using the benchmark's canonical train/test split.

## Test coverage at a glance

| Suite | Package? | Covered by |
|---|---|---|
| `causal_synthetic_scenarios/` | no `__init__.py` | `tests/test_run_dpg_causal_synthetic.py` |
| `local_explanation/` | yes — the only one | `tests/test_smoke.py` |
| `monks/` | no `__init__.py` | none |

## See also

* [Config resolution](/conventions/config-resolution.md) - Runners are launched from varying working directories; this is what decides their `perc_var` and `decimal_threshold`.
* [dpg.explainer](/modules/dpg-explainer.md) - The API all three suites drive.
