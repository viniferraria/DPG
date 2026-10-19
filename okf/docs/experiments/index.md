# Experiment suites

Research runners under `experiments/`. None of them are type checked by mypy, and each writes
timestamped logs and result artifacts next to itself — treat `results/`, `states/`, and `*.log` as
output, not source.

* [ACIC 2019 Data Challenge](acic2019.md) - Runs the DPG causal-accuracy protocol on ACIC 2019 treatment-effect data, scoring whether a CausalForest's DPG ranks the known effect-modifier feature highly; documents the ATE-versus-feature-ranking mismatch.
* [Causal synthetic scenarios](causal-synthetic-scenarios.md) - Measures whether DPG node-centrality rankings recover the known causal features of four synthetic datasets whose data-generating process is fixed by generator scripts.
* [Local explanation experiments](local-explanation.md) - Parameter sweep that measures how often a DPG local explanation reproduces the model's prediction on built-in sklearn datasets, plus an analysis pass that aggregates the resulting CSVs.
* [MONK's problems](monks.md) - Runs the DPG causal-accuracy protocol against the UCI MONK's benchmark, whose target concepts are exactly known, using the benchmark's canonical train/test split.
* [COMPAS recidivism score](compas.md) - Fits tree ensembles to ProPublica's COMPAS violent-recidivism score and checks DPG feature rankings against the priors/age reference ranking of Vuk et al.; real data, so no ground truth.

All four causal-accuracy suites score and write through one shared module,
[experiments.causal_accuracy](/modules/experiments-causal-accuracy.md) (reference scoring, `METRICS`,
`node_metrics` / `causal_accuracy` CSV writers). Suite pages name only what differs: ground-truth source and
call-site keywords.

## Test coverage at a glance

| Suite | Package? | Covered by |
|---|---|---|
| `causal_synthetic_scenarios/` | no `__init__.py` | `tests/test_run_dpg_causal_synthetic.py` |
| `local_explanation/` | yes — the only one | `tests/test_smoke.py` |
| `monks/` | no `__init__.py` | `tests/test_run_monk.py` (metric columns, deviation keywords) |
| `acic2019/` | no `__init__.py` | `tests/test_run_acic2019.py` |
| `compas/` | no `__init__.py` | `tests/test_run_compas.py` |
| shared `causal_accuracy.py` | module (no package) | `tests/test_causal_accuracy.py` |

## See also

* [Running all experiment suites](/workflows/run-all-experiments.md) - `experiments/run_all.sh` runs every suite with default arguments.
* [Config resolution](/conventions/config-resolution.md) - Runners are launched from varying working directories; this is what decides their `perc_var` and `decimal_threshold`.
* [dpg.explainer](/modules/dpg-explainer.md) - The API all three suites drive.
