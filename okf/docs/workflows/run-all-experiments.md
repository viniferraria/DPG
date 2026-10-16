---
type: Playbook
title: Running all experiment suites
description: How experiments/run_all.sh runs every experiment suite through uv with default arguments, in what order, and how to update it when a suite is added or changed.
resource: https://github.com/viniferraria/DPG/blob/main/experiments/run_all.sh
tags: [experiments, playbook, uv, bash]
generated: { by: claude_code/claude-opus-5-5, at: 2026-10-15T00:00:00Z }
status: stable
---

# When to use

Run every suite end to end with one command, using each runner's default arguments. For a single
suite or non-default flags, use the commands on that suite's page under
[experiment suites](/experiments/index.md).

# How to run

From anywhere (the script `cd`s to the repo root first):

```bash
./experiments/run_all.sh
```

It uses `set -euo pipefail`, so it stops at the first failing step. Each step prints `==> <command>`
before running.

# What it runs, in order

| Step | Command |
|---|---|
| Causal synthetic (reference) | `uv run python experiments/causal_synthetic_scenarios/run_dpg_causal_synthetic.py` |
| Synthetic | `uv run python experiments/causal_synthetic_scenarios/run_dpg_synthetic.py` |
| MONK's | `uv run python experiments/monks/run_monk.py` |
| COMPAS | `uv run python experiments/compas/run_compas.py` |
| ACIC 2019 fetch | `uv run python experiments/acic2019/fetch_acic2019.py` — only if `datasets/true_ate.csv` is missing (~200 MB download) |
| ACIC 2019 | `uv run --extra causal python experiments/acic2019/run_acic2019.py` |
| Local explanation sweep | `uv run python -m experiments.local_explanation.run_local_explanations` |
| Local explanation analysis | `uv run python -m experiments.local_explanation.analyze_results --results_dir experiments/local_explanation/results --out_dir experiments/local_explanation/analysis` |

Notes:

- `run_dpg_causal_synthetic.py` runs without `--scm`, so the opt-in SCM scenarios are skipped — see
  [causal synthetic scenarios](/experiments/causal-synthetic-scenarios.md).
- `run_dpg_synthetic.py` reads `test_datasets/`, which the scenario generators create; the script does
  not run the generators.
- ACIC 2019 is the only step that needs the `causal` extra (econml).
- `analyze_results` requires `--results_dir` and `--out_dir` (`required=True`), so they are passed
  explicitly.

# Updating it

When a suite is added, removed, or its entrypoint or required flags change, edit
`experiments/run_all.sh` and the table above in the same change, and add a dated entry to
[the log](/log.md).
