---
type: Python Module
title: dpg.cli — packaged command-line interface
description: build_parser/main for the dpg command that trains a sklearn tree ensemble and exports its DPG metrics — and why the installed dpg console script does not reach it.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/cli.py
tags: [python, module, cli, argparse, packaging, new-in-0.3.0]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# Responsibility

`dpg/cli.py` (97 lines) is new in 0.3.0: a real, packaged `main()` for the `dpg` command, wrapping
[`dpg.sklearn_dpg.test_dpg`](/modules/dpg-sklearn-dpg.md). It exports `build_parser` and `main`, and
runs `raise SystemExit(main())` under `if __name__ == "__main__":`.

**The installed `dpg` console script does not reach this module.** `pyproject.toml` declares the
script twice, in two different tables, and they disagree:

```toml
[project]
scripts = { "dpg" = "scripts.run_dpg_standard:main" }   # pyproject.toml:12

[tool.poetry.scripts]
dpg = "dpg.cli:main"                                     # pyproject.toml:127-128
```

PEP 621 `[project.scripts]` wins whenever both are present, and `pyproject.toml:12` points at a
`scripts` package that does not exist in this repo. Verified directly:

```
$ uv run dpg --help
Traceback (most recent call last):
  File ".../.venv/bin/dpg", line 4, in <module>
    from scripts.run_dpg_standard import main
ModuleNotFoundError: No module named 'scripts'
```

```
$ uv run python -m dpg.cli --dataset iris --n_learners 3 --dir /tmp/dpgclitest/out
... (trains, builds the DPG) ...
$ ls /tmp/dpgclitest/out
iris_seed160898_dpg_metrics.txt
iris_seed160898_edge_metrics.csv
iris_seed160898_node_metrics.csv
iris_seed160898_stats.txt
```

Both commands run at HEAD (`276a503`). `uv run python -m dpg.cli ...` is the only way to invoke this
module today. See
[/side-effects/console-script-entrypoint.md](/side-effects/console-script-entrypoint.md) and, for the
older `examples/run_dpg_standard.py --ds ...` invocation this module does **not** replace,
[/workflows/cli-entrypoints.md](/workflows/cli-entrypoints.md).

# API

## `build_parser() -> argparse.ArgumentParser`

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--dataset` / `--ds` | str | `"iris"` | sklearn dataset name or CSV path; `dest="dataset"` |
| `--target_column` | str | `None` | Target column when `--dataset` is a CSV |
| `--n_learners` / `--l` | int | `5` | `dest="n_learners"` |
| `--model_name` | str | `"RandomForestClassifier"` | Forwarded to `test_dpg` |
| `--dir` | str | `"examples/results"` | Output directory, created with `mkdir(parents=True, exist_ok=True)` |
| `--plot` | flag | `False` | |
| `--save_plot_dir` | str | `"examples/results"` | |
| `--attribute` | str | `None` | |
| `--communities` | flag | `False` | |
| `--clusters` | flag | `False` | |
| `--threshold_clusters` | float | `None` | |
| `--t` | int | `3` | Threshold decimal precision, forwarded as `decimal_threshold` |
| `--class_flag` | flag | `False` | |
| `--seed` | int | `160898` | Unlike `examples/run_dpg_standard.py`'s `--seed` (no default), this one always has a value |
| `--pv` | float | `1e-9` | Minimum path frequency proportion, forwarded as `perc_var` |

Note the two default divergences from `examples/run_dpg_standard.py`
([/workflows/cli-entrypoints.md](/workflows/cli-entrypoints.md)): that script's `--seed` has no
default (`None`) and its `--t`/`--pv` default to `None` (falling through to `config.yaml`); this
module's `--seed` defaults to `160898` and `--t`/`--pv` always have explicit values
(`3`/`1e-9`, matching `DEFAULT_DPG_CONFIG`), so `test_dpg` here is never handed `None` for them.
`n_jobs` is hardcoded to `-1` in `main()`, not exposed as a flag.

## `main(argv: Sequence[str] | None = None) -> int`

1. Parses `argv` (or `sys.argv` when `None`), creates `--dir`.
2. Calls `test_dpg(...)` with every parsed flag, `n_jobs=-1`, and
   `file_name=str(output_dir / f"{Path(args.dataset).stem}_seed{args.seed}_stats.txt")`.
3. `test_dpg` returns either `None`/a 2-tuple `(None, None)` (both signaled by `len(result) != 6`) or
   the full 6-tuple `(df, df_edges, graph_metrics, clusters, node_prob, confidence)`. `main` checks
   `result is None or len(result) != 6` and **returns `1`** on that path — unlike
   `examples/run_dpg_custom.py`, which unconditionally unpacks 2 values and crashes (see
   [/workflows/cli-entrypoints.md](/workflows/cli-entrypoints.md) and CLAUDE.md's Known breakage).
4. On success, writes three files into `--dir` (stem is `Path(args.dataset).stem`, so a CSV path's
   extension is dropped and a bare dataset name like `"iris"` is used as-is):
   - `{stem}_seed{seed}_node_metrics.csv` — `df.to_csv(..., index=False)`
   - `{stem}_seed{seed}_edge_metrics.csv` — `df_edges.to_csv(..., index=False)`
   - `{stem}_seed{seed}_dpg_metrics.txt` — one `f"{key}: {value}\n"` line per `graph_metrics` item
   - and, only `if clusters is not None`, `{stem}_seed{seed}_clusters.txt` with `Clusters:`,
     `Probability:`, `Confidence:` lines
5. Also indirectly writes `{stem}_seed{seed}_stats.txt` (accuracy/F1/confusion matrix), since that's
   the `file_name` `test_dpg` itself writes to — see
   [/modules/dpg-sklearn-dpg.md](/modules/dpg-sklearn-dpg.md).
6. Returns `0` on success, `1` on the "insufficient nodes" signal from `test_dpg`. There is no other
   non-zero return path — an exception inside `test_dpg` propagates uncaught rather than being turned
   into a return code.

# Gotchas

- **This module's own file naming differs from `examples/run_dpg_standard.py`'s.** The example script
  names outputs `{ds}_l{l}_seed{seed}_...`; `dpg/cli.py` drops `_l{l}` from the pattern
  (`{stem}_seed{seed}_...`). Don't assume the two are drop-in compatible for downstream tooling that
  parses filenames.
- **`--pv`/`--t` here are not the "rename only" no-ops they are in `run_dpg_standard.py`.** `main()`
  passes `args.pv`/`args.t` straight through to `test_dpg` as `perc_var`/`decimal_threshold`, which
  `test_dpg` forwards into `DecisionPredicateGraph(dpg_config=...)` — they do change the graph. (Verify
  against [/modules/dpg-sklearn-dpg.md](/modules/dpg-sklearn-dpg.md) if `test_dpg`'s config wiring
  changes.)
- `main()` never reads `config.yaml` itself — every `DecisionPredicateGraph` setting `test_dpg` cares
  about is passed explicitly from parsed flags, so CWD-dependence
  ([/conventions/config-resolution.md](/conventions/config-resolution.md)) does not apply to this
  entrypoint the way it does to `examples/run_dpg_standard.py`.
- `if __name__ == "__main__":` means `python dpg/cli.py` (as opposed to `python -m dpg.cli`) fails with
  the usual `ImportError: attempted relative import` style failure from the package-relative
  `from .sklearn_dpg import test_dpg` — always invoke as a module.

# Example

```bash
uv run python -m dpg.cli --dataset iris --n_learners 3 --dir results/iris_cli
```
