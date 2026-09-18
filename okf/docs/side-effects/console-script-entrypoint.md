---
type: Side Effect
title: The installed dpg console script is dead
description: pyproject.toml declares the same console-script name twice, under [project.scripts] and [tool.poetry.scripts]; the PEP 621 [project] table wins during a uv/pip build, so the installed `dpg` command points at a module that doesn't exist and crashes on invocation.
resource: https://github.com/viniferraria/DPG/blob/main/pyproject.toml
tags: [packaging, cli, entry-points, pyproject, regression]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed

`pyproject.toml` declares the `dpg` console script twice, in two different tables, pointing at two
different targets:

```toml
# pyproject.toml:12 — [project] table (PEP 621)
scripts = { "dpg" = "scripts.run_dpg_standard:main" }
```

```toml
# pyproject.toml:127 — [tool.poetry.scripts]
[tool.poetry.scripts]
dpg = "dpg.cli:main"
```

`dpg/cli.py` (with a real `build_parser()` / `main()`) is a 0.3.0 addition, added specifically to give
the package a working CLI entry point. But the build backend is `poetry-core` building a **PEP 621**
project (`[build-system] build-backend = "poetry.core.masonry.api"` with a populated `[project]`
table) — and for a PEP 621 project, `poetry-core` and `uv` both read `[project.scripts]`, not
`[tool.poetry.scripts]`. The legacy `[tool.poetry.scripts]` table is silently ignored. `dpg.cli:main`
is unreachable as an installed command; only `python -m dpg.cli` reaches it directly.

# Why

This looks like an incomplete migration: `dpg/cli.py` was added to replace the old
`scripts.run_dpg_standard:main` target, but the old `[project.scripts]` entry was never updated to
point at it, and a new (also stale) entry was added under `[tool.poetry.scripts]` — a table this
build no longer consults for entry points now that `[project]` is populated.

# Who is affected

Anyone who runs `uv run dpg`, `pip install .` then `dpg`, or otherwise invokes the console script
installed under the name `dpg`.

# Evidence

The installed distribution's `entry_points.txt` confirms which table won:

```text
$ cat .venv/lib/python3.14/site-packages/dpg-0.3.0.dist-info/entry_points.txt
[console_scripts]
dpg=scripts.run_dpg_standard:main
```

Running the installed script:

```text
$ uv run dpg --help
Traceback (most recent call last):
  File "/Users/vinicius/Projects/DPG/.venv/bin/dpg", line 4, in <module>
    from scripts.run_dpg_standard import main
ModuleNotFoundError: No module named 'scripts'
```

`scripts/` is not a package in this repository (`examples/run_dpg_standard.py` is a script, not an
importable `scripts.run_dpg_standard` module) — the target has never resolved.

# Workaround

Run `dpg/cli.py` as a module instead of via the installed script:

```text
$ uv run python -m dpg.cli --help
usage: python3 -m dpg.cli [-h] [--dataset DATASET]
                          [--target_column TARGET_COLUMN]
                          [--n_learners N_LEARNERS] [--model_name MODEL_NAME]
                          [--dir DIR] [--plot] [--save_plot_dir SAVE_PLOT_DIR]
                          [--attribute ATTRIBUTE] [--communities] [--clusters]
                          [--threshold_clusters THRESHOLD_CLUSTERS] [--t T]
                          [--class_flag] [--seed SEED] [--pv PV]

Train a sklearn tree ensemble and export its DPG metrics.
...
```

This works today: verified live against HEAD with `uv run python -m dpg.cli --help`.

# File:line citations (HEAD `276a503`)

- `pyproject.toml:12` — `[project] scripts = { "dpg" = "scripts.run_dpg_standard:main" }` (the table
  that wins).
- `pyproject.toml:127` — `[tool.poetry.scripts] dpg = "dpg.cli:main"` (ignored).
- `dpg/cli.py` — the real, working `build_parser()` / `main()` this entry should point to.
- `.venv/lib/python3.14/site-packages/dpg-0.3.0.dist-info/entry_points.txt` — installed proof of
  which target won.

# Related

- [dpg.cli](/modules/dpg-cli.md)
- [Running the DPG CLI entrypoints](/workflows/cli-entrypoints.md)
