---
type: Playbook
title: Testing, linting, and type checking
description: Run pytest, ruff, and mypy exactly as the Feature CI workflow does, including the repo-root working-directory requirement and the narrow mypy scope.
resource: https://github.com/viniferraria/DPG/blob/main/pyproject.toml
tags: [testing, pytest, ruff, mypy, ci, quality-gates]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# When to use

Before pushing to a `feature/**` branch — `.github/workflows/feature-pr.yaml` runs these three
commands in this order and only opens the automated PR if all three pass. Environment prerequisites are
in [/workflows/development-setup.md](/workflows/development-setup.md); the surrounding automation is in
[/workflows/ci-pipelines.md](/workflows/ci-pipelines.md).

# Steps

## 1. Sync the dev group, then run ruff over tracked files only

CI passes an explicit file list produced by `git ls-files` (49 tracked `.py` files at time of writing)
rather than letting ruff walk the tree:

```bash
uv sync --locked --dev
uv run ruff check $(git ls-files '*.py')
```

Expected output: `All checks passed!`

Enumerating tracked files keeps ruff out of `.venv/`, `.mypy_cache/`, `.ruff_cache/`, and generated
artifacts under `experiments/` — none of which are covered by an ignore file, because **there is no
`[tool.ruff]` table in `pyproject.toml` and no `ruff.toml` / `.ruff.toml` in the repo**. Ruff runs on
its built-in defaults: resolved settings report `linter.unresolved_target_version = 3.10` (inferred
from `requires-python`), `linter.preview = disabled`, `linter.exclude = []`,
`linter.per_file_ignores = {}`.

Verified behavior of that default set with the pinned ruff (`ruff>=0.15.13`): `F401` (unused import)
and `I001` (unsorted import block) fire; `E501` (line too long) does not. Confirm rather than assume:
`uv run ruff check --show-settings dpg/core.py`.

## 2. Mypy — only `dpg/` and `metrics/`

```bash
uv run mypy dpg/ metrics/
```

Expected output: `Success: no issues found in 13 source files`

Scope is deliberate: `examples/`, `experiments/`, `tests/`, and `tutorials/` are **not** type checked.
The real flags in `[tool.mypy]`:

| Setting | Value |
|---|---|
| `python_version` | `"3.10"` |
| `disallow_untyped_defs` | `true` |
| `warn_return_any` | `true` |
| `warn_unused_ignores` | `true` |
| `no_implicit_optional` | `true` |
| `pretty` | `true` |
| `show_error_codes` | `true` |

Two override blocks relax that baseline: `ignore_missing_imports = true` for `graphviz.*`,
`networkx.*`, `sklearn.*`, `joblib.*`, `tqdm.*`, `scipy.*`, `omegaconf.*`, `PIL.*`, `igraph.*`
(third-party packages shipping no stubs); and `disallow_untyped_defs = false` for `dpg.visualizer`
only — every other check stays on there, and the override carries a TODO to annotate its plot
functions and drop it. There is no `strict = true` and no `disallow_any_*` / `warn_unreachable`.

## 3. Pytest — from the repo root

```bash
uv run pytest
uv run pytest tests/test_explainer.py
uv run pytest tests/test_explainer.py::test_name
```

**Tests must be invoked from the repo root.** Two independent reasons:

1. `tests/conftest.py` computes `ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))`
   and prepends it to `sys.path`. That makes `import dpg` / `import metrics` work without installing
   the package — but it puts *the repo root*, not the CWD, on the path, so it fixes imports only, not
   data-file lookups.
2. `tests/test_integration_dpg.py::_build_small_dpg` resolves its fixture through the working
   directory: `base_dir = os.getcwd()`, then `os.path.join(base_dir, "datasets", "custom.csv")`. Run
   pytest from anywhere else and that read fails with `FileNotFoundError`.

`tests/README.md` documents the equivalent no-uv invocation: `PYTHONPATH=. pytest -q`.

## 4. Full local gate

```bash
uv sync --locked --dev \
  && uv run ruff check $(git ls-files '*.py') \
  && uv run mypy dpg/ metrics/ \
  && uv run pytest
```

`--locked` fails outright on a stale lockfile, so commit `uv.lock` alongside any dependency edit.

# Gotchas

## No pytest configuration exists

There is no `[tool.pytest.ini_options]` in `pyproject.toml` and no `pytest.ini`, `tox.ini`, or
`setup.cfg`. So there are no `testpaths`, no default `addopts`, no marker registry, and no
`filterwarnings`. Bare `uv run pytest` collects from the CWD — another reason the repo root is the only
correct place to run it. `test_datasets/` exists but is empty; the only fixture data is
`datasets/custom.csv`.

## Plot tests skip silently without Graphviz

`test_dpg_plots_render` calls `shutil.which("dot")` and issues `pytest.skip` when the binary is absent.
A green run on a machine without Graphviz has not exercised rendering at all. `test_integration_dpg.py`
also sets `os.environ.setdefault("MPLBACKEND", "Agg")` at import time so matplotlib never opens a
window — preserve that if you add plotting tests.

## Config-dependent test outcomes

Tests constructing `DecisionPredicateGraph` without an explicit `dpg_config` pick up the repo-root
`config.yaml` (`perc_var: 0.0001`, `decimal_threshold: 3`), which differs from `DEFAULT_DPG_CONFIG`. A
test that passes here can fail when the same code runs elsewhere; pass `dpg_config` explicitly.

## Ruff and mypy scopes differ for the experiment runners

`experiments/` scripts are linted by ruff (they are tracked `.py` files) but are **not** type checked.
Only `tests/test_run_dpg_causal_synthetic.py` touches that tree from the test suite.
