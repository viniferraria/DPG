---
type: Playbook
title: Development environment setup
description: Provision a working DPG development environment with uv, the optional dev and docs dependency sets, and the system Graphviz binary the renderers shell out to.
resource: https://github.com/viniferraria/DPG/blob/main/pyproject.toml
tags: [setup, uv, dependencies, graphviz, poetry-core, python]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# When to use

First checkout of the repo, after a `pyproject.toml` / `uv.lock` change, or when a plot function fails
with a missing `dot` executable. Once the environment exists, see
[/workflows/testing-and-linting.md](/workflows/testing-and-linting.md) for the verification loop and
[/workflows/ci-pipelines.md](/workflows/ci-pipelines.md) for what the hosted runners do with the same
commands.

# Steps

## 1. Install uv

Dependencies are managed by `uv` against the committed `uv.lock` (`version = 1`, `revision = 3`,
`requires-python = ">=3.10"`). There is no `.python-version` file, so uv picks any interpreter
satisfying `requires-python` unless you pin one.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

## 2. Sync dependencies

Three sync shapes, matching the three dependency sets declared in `pyproject.toml`:

```bash
# runtime dependencies only ([project.dependencies])
uv sync

# add the dev group ([dependency-groups].dev): mypy, ruff, pytest, ipython,
# jupyterlab, pyrefly, ty, pandas-stubs, types-PyYAML, types-seaborn
uv sync --group dev

# add the docs extra ([project.optional-dependencies].docs):
# sphinx, myst-nb, myst-parser, pydata-sphinx-theme, sphinx-autoapi,
# sphinx-copybutton, sphinx-design
uv sync --extra docs
```

`uv sync --dev` is an accepted alias for `--group dev`. CI uses that spelling with `--locked`, which
fails instead of updating the lockfile; add `--python` to reproduce a CI run exactly:

```bash
uv sync --locked --dev --python 3.11
uv run python -V        # -> Python 3.11.x
```

## 3. Install the Graphviz system binary

`graphviz` on PyPI is only a thin wrapper: it shells out to the `dot` executable, a system package.
Without it, `dpg/visualizer.py` cannot render and `tests/test_integration_dpg.py::test_dpg_plots_render`
skips itself via `shutil.which("dot")`.

```bash
# macOS
brew install graphviz

# Debian / Ubuntu (this is what .github/workflows/docs.yml runs)
sudo apt-get install -y graphviz

dot -V                   # e.g. "dot - graphviz version 15.1.0"
```

## 4. Smoke-test the environment

```bash
uv run python -c "import dpg, metrics, graphviz, igraph; print('ok')"
uv run pytest -q
```

Both CLI entry points live in `examples/` and must run from the repo root:

```bash
uv run python examples/run_dpg_standard.py --dataset iris --n_learners 5 --pv 0.001 --t 2 --plot
uv run python examples/run_dpg_custom.py --ds datasets/custom.csv --target_column <col>
```

# Gotchas

## Python version support

`requires-python = ">=3.10"` with **no upper bound**, and `uv.lock` carries the same floor. The only
classifier declared is `"Programming Language :: Python :: 3"` — there are no per-minor classifiers, so
do not read a supported-version matrix out of them. `[tool.mypy] python_version = "3.10"` sets the
lowest syntax/typing target. Every workflow that pins an interpreter pins **3.11**
(`.github/workflows/docs.yml`, `.github/workflows/feature-pr.yaml`, and `.readthedocs.yaml`), so 3.11
is the only version actually exercised by automation.

## Build backend is poetry-core, dependency manager is uv

`[build-system]` is `poetry-core>=1.9.0` / `poetry.core.masonry.api`, and `[tool.poetry].packages`
lists `dpg` and `metrics`. Dependencies, however, are declared in PEP 621 `[project]` tables and
resolved by uv. Consequence: `uv add` / `uv sync` edit `[project.dependencies]` and `uv.lock`, while
wheel building goes through poetry-core reading those same tables. Do not add a
`[tool.poetry.dependencies]` table — it would create a second, divergent source of truth.

## The docs dependencies are declared twice

`[dependency-groups].docs` and `[project.optional-dependencies].docs` list the same seven packages;
`--extra docs` reads the latter, `--group docs` the former. They currently agree — edit both together
or one path silently installs a stale set.

## requirements.txt is a separate, pinned file

`requirements.txt` exists with `==` pins (e.g. `numpy==2.1.3`, `scikit-learn==1.5.2`) plus stub and
docs entries. It is **not** generated from `uv.lock` and can drift from `[project.dependencies]`'s `>=`
floors. It also lists `matplotlib-stubs`, which `[dependency-groups].dev` explicitly warns against
because it shadows matplotlib's inline `py.typed` annotations. Prefer `uv` locally; see the drift note
in [/workflows/ci-pipelines.md](/workflows/ci-pipelines.md).

## No dev tool config beyond mypy

`pyproject.toml` contains `[tool.mypy]` (plus two override blocks) and nothing else for tooling —
there is no `[tool.ruff]`, no `[tool.pytest.ini_options]`, and no `ruff.toml`, `pytest.ini`,
`setup.cfg`, or `tox.ini` in the repo. Ruff and pytest therefore run on their built-in defaults.

## config.yaml is read relative to the CWD

`DecisionPredicateGraph` looks for `config.yaml` in the process working directory, so a script launched
from `experiments/` picks up different `perc_var` / `decimal_threshold` values than the same script run
from the repo root. Pass `dpg_config` explicitly.
