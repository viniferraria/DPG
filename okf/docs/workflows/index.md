# Workflows

Operational playbooks: how to set the project up, verify it, ship it, and run it.

* [Development environment setup](development-setup.md) - Provision a working DPG development environment with uv, the optional dev and docs dependency sets, and the system Graphviz binary the renderers shell out to.
* [Testing, linting, and type checking](testing-and-linting.md) - Run pytest, ruff, and mypy exactly as the Feature CI workflow does, including the repo-root working-directory requirement and the narrow mypy scope.
* [CI pipelines](ci-pipelines.md) - What each file in .github/workflows actually does — triggers, jobs, installed toolchains — and which of them are currently commented out and never run.
* [Running the DPG CLI entrypoints](cli-entrypoints.md) - How to run examples/run_dpg_standard.py and examples/run_dpg_custom.py, what flags they accept, what they write to disk, and why the declared dpg console script does not work.

## The short version

```bash
uv sync --group dev                              # setup (needs `dot` on PATH)
uv run ruff check $(git ls-files '*.py')         # lint, tracked files only
uv run mypy dpg/ metrics/                        # type check, these two packages only
uv run pytest                                    # must run from the repo root
```

## See also

* [Experiment suites](/experiments/index.md) - Longer-running research runners, outside the CI gates.
