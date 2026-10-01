---
type: Playbook
title: CI pipelines
description: What each file in .github/workflows actually does — triggers, jobs, installed toolchains — and which of them are currently commented out and never run.
resource: https://github.com/viniferraria/DPG/blob/main/.github/workflows
tags: [ci, github-actions, workflows, uv, sphinx, automation]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# When to use

Before pushing a branch, when a check is unexpectedly missing, or when editing anything the runners
install. The commands these workflows run locally are in
[/workflows/testing-and-linting.md](/workflows/testing-and-linting.md); the toolchain they assume is in
[/workflows/development-setup.md](/workflows/development-setup.md).

## Overview

| Workflow file | Trigger | What it runs |
|---|---|---|
| `feature-pr.yaml` | `push` to `feature/**` or `feature*` | `uv sync --locked --dev`, ruff, mypy, pytest, then `gh pr create` into `develop` |
| `docs.yml` | `push` to `main`, `pull_request` targeting `main` | apt `graphviz`, `pip install ".[docs]"`, `sphinx-build`, upload HTML artifact |
| `claude.yml` | `pull_request` (any base branch) | `anthropics/claude-code-action@v1`, no inputs |
| `ci.yml` | **none — file is entirely commented out** | nothing |
| `formater.yaml` | **none — file is entirely commented out** | nothing |
| `telegram-on-commit.yml` | **none — file is entirely commented out** | nothing |

# Steps

## feature-pr.yaml — "Feature CI"

The only quality gate that actually executes.

- **Trigger:** `on.push.branches: ["feature/**", "feature*"]`. Not triggered by pull requests, and not
  by pushes to `main` or `develop`.
- **Permissions:** `contents: read`, `pull-requests: write`.
- **Job `test`** on `ubuntu-latest`: `actions/checkout@v7`; `astral-sh/setup-uv@v9.0.0` with
  `python-version: "3.11"`; `uv sync --locked --dev`; `uv run ruff check $(git ls-files '*.py')`;
  `uv run mypy dpg/ metrics/`; `uv run pytest`; then open a PR from the pushed branch into `develop`
  with `gh pr create`, using `GH_TOKEN: ${{ github.token }}`.

Reproduce the gate locally with the four commands in
[/workflows/testing-and-linting.md](/workflows/testing-and-linting.md).

The PR step first queries for an existing open PR (`gh pr list --base develop --head "$GITHUB_REF_NAME"
--state open --json number --jq '.[0].number'`) and exits 0 if one is found. Failures from `gh pr create` fail the job, **except** a message matching `No commits between`, which is
treated as a normal no-op with a `::notice::`. Any other error (e.g. the repo blocking Actions from
creating PRs) produces `::error::` and a red run — this was previously swallowed into a warning.

## docs.yml — "Docs"

- **Trigger:** `on.push.branches: [main]` and `on.pull_request.branches: [main]`.
- **Job `build`** on `ubuntu-latest`: `actions/checkout@v4`; `actions/setup-python@v5` with
  `python-version: "3.11"`; `sudo apt-get install -y graphviz` (autoapi imports modules that reference
  it); `pip install ".[docs]"` — **pip + the `docs` extra**, not uv; `sphinx-build -b html docs/
  docs/_build/html` tee'd to `/tmp/sphinx_out.txt`; `actions/upload-artifact@v4` publishing
  `docs/_build/html/` as `html-docs` with `retention-days: 7`.

**Sphinx warnings are treated as errors, but by grep, not by `-W`.** The step captures
`PIPESTATUS[0]`, fails on a non-zero Sphinx exit, then fails again if any line matching `^WARNING:`
survives filtering out `Unknown type: placeholder` (a known sphinx-autoapi/astroid limitation with
C-extension types). `--keep-going` is deliberately not used, because extension errors make Sphinx exit
2 with partial output that would still be uploaded and mask the real failure.

Reproduce locally:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs/ docs/_build/html 2>&1 | tee /tmp/sphinx_out.txt
grep -v "Unknown type: placeholder" /tmp/sphinx_out.txt | grep "^WARNING:"   # must print nothing
```

## claude.yml — "Claude"

`on: pull_request` (all base branches, default activity types); permissions `contents: read`,
`pull-requests: write`; job `claude` on `ubuntu-latest` with a single step running
`anthropics/claude-code-action@v1`. The step declares no `with:` block at all — no `prompt`, no
checkout step, no credential secret — and the job has no `id-token: write` permission.

## Disabled workflows

`ci.yml`, `formater.yaml`, and `telegram-on-commit.yml` are **100% commented out** — every line,
including `name:` and `on:`. GitHub parses them as empty documents: no triggers, never executed. Their
commented-out intent:

- `ci.yml` — `push` to `main` + `pull_request`; `actions/setup-python@v5` with a single-version matrix
  `["3.11"]`; `pip install -r requirements.txt` plus `pip install pytest`; `pytest -q`.
- `formater.yaml` — `on: [push]`; matrix `["3.10", "3.11"]`; `uv sync --locked --dev`; `uv run ruff
  check $(git ls-files '*.py') -v`.
- `telegram-on-commit.yml` — `push` to `main`; curl to the Telegram Bot API using
  `secrets.TELEGRAM_BOT_TOKEN` / `secrets.TELEGRAM_CHAT_ID`.

# Gotchas

## No CI runs on main, develop, or pull requests into develop

The only test-running workflow triggers on `push` to `feature/**` / `feature*`. Pushing directly to
`main` or `develop`, or opening a PR into `develop`, runs **no** ruff/mypy/pytest — `docs.yml` covers
`main` only, and only for the Sphinx build. Merges into `develop` are effectively unverified unless the
feature branch was pushed first.

## requirements.txt has no automated consumer

Because `ci.yml` is commented out, nothing in `.github/workflows/` installs from `requirements.txt`
today. It is still a real drift hazard: its `==` pins (`numpy==2.1.3`, `scikit-learn==1.5.2`,
`matplotlib==3.9.2`, …) are maintained by hand against `[project.dependencies]`'s `>=` floors and
`uv.lock`. If `ci.yml` is ever re-enabled, that file must be brought back in sync first — and it
currently lists `matplotlib-stubs`, which `[dependency-groups].dev` explicitly warns shadows
matplotlib's inline `py.typed` annotations.

## Action version pinning is inconsistent

`feature-pr.yaml` uses `actions/checkout@v7` and `astral-sh/setup-uv@v9.0.0`; `docs.yml` uses
`actions/checkout@v4`. Nothing is SHA-pinned, so floating major tags can shift under you.

## docs.yml installs with pip, so uv.lock is not exercised there

`pip install ".[docs]"` resolves `[project.optional-dependencies].docs` fresh on every run, ignoring
`uv.lock`. A new upstream Sphinx or autoapi release can therefore break the docs job with no repo
change. `[dependency-groups].docs` duplicates the same seven packages and is what `uv sync --group docs`
would use — keep the two lists identical.

## .readthedocs.yaml points at a file that does not exist

`.readthedocs.yaml` (build `ubuntu-22.04`, Python `3.11`) installs
`- requirements: docs/requirements.txt`, but **`docs/requirements.txt` is not present in the repo**, so
Read the Docs builds will fail at the install step. It also intentionally omits `fail_on_warning`,
leaving the warning-as-error gate solely to `docs.yml`.
