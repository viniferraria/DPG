---
okf_version: "0.2"
---

# DPG knowledge bundle

Agent-agnostic knowledge for the [DPG](https://github.com/viniferraria/DPG) repository — Decision
Predicate Graph, a library that explains scikit-learn tree ensembles by mining them into a single
interpretable graph. Plain markdown with YAML frontmatter; no tooling required to read it.

Generated on 2026-08-02 by reading the source at version `0.1.6`.

## Start here

* [Project overview](overview.md) - What the DPG project is, how its two packages relate, and where to start reading.
* [The DPG pipeline](pipeline.md) - How DPG turns a fitted sklearn ensemble into a decision predicate graph via process mining, stage by stage.

## Conventions

Cross-module contracts. Read these before changing anything in `dpg/core.py`.

* [Event label contract](conventions/label-contract.md) - The exact string formats DPG uses for graph node labels, and every module that parses them.
* [DPG config resolution order](conventions/config-resolution.md) - How DecisionPredicateGraph.__init__ picks configuration — explicit dpg_config, then config.yaml relative to the CWD, then DEFAULT_DPG_CONFIG — and where those sources disagree.
* [Graph construction modes](conventions/graph-construction-modes.md) - The two supported dpg.graph_construction.mode values, their thresholds, failure mode, and effect on local-explanation validity flags.

## Modules

* [modules/](modules/) - Source-level reference for the six `dpg/` modules and the three `metrics/` modules.

## Concepts

* [concepts/](concepts/) - Explanation dataclasses and the faithfulness evaluation metric.

## Workflows

* [workflows/](workflows/) - Setup, testing/linting, CI pipelines, and the CLI entrypoints.

## Experiments

* [experiments/](experiments/) - The three research suites under `experiments/`: causal synthetic scenarios, local explanation, MONK's.

## History

* [log.md](log.md) - Change history for this bundle.
