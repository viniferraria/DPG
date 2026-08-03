# Conventions

Cross-module contracts. Breaking one of these breaks modules that never import each other.

* [Event label contract](label-contract.md) - The exact string formats DPG uses for graph node labels, and every module that parses them.
* [DPG config resolution order](config-resolution.md) - How DecisionPredicateGraph.__init__ picks configuration — explicit dpg_config, then config.yaml relative to the CWD, then DEFAULT_DPG_CONFIG — and where those sources disagree.
* [Graph construction modes](graph-construction-modes.md) - The two supported dpg.graph_construction.mode values — aggregated_transitions (variant-level filtering) and execution_trace (edge-level filtering) — their thresholds, failure mode, and effect on local-explanation validity flags.

## See also

* [The DPG pipeline](/pipeline.md) - Why node identity is the label text in the first place.
* [dpg.core](/modules/dpg-core.md) - Where all three conventions are implemented.
