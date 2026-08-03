---
type: Convention
title: DPG config resolution order
description: How DecisionPredicateGraph.__init__ picks configuration — explicit dpg_config, then config.yaml relative to the CWD, then DEFAULT_DPG_CONFIG — and where those sources disagree.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [configuration, defaults, yaml, omegaconf, reproducibility]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`DecisionPredicateGraph.__init__` resolves four behavioral settings (`perc_var`,
`decimal_threshold`, `n_jobs`, `graph_construction.mode`) plus a `visualization` block. The source it
reads from depends on arguments *and* on the process working directory. See
[/modules/dpg-core.md](/modules/dpg-core.md).

# Behavior

## Resolution order

1. **`dpg_config` argument** — if not `None`, it is used as the config object; `config_file` is never
   opened.
2. **`config_file`** (default `"config.yaml"`) — if the argument is truthy and `os.path.exists(config_file)`,
   it is loaded with `yaml.safe_load`. The path is **relative, so it resolves against the current
   working directory**. If the file is missing, the constructor prints
   `Config file not found at '<path>'. Using built-in defaults.`
3. **`DEFAULT_DPG_CONFIG`** — used when no `dpg_config` was given and the file was absent or parsed
   to `None` (e.g. an empty YAML file).

## Normalization

| Input shape | Handling |
|---|---|
| OmegaConf `DictConfig` | `OmegaConf.to_container(config, resolve=True)` when `omegaconf` is importable (`HAS_OMEGACONF`) |
| Object exposing `.to_dict()` | Converted via `config.to_dict()` |
| Plain `dict` | Used as-is |

## Per-key fallback

Resolution is **per key**, not whole-object: each value is read with
`config["dpg"]["default"].get(key, DEFAULT_DPG_CONFIG["dpg"]["default"][key])`, and the mode with
`config["dpg"]["graph_construction"].get("mode", "aggregated_transitions")`. A partial config
therefore inherits the missing keys from `DEFAULT_DPG_CONFIG` — but a key present with an explicit
`null` resolves to `None` and raises `DPGConfigurationError.missing_perc_var()`,
`.missing_decimal_threshold()`, or `.missing_n_jobs()`. `visualization` falls back to
`DEFAULT_DPG_CONFIG["dpg"]["visualization"]`, which is `{}`.

# Gotchas

## Where the two sources disagree

Read from `config.yaml` at the repo root and `DEFAULT_DPG_CONFIG` in `dpg/core.py`:

| Key | `config.yaml` | `DEFAULT_DPG_CONFIG` | Effect of the difference |
|---|---|---|---|
| `dpg.default.perc_var` | `0.0001` | `0.000000001` (1e-9) | Repo file filters far more aggressively |
| `dpg.default.decimal_threshold` | `3` | `6` | Repo file rounds thresholds harder, merging more nodes |
| `dpg.default.n_jobs` | `-1` | `-1` | No difference (both fan out across all cores) |
| `dpg.graph_construction.mode` | *absent* | `"aggregated_transitions"` | Falls back to the default mode |
| `dpg.visualization` | `graph_attrs` (`bgcolor: white`, `rankdir: R`), `node_attrs` (`shape: box`, `fillcolor: #ffc3c3`), `class_node` block | `{}` | With defaults, `generate_dot` drops `bgcolor`/`rankdir`/`shape` and passes `fillcolor=None` |

`n_jobs` also selects the tracing entry point in `_extract_trace_log`: `n_jobs == 1` dispatches
`tracing_ensemble` (generator), anything else dispatches `tracing_ensemble_parallel`.

## CWD dependence

Because `config_file` is a bare relative name, the *same code* produces different graphs depending on
where the process was launched: run from the repo root it picks up `config.yaml`
(`perc_var=0.0001`, `decimal_threshold=3`); run from any other directory it silently falls back to
`DEFAULT_DPG_CONFIG` (`perc_var=1e-9`, `decimal_threshold=6`) after printing the "not found" notice.
Since `decimal_threshold` controls label rounding and node identity, this changes node counts, not
just filtering.

## Recommended practice

Always pass `dpg_config` explicitly in tests, experiments, and library code:

```python
from dpg.core import DecisionPredicateGraph

config = {
    "dpg": {
        "default": {"perc_var": 0.0001, "decimal_threshold": 3, "n_jobs": 1},
        "graph_construction": {"mode": "aggregated_transitions"},
        "visualization": {},
    }
}
dpg = DecisionPredicateGraph(model, feature_names, target_names, dpg_config=config)
```

Passing `config_file=""` (falsy) skips the file lookup entirely and uses `DEFAULT_DPG_CONFIG`
without the not-found print. Unsupported `mode` strings fail fast — see
[/conventions/graph-construction-modes.md](/conventions/graph-construction-modes.md).
