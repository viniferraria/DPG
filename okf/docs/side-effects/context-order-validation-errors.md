---
type: Side Effect
title: context_order and decimal_threshold fail fast on invalid config
description: DecisionPredicateGraph.__init__ raises DPGError for a non-positive/non-integer context_order, for context_order="auto" or >1 outside execution_trace mode, and for a non-integer/negative decimal_threshold; resolve_context_order raises ValueError for a bad or insufficient max_k cap.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [validation, context_order, decimal_threshold, dpg-k, config, error-handling]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed

0.3.0 adds `context_order` (DPG-k) to `dpg_config["dpg"]["graph_construction"]`, and validates it —
and revalidates `decimal_threshold` in the same pass — eagerly in `DecisionPredicateGraph.__init__`,
before any data is seen:

```python
# dpg/core.py:170-216
if self.decimal_threshold != "auto" and (
    isinstance(self.decimal_threshold, bool)
    or not isinstance(self.decimal_threshold, int)
    or self.decimal_threshold < 0
):
    raise DPGError("decimal_threshold must be a non-negative integer or 'auto'")
...
if self.context_order != "auto" and (
    isinstance(self.context_order, bool)
    or not isinstance(self.context_order, int)
    or self.context_order <= 0
):
    raise DPGError("context_order must be a positive integer or 'auto'")
if (
    (
        self.context_order == "auto"
        or (isinstance(self.context_order, int) and self.context_order > 1)
    )
    and self.graph_construction_mode != "execution_trace"
):
    raise DPGError(
        "context_order='auto' or context_order > 1 requires "
        "mode='execution_trace'"
    )
```

(This block is duplicated verbatim at `dpg/core.py:200-216` — both copies run and raise identically;
it does not change observable behavior, just runs the same check twice.)

Separately, `dpg/context_order.py`'s `resolve_context_order` — used when `context_order="auto"` — takes
an optional `max_k` cap and raises `ValueError` both for an invalid cap and for a cap too small to
eliminate all trace recombination:

```python
# dpg/context_order.py:88-118
if max_k is not None and (
    isinstance(max_k, bool) or not isinstance(max_k, int) or max_k < 1
):
    raise ValueError("max_k must be a positive integer or None")
...
for k in range(1, max_k + 1):
    violations = path_violations(materialized, k)
    history[k] = violations
    if violations == 0:
        return k, history
raise ValueError(
    f"no context order <= max_k={max_k} eliminates all global trace violations"
)
```

# Why

`context_order > 1` only has a defined graph-construction path in `execution_trace` mode
(`discover_dfg_context`); `aggregated_transitions` has no contextual variant. Rather than silently
ignoring `context_order` in the wrong mode (which would make DPG-k a no-op with no explanation),
construction fails immediately with a message naming the exact requirement. Likewise,
`resolve_context_order` is documented as needing to "still succeed within [a supplied] cap" — returning
an order whose history still shows violations would silently defeat the whole point of `"auto"`
(zero-recombination resolution), so it raises instead of returning a wrong answer.

# Who is affected

Anyone constructing a `DecisionPredicateGraph` / `DPGExplainer` with:
- `context_order` set to zero, negative, a float, a bool, or anything else that isn't a positive `int`
  or the string `"auto"`.
- `context_order="auto"` or an explicit `context_order > 1` together with
  `graph_construction.mode != "execution_trace"` (i.e. the default `"aggregated_transitions"`).
- a non-`"auto"`, negative, non-integer, or boolean `decimal_threshold`.

And anyone calling `resolve_context_order(traces, max_k=...)` directly with an invalid `max_k`, or a
`max_k` too small for the observed traces to resolve without recombination.

# Reproduction

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dpg.core import DecisionPredicateGraph, DPGError
from dpg.context_order import resolve_context_order

X = np.array([[0.1], [0.9], [0.5], [0.2]])
y = np.array([0, 1, 0, 1])
model = RandomForestClassifier(n_estimators=2, max_depth=1, random_state=0).fit(X, y)

def base(**overrides):
    gc = {"mode": "aggregated_transitions", "context_order": 1}
    gc.update(overrides)
    return {"dpg": {"default": {"perc_var": 1e-9, "decimal_threshold": 3, "n_jobs": 1},
                     "graph_construction": gc, "visualization": {}}}

try:
    DecisionPredicateGraph(model, feature_names=["f0"], target_names=["0", "1"],
                            dpg_config=base(context_order=2))
except DPGError as e:
    print("1)", e)

try:
    DecisionPredicateGraph(model, feature_names=["f0"], target_names=["0", "1"],
                            dpg_config=base(context_order="auto"))
except DPGError as e:
    print("2)", e)

try:
    DecisionPredicateGraph(model, feature_names=["f0"], target_names=["0", "1"],
                            dpg_config=base(mode="execution_trace", context_order=0))
except DPGError as e:
    print("3)", e)

try:
    resolve_context_order([("a", "b", "Class 0")], max_k=0)
except ValueError as e:
    print("4)", e)

traces = [("A", "B", "C", "Class 0"), ("X", "B", "C", "Class 1")]
try:
    resolve_context_order(traces, max_k=1)
except ValueError as e:
    print("5)", e)
print("6) uncapped resolves to:", resolve_context_order(traces))
```

Run with `uv run python` from the repo root. Real output:

```text
1) context_order='auto' or context_order > 1 requires mode='execution_trace'
2) context_order='auto' or context_order > 1 requires mode='execution_trace'
3) context_order must be a positive integer or 'auto'
4) max_k must be a positive integer or None
5) no context order <= max_k=1 eliminates all global trace violations
6) uncapped resolves to: (3, {1: 2, 2: 1, 3: 0})
```

(Case 5's two traces both pass through predicate `B` then `C` before diverging in their history —
`A-B-C` and `X-B-C` — so at `k=1`, `C`'s context is just `C` for both, and the pooled graph would allow
the unobserved recombination `A-B-C-1` / `X-B-C-0`; only `k=3` observes the full distinguishing
history, matching `resolve_context_order`'s uncapped answer in case 6.)

# Workaround

Set `graph_construction.mode = "execution_trace"` whenever using `context_order > 1` or `"auto"`.
Supply a `decimal_threshold` that is `"auto"` or a non-negative integer. For `resolve_context_order`,
either omit `max_k` (defaults to the longest observed trace, always sufficient) or raise the supplied
cap until it's large enough to observe zero violations — case 6 above shows how to check the
uncapped answer.

# File:line citations (HEAD `276a503`)

- `dpg/core.py:170-175` — `decimal_threshold` type/range validation.
- `dpg/core.py:183-199` and `200-216` — duplicated `context_order` validation and the
  `execution_trace`-only requirement.
- `dpg/context_order.py:88-91` — `max_k` validation.
- `dpg/context_order.py:113-118` — insufficient-`max_k` `ValueError`.
- `dpg/exceptions.py` — `DPGError` base class used for all `core.py` config-validation raises.

# Related

- [decimal-threshold-auto-warning](decimal-threshold-auto-warning.md)
- [context-order-pairwise-crash](context-order-pairwise-crash.md)
- [Graph construction modes](/conventions/graph-construction-modes.md)
