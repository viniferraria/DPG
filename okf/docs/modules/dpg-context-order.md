---
type: Python Module
title: dpg.context_order — resolving DPG-k
description: Trie-based resolution of the smallest context order (DPG-k) with no pooled-graph path recombination, without enumerating simple paths.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/context_order.py
tags: [python, module, context-order, dpg-k, trie, process-mining, new-in-0.3.0]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# Responsibility

`dpg/context_order.py` (115 lines) is new in 0.3.0. It answers one question cheaply: *given the
observed execution traces, what is the smallest context order `k` at which the pooled DPG contains
no path that wasn't actually observed?* This is the "DPG-k" feature — see
[/modules/dpg-core.md](/modules/dpg-core.md) for how `DecisionPredicateGraph` consumes the result via
`graph_construction.context_order` and `resolve_context_order`.

It exports `path_violations` and `resolve_context_order`. The module-private `_node_windows` builds
the contextual-node representation both functions share. No `dpg` or `metrics` imports — only
`math` and `collections`/`collections.abc` from the standard library.

# API

| Name | Signature | Returns |
|---|---|---|
| `_node_windows` | `(sequence: Sequence[str], k: float) -> list[object]` | One contextual node per trace position: `("sink", label)` for a `"Class "`/`"Pred "` terminal, `("ctx", tuple)` otherwise — the tuple is the last `k` labels ending at that position (or the full prefix when `k` is `math.inf`). |
| `path_violations` | `(traces: Iterable[Sequence[str]], k: float) -> int` | A diagnostic count of contextual nodes whose observed continuations disagree — see Behavior. `0` is the acceptance condition. |
| `resolve_context_order` | `(traces: Iterable[Sequence[str]], max_k: int \| None = None) -> tuple[int, dict[int, int]]` | `(resolved_k, history)` — the smallest `k` in `[1, max_k]` with zero violations, plus every tested `k`'s violation count. |

`traces` in both functions is an iterable of label sequences — in practice the per-`(sample, tree)`
predicate/leaf sequences produced by `DecisionPredicateGraph._trace_sequences`
([/modules/dpg-core.md](/modules/dpg-core.md)). Empty traces are dropped (`if trace`) before either
function does anything; an all-empty `traces` makes `resolve_context_order` return `(1, {1: 0})`
without running the algorithm at all.

# Behavior

## The phantom-path problem

A pooled directly-follows graph can be locally consistent — every edge has a real witness — while
still admitting an unobserved path once you chain several hops. `path_violations`'s own docstring
gives the canonical case: observing `A-B-C-D` and `X-B-C-E` gives every adjacent transition (`A→B`,
`B→C`, `C→D`, `X→B`, `C→E`) a witness, but pooling them into one graph also creates the edge set for
`A-B-C-E` — a path neither trace took.

## The trie / future-signature algorithm

`path_violations(traces, k)`:

1. Converts every trace to its contextual-node sequence via `_node_windows(sequence, k)`.
2. Inserts each contextual sequence into a shared trie (`trie_children`, `trie_terminal`,
   `trie_context` — parallel lists indexed by trie-node id, root is `0`).
3. Computes each trie node's **future signature** bottom-up: `(is_terminal, sorted tuple of
   (child_label, child_future_signature))`. Two trie nodes have the same signature iff every
   continuation reachable from them is identical.
4. Groups trie nodes by the **contextual node value** they represent (`trie_context[trie_node]`,
   i.e. the same key `generate_dot`/`discover_dfg_context` would collapse them under) and counts, per
   group, `max(0, len(distinct future signatures) - 1)`.

A group with more than one distinct future signature means: two different points in the observed
traces map onto the same pooled-graph node at this `k`, but they don't agree on what can follow —
pooling them would let the graph recombine a path nothing observed took. The sum over all groups is
the returned violation count. It is explicitly a **diagnostic count, not an exhaustive count of every
phantom path** the pooled graph could produce; `0` is the meaningful acceptance condition, not the
magnitude for nonzero values.

`resolve_context_order(traces, max_k=None)`:

- `max_k` must be `None` or a positive `int` (not `bool`) — anything else raises `ValueError("max_k
  must be a positive integer or None")` before any trace is touched.
- Default `max_k` is the longest observed trace length — at that order every observed prefix is
  distinct by construction, so resolution is guaranteed to succeed; the docstring calls this "a
  proof-based bound rather than a user-facing cap."
- It tries `k = 1, 2, …, max_k` in order, calling `path_violations` at each, and returns the first `k`
  with zero violations together with the full `{k: violations}` history for every `k` tried so far.
- If a caller supplies a smaller `max_k` and no `k` in range reaches zero violations,
  `resolve_context_order` raises `ValueError(f"no context order <= max_k={max_k} eliminates all
  global trace violations")` rather than silently returning a `k` that still has recombination.

# Gotchas

- **`resolve_context_order`'s `ValueError` is reachable from `DecisionPredicateGraph.fit()`.** When
  `context_order="auto"`, `fit()` calls `resolve_context_order(traces)` with no `max_k`, so the
  guaranteed-success default always applies there — the `ValueError` path only fires if a caller
  invokes `resolve_context_order` directly with an explicit `max_k` too small for the data.
- **Calling `DecisionPredicateGraph.fit()` with `context_order > 1` (explicit or `"auto"`-resolved)
  used to crash** in the DFG-building step downstream — `discover_dfg_context` passed two arguments
  to `itertools.pairwise`, which only accepts one. This is now fixed on `feature/first_runs` (see
  [/side-effects/context-order-pairwise-crash.md](/side-effects/context-order-pairwise-crash.md)), so
  `fit()` with `context_order > 1` works end to end. The example below still exercises
  `context_order.py` directly, since it is demonstrating the resolver in isolation, not `fit()`.
- `_node_windows` treats `math.isinf(k)` as "use the full prefix" — `resolve_context_order` never
  passes `math.inf` itself (its loop is over `int`s), but `path_violations` is a public function and
  accepts it if called directly.
- Sink nodes (`"Class "`/`"Pred "` labels) are never contextualized inside `_node_windows`, matching
  `DecisionPredicateGraph._context_node` in `dpg/core.py` — a terminal's identity doesn't change with
  `k`. Only non-terminal predicate nodes get a growing context window.

# Example (run against this repo at HEAD, `uv run python`)

```python
from dpg.context_order import path_violations, resolve_context_order

traces = [
    ("A", "B", "C", "D"),
    ("X", "B", "C", "E"),
]

for k in (1, 2, 3, 4):
    print(f"k={k}: path_violations={path_violations(traces, k)}")

order, history = resolve_context_order(traces)
print("resolve_context_order ->", order, history)

resolve_context_order(traces, max_k=1)   # raises ValueError: no order <= 1 works
resolve_context_order(traces, max_k=0)   # raises ValueError: bad max_k
```

Actual output:

```
k=1: path_violations=2
k=2: path_violations=1
k=3: path_violations=0
k=4: path_violations=0
resolve_context_order -> 3 {1: 2, 2: 1, 3: 0}
ValueError (max_k too small): no context order <= max_k=1 eliminates all global trace violations
ValueError (bad max_k): max_k must be a positive integer or None
```

`k=3` is the smallest order where `B`'s and `C`'s context includes enough of the preceding path
(`A-B`/`A-B-C` vs. `X-B`/`X-B-C`) that the two traces no longer share a contextual node at all, so
there is nothing left to recombine.
