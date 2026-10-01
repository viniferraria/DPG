---
type: Side Effect
title: decimal_threshold="auto" and its off-grid RuntimeWarning
description: decimal_threshold="auto" derives label rounding precision from the training data's own decimal places (max + 1) rather than a fixed default, resolved once per fit(); calling get_decimal_threshold() before fit() raises DPGError, and thresholds that don't land on that derived grid emit a RuntimeWarning.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/core.py
tags: [decimal_threshold, auto, warning, label-precision, config]
generated: { by: claude_code/claude-sonnet-5, at: 2026-09-18T00:00:00Z }
status: stable
---

# What changed

0.3.0 adds `decimal_threshold="auto"` as a valid value alongside a fixed non-negative integer. Instead
of a constant precision applied to every predicate label, `"auto"` derives precision **from the
training data actually passed to `fit()`**: the maximum number of decimal places across every value in
`X_train`, plus one:

```python
# dpg/core.py:337-368
def _resolve_decimal_threshold(self, X_train: Any) -> int:
    """Resolve ``decimal_threshold='auto'`` from data precision and audit it."""
    if self.decimal_threshold != "auto":
        self._resolved_decimal_threshold = int(self.decimal_threshold)
        return self._resolved_decimal_threshold

    values = np.asarray(X_train)
    precision = 0
    for value in values.reshape(-1):
        precision = max(precision, self._decimal_places(value))
    resolved = precision + 1
    self._resolved_decimal_threshold = resolved

    offending_features = set()
    for tree in self.model.estimators_:
        tree_ = tree.tree_
        for node, feature_index in enumerate(tree_.feature):
            if feature_index < 0:
                continue
            threshold = float(tree_.threshold[node])
            if not np.isclose(threshold, round(threshold, resolved), rtol=0.0, atol=1e-12):
                offending_features.add(self.feature_names[feature_index])
    if offending_features:
        names = ", ".join(sorted(offending_features))
        warnings.warn(
            "decimal_threshold='auto' found thresholds off the data-derived "
            f"{resolved}-decimal grid for feature(s): {names}; exact routing "
            "is retained and only predicate labels are rounded.",
            RuntimeWarning,
            stacklevel=2,
        )
    return resolved
```

The resolved integer is cached in `self._resolved_decimal_threshold` and re-derived on every `fit()`
call (it is only meaningful after fitting). `get_decimal_threshold()` reads that cache and raises if
it hasn't been resolved yet:

```python
# dpg/core.py:370-374
def get_decimal_threshold(self) -> int:
    """Return the effective precision used by the last trace extraction."""
    if self._resolved_decimal_threshold is None:
        raise DPGError("decimal_threshold='auto' is resolved when fit() is called")
    return self._resolved_decimal_threshold
```

# Why

A fixed `decimal_threshold` (config default `1e-9`'s companion is `6`; `config.yaml`'s is `3`) can be
too coarse or too fine depending on the dataset's own numeric precision. `"auto"` ties label rounding
to what the data actually looks like, at the cost of being a property of the *fit call*, not the
config alone — the same `dpg_config` can resolve to different precisions on different datasets, and
`get_decimal_threshold()` is meaningless (and errors) before any `fit()`.

The off-grid warning exists because sklearn computes split thresholds as midpoints between adjacent
training values, which are not guaranteed to land on any fixed decimal grid — rounding one to the
data-derived precision can still shift it slightly. The warning says exact **routing** is unaffected
(splits still follow the model's real threshold in `execution_trace` / local explanations — see
[exact-routing-label-shift](exact-routing-label-shift.md)); only the **label** is rounded, so it may
not exactly equal the true threshold for the reported feature(s).

# Who is affected

Anyone setting `decimal_threshold: "auto"` in `dpg_config`, especially with irregular/high-precision
feature values (e.g. sensor data, unrounded ratios) where split thresholds don't fall near round
decimal values. Also anyone calling `get_decimal_threshold()` on a builder that hasn't been fit yet.

# Reproduction

```python
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dpg.core import DecisionPredicateGraph

X = np.array([[0.1234567], [0.99], [0.5], [0.2]] * 3)
y = np.array([0, 1, 0, 1] * 3)
model = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=0).fit(X, y)

dpg_config = {
    "dpg": {
        "default": {"perc_var": 1e-9, "decimal_threshold": "auto", "n_jobs": 1},
        "graph_construction": {"mode": "aggregated_transitions"},
        "visualization": {},
    },
}
builder = DecisionPredicateGraph(model, feature_names=["f0"], target_names=["0", "1"], dpg_config=dpg_config)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    builder.fit(X)
    for w in caught:
        print("WARNING:", w.category.__name__, "-", str(w.message))
print("resolved decimal_threshold:", builder.get_decimal_threshold())
```

Run with `uv run python` from the repo root. Real output:

```text
WARNING: RuntimeWarning - decimal_threshold='auto' found thresholds off the data-derived 8-decimal grid for feature(s): f0; exact routing is retained and only predicate labels are rounded.
resolved decimal_threshold: 8
```

And calling `get_decimal_threshold()` before any `fit()`:

```python
b = DecisionPredicateGraph(model, feature_names=["f0"], target_names=["0", "1"], dpg_config=dpg_config)
b.get_decimal_threshold()
```

```text
dpg.exceptions.DPGError: decimal_threshold='auto' is resolved when fit() is called
```

# Workaround

If stable, dataset-independent label precision matters more than data-adaptive precision, use a fixed
integer `decimal_threshold` instead of `"auto"`. The off-grid warning itself has no workaround beyond
accepting that labels are a rounded display of the threshold, not the threshold itself — routing
remains exact wherever exact routing is used (see
[exact-routing-label-shift](exact-routing-label-shift.md); note `aggregated_transitions` routes on the
rounded value regardless of how that value was derived).

# File:line citations (HEAD `276a503`)

- `dpg/core.py:170-175` — `decimal_threshold` validation (`"auto"` or non-negative integer).
- `dpg/core.py:233-235` — `_resolved_decimal_threshold` initialized from a fixed int, or `None` for
  `"auto"` pending `fit()`.
- `dpg/core.py:304` — `_resolve_decimal_threshold` called from `_extract_trace_log` at the start of
  every `fit()`.
- `dpg/core.py:324-336` — `_decimal_places`, the per-value precision helper.
- `dpg/core.py:337-368` — `_resolve_decimal_threshold`, resolution and the `RuntimeWarning`.
- `dpg/core.py:370-374` — `get_decimal_threshold`, the pre-fit `DPGError`.
- CHANGELOG.md 0.3.0 "Added": "Added `decimal_threshold='auto'`, which derives precision from the data
  and warns when a tree threshold is off the derived grid."

# Related

- [exact-routing-label-shift](exact-routing-label-shift.md)
- [DPG config resolution order](/conventions/config-resolution.md)
- [context-order-validation-errors](context-order-validation-errors.md)
