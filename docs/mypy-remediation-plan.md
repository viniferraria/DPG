# mypy remediation plan — matplotlib colormap API drift

**Status:** proposed
**Scope:** `uv run mypy dpg/ metrics/` — currently **11 errors in 3 files**

---

## TL;DR

Of the 11 errors, **3 are real runtime bugs** and 8 are false positives from
third-party stubs. The `legacy` theme is currently **broken at import-time of its
own factory** on the matplotlib version installed locally.

Underneath sits a worse problem: the local venv has drifted to matplotlib
**3.11.1** while `uv.lock` still pins **3.10.9**, and the two versions require
**mutually exclusive source code**. The tree currently satisfies 3.11.1 and would
*fail* CI on 3.10.9.

---

## Verification method

Every claim below was checked against a real interpreter, not inferred from the
error text. Version comparisons used throwaway environments
(`uv run --isolated --no-project --with matplotlib==<v>`) so the project venv was
never disturbed.

---

## Finding 1 — `cm.get_cmap` is a real bug (3 errors)

```
dpg/themes.py:258     error: Module has no attribute "get_cmap"  [attr-defined]
dpg/themes.py:259     error: Module has no attribute "get_cmap"  [attr-defined]
dpg/visualizer.py:1378 error: Module has no attribute "get_cmap"  [attr-defined]
```

`matplotlib.cm.get_cmap` was **removed in matplotlib 3.11**. This is not a stub
gap — the attribute is gone at runtime:

```
matplotlib 3.10.9: cm.get_cmap present = True
matplotlib 3.11.1: cm.get_cmap present = False
```

Confirmed end-to-end — this is a live crash, not a theoretical one:

```
>>> resolve_theme_context(theme="legacy")
AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'
```

### Blast radius

| Site | Reachability | Impact |
|---|---|---|
| `themes.py:258` `"community_cmap": cm.get_cmap("tab20")` | **eager** — evaluated while building the dict | `resolve_theme_context(theme="legacy")` raises; the whole legacy theme is unusable |
| `themes.py:259` `"class_cmap": lambda n: cm.get_cmap("viridis", n)` | lazy | unreachable today — line 258 raises first |
| `visualizer.py:1378` `_feature_color_map` | **dead code** | nothing calls it; `plot_lrc_vs_rf_importance:1452` uses `theme_context["feature_color_map"]` instead |

Both `get_cmap` sites in `themes.py` are inside the `theme_name == "legacy"`
branch, so the default `dpg` theme is unaffected — which is why nothing was
noticed.

`_feature_color_map` is worth flagging separately: it is an unused, broken
duplicate of the working `themes.feature_color_map` (`themes.py:192`), which
builds colors via `discrete_palette` + `mcolors.to_rgba` and never touches
`get_cmap`. It became a duplicate in `79d5cbc`, which rewrote it to inline
`cm.get_cmap` instead of delegating.

### Why the test suite is green

**229 tests pass.** No test constructs the `legacy` theme, and nothing calls
`_feature_color_map`. These paths have zero coverage.

---

## Finding 2 — matplotlib version straddles a breaking change

> **Correction.** An earlier revision of this document claimed the venv had
> "drifted" from `uv.lock`. That was wrong. `uv.lock` contains **two** matplotlib
> entries — `3.10.9` for `python_full_version < '3.11'` and `3.11.1` for `>= 3.11`
> — so the installed 3.11.1 on this Python 3.11 venv is the lock being honored
> correctly. `uv sync --locked --dev` passes.

`pyproject.toml` declares `matplotlib>=3.9` with **no upper bound**, and
`requires-python = ">=3.10"`. The project therefore spans the 3.11 boundary where
`cm.get_cmap` was removed:

| Python | matplotlib | `cm.get_cmap` | Consequence |
|---|---|---|---|
| 3.10 | 3.10.9 | present | legacy theme works |
| ≥3.11 | 3.11.1 | **removed** | legacy theme raises `AttributeError` |

CI (`feature-pr.yaml`) pins `python-version: "3.11"`, so CI *does* get 3.11.1 and
would surface Finding 1 — the reason it hasn't is purely the missing test
coverage below. (`ci.yml`'s 3.10/3.11/3.12 matrix is currently commented out.)

The two versions still demand opposite source, because the stubs moved in both
directions at once:

| Symbol | 3.10.9 stub | 3.11.1 stub |
|---|---|---|
| module-level `get_cmap` | declared ✅ | **removed** ❌ |
| `cm.Blues` / `tab10` / `tab20` | **absent** ❌ | declared ✅ |

With `warn_unused_ignores = true`, an ignore that is required on one version is
an *error* on the other. The source is consequently tuned to 3.11.1 only.
Measured with mypy run against each version:

```
--- mypy under matplotlib 3.10.9 ---
dpg/visualizer.py:524:  error: "Spine" not callable          [operator]
dpg/visualizer.py:1053: error: "Spine" not callable          [operator]
dpg/visualizer.py:2122: error: Unused "type: ignore" comment [unused-ignore]
dpg/visualizer.py:2127: error: "Spine" not callable          [operator]
--- mypy under matplotlib 3.11.1 ---
Success: no issues found in 2 source files
```

This is **not** currently a CI failure — CI type-checks on Python 3.11 only, so it
sees 3.11.1. It only bites a contributor developing on Python 3.10.

Accepting that is reasonable; the alternative is dropping `warn_unused_ignores`
or adding a 3.10 mypy job. What matters is that it is a deliberate choice rather
than an accident. Step 1's `matplotlib.colormaps[...]` rewrite removes the
`attr-defined` half of the problem regardless.

---

## Finding 3 — the remaining 8 errors are false positives

Both clusters are stub over-narrowness. Neither indicates a defect; neither
should be "fixed" by changing working code.

### `visualizer.py:2120` — 6 errors, `Norm.__call__` typing

```python
text_color = colors["paper"] if im.norm(value) > 0.55 else colors["charcoal"]
```

matplotlib 3.11 types `Norm.__call__` as taking/returning array-likes, so mypy
sees `float > _Buffer`, `str > float`, and friends. At runtime the scalar path
is exactly what matplotlib supports:

```
im.norm(5)     -> np.float64(0.5)
im.norm(5)>0.55 -> False   (bool)
```

### `core.py:212` — 1 error, `no-any-return`

```python
return pd.DataFrame(log, columns=["case:concept:name", "concept:name"])
```

In `pandas-stubs` 3.0.3 **both** `DataFrame.__init__` overloads are declared
`-> Any`, so *every* `pd.DataFrame(...)` call is untyped regardless of arguments
— not a resolution quirk of these particular args:

```
reveal_type(pd.DataFrame)                       -> Overload(def (...) -> Any, def (...) -> Any)
reveal_type(pd.DataFrame([[1,2]], columns=[…])) -> Any
```

There are 20 such construction sites in the mypy scope. Only this one trips
`warn_return_any`, because it is the one returning the frame directly from a
function declared `-> pd.DataFrame`; the others assign to a local or live in
functions already declared `-> Any` (`local_path_dataframe`, `_get_node_metrics`,
`_get_edge_metrics`).

Consequence for the fix: no argument shape will make the stub yield a real
`DataFrame`, so `cast(pd.DataFrame, ...)` is the right remedy rather than
restructuring the call.

**pandas-stubs is otherwise load-bearing and should be kept.** pandas ships no
`py.typed` and `pandas` is absent from the `ignore_missing_imports` override
list, so removing the stubs turns `import pandas` into an `import-untyped` error
across six modules. It also catches genuine mistakes — bad `groupby` keywords and
argument types, and it types `read_csv` → `DataFrame` and `df[col]` →
`Series[Any]`. The constructor is its one blind spot.

---

## Plan

Ordered by risk. Steps 1–2 are the actual bug fix; 3–4 close the hole that let it
through; 5 silences the false positives.

### 1. Replace the removed API with the version-agnostic one — `dpg/themes.py`

`matplotlib.colormaps` is the documented replacement, present and **typed in
both 3.10.9 and 3.11.1**, so it fixes the crash *and* the type error on every
supported version.

| Current | Replacement |
|---|---|
| `cm.get_cmap("tab20")` | `matplotlib.colormaps["tab20"]` |
| `cm.get_cmap("viridis", n)` | `matplotlib.colormaps["viridis"].resampled(n)` |
| `cm.Blues` / `cm.Greys` | `matplotlib.colormaps["Blues"]` / `["Greys"]` |
| `cm.tab20(x)` / `cm.tab10(x)` | `matplotlib.colormaps["tab20"](x)` / `["tab10"](x)` |

Equivalence verified numerically on 3.10.9, where both APIs still exist:

```
get_cmap(name, lut) == colormaps[name].resampled(lut) : True   (N: 5 == 5)
get_cmap(name)      == colormaps[name]                : True
cm.Blues            == colormaps["Blues"]             : True
```

Converting `cm.Blues`/`tab10`/`tab20` too is what makes the module compile
cleanly on **both** matplotlib versions and lets the stale
`# type: ignore[attr-defined]` comments — and the now-inaccurate comment above
them claiming "matplotlib ships no stub entries" — be deleted for good rather
than re-added.

### 2. Delete the dead duplicate — `dpg/visualizer.py:1376`

Remove `_feature_color_map` outright. It has no callers, and
`themes.feature_color_map` is the live, working implementation. Deleting is
preferable to porting a second copy of the same logic to `colormaps[...]`.

*If* it should stay as a public-ish helper, make it delegate:
`return feature_color_map(features)` — its behavior before `79d5cbc`.

### 3. Bound the dependency — `pyproject.toml`

`matplotlib>=3.9` silently admits the breaking 3.11 change. Either:

- **(a)** widen deliberately: keep `>=3.9`, and make the code work across the
  range — which step 1 achieves; or
- **(b)** pin the tested range: `matplotlib>=3.9,<3.12`.

Recommend **(a) plus running CI against the upper bound** (below), since step 1
already removes the incompatibility; (b) alone just defers the problem.

Then re-lock so `uv.lock` and the venv agree again: `uv lock --upgrade-package matplotlib`.

### 4. Close the coverage gap — `tests/`

Two cheap tests that would have caught this at authoring time:

```python
@pytest.mark.parametrize("theme", ["dpg", "legacy"])
def test_resolve_theme_context_builds(theme):
    ctx = resolve_theme_context(theme=theme)
    assert ctx["class_cmap"](3) is not None      # exercises the lazy lambda too
    assert ctx["community_cmap"] is not None
```

The `legacy` branch has no test at all today; the `class_cmap` lambda needs
calling explicitly, since building the dict alone would not have caught
`themes.py:259`.

Also worth considering: a CI matrix entry that resolves dependencies
*unlocked* (`uv sync --upgrade`) so upstream removals surface as a failing job
rather than at a user's install.

### 5. Suppress the two false-positive clusters

Only after 1–4, and each with a comment naming the stub as the cause:

- `visualizer.py:2120` — `# type: ignore[operator,arg-type]` on the `im.norm(...)`
  comparison. Alternatively hoist `float(im.norm(value))`, which is clearer and
  needs no ignore — behavior-identical, since the value is already `np.float64`.
- `core.py:212` — annotate the local (`log_df: pd.DataFrame = pd.DataFrame(...)`)
  or `cast(pd.DataFrame, ...)`. Prefer this over relaxing `warn_return_any`.

---

## Verification

1. `uv run mypy dpg/ metrics/` → `Success: no issues found`.
2. **Both matplotlib versions** — the point of the exercise:
   `uv run --isolated --no-project --with matplotlib==3.10.9 ... mypy dpg/ metrics/`
   and the same for `3.11.1`. Neither may report errors, and neither may report
   *unused* ignores.
3. `uv run pytest -q` → 229 passed, plus the new theme tests.
4. Runtime proof the crash is gone:
   `python -c "from dpg.themes import resolve_theme_context as r; [r(theme=t)['class_cmap'](3) for t in ('dpg','legacy')]"`
5. `uv run ruff check $(git ls-files '*.py')` → clean.
6. `uv sync --locked --dev` must still succeed — i.e. `uv.lock` was re-locked and
   committed in step 3.
