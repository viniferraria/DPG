---
type: Python Module
title: dpg.themes and dpg.utils
description: Theme and palette resolution for every DPG plot, plus the small set of shared Graphviz and filesystem helpers.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/themes.py
tags: [python, module, theming, palette, matplotlib, graphviz, utils, dpg]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`dpg/themes.py` owns all color decisions: raw brand hexes, class/predicate palettes, matplotlib
rcParams bundles, colormaps, and the single resolver `resolve_theme_context` that every plotting
function in `/modules/dpg-visualizer.md` calls to obtain its colors. No hex literal should be
written inside a plot function; it should come out of the theme context.

`dpg/utils.py` is a small grab-bag of side-effecting helpers over Graphviz `Digraph` bodies and the
filesystem. It is *not* a general utility layer — it holds three functions.

# API

## themes.py — module constants

| Name | Type | Contents |
|---|---|---|
| `DPG_COLORS` | `dict[str, str]` | 30 named brand hexes: `gold`, `amber`, `orange`, `olive`, `brown`, `olive_light`, `sage`, `moss`, `fern`, `pine`, `steel`, `plum`, `sand`, `clay`, `ink`, `charcoal`, `mid_gray`, `light_gray`, `soft_gray`, `paper`, `node_fill`, `node_muted`, `grid`, `edge`, `range_fill`, `range_marker`, `success`, `danger`. |
| `DPG_CLASS_PALETTE` | `list[str]` | 14 hexes, gold-first ordering — default categorical palette. |
| `DPG_OLIVE_CLASS_PALETTE` | `list[str]` | 14 hexes, sand→pine gradient ordering — the `"olive"` palette. |
| `DPG_PREDICATE_LINE_PALETTE` | `list[str]` | 8 hexes used for per-feature predicate lines. |
| `LEGACY_COLORS` | `dict[str, str]` | Pre-rebrand blue/white set; the only mapping that defines `class_fill` natively (`#FFC000`). |
| `DPG_MPL_STYLE`, `LEGACY_MPL_STYLE` | `dict[str, Any]` | matplotlib rcParams overlays (facecolors, tick/label colors, grid alpha, `DejaVu Sans`, legend styling). |

## themes.py — functions

| Function | Signature | Purpose |
|---|---|---|
| `brand_sequential_cmap` | `() -> LinearSegmentedColormap` | `paper → #F2E7AD → gold → amber → orange`, named `dpg_brand_sequential`. |
| `edge_sequential_cmap` | `() -> LinearSegmentedColormap` | `light_gray → #B9AF98 → edge → olive`, named `dpg_edge_sequential`. |
| `community_cmap` | `() -> ListedColormap` | `DPG_CLASS_PALETTE` as a discrete map named `dpg_community`. |
| `class_cmap` | `(n_classes: int) -> ListedColormap` | Discrete map over `discrete_palette(n_classes)`. |
| `discrete_palette` | `(n_colors: int) -> list[str]` | Cyclic repeat of `DPG_CLASS_PALETTE`; `[]` for `n_colors <= 0`. |
| `feature_color_map` | `(features: list[str]) -> dict[str, Any]` | Dedupes preserving order, maps each feature to an RGBA tuple from `discrete_palette`. |
| `predicate_line_color_map` | `(features: list[str]) -> dict[str, Any]` | Same, over `DPG_PREDICATE_LINE_PALETTE`. |
| `_palette_values` | `(palette: str) -> list[str]` | Private. `default`/`brand`/`extended` → `DPG_CLASS_PALETTE`; `olive` → `DPG_OLIVE_CLASS_PALETTE`; anything else raises `DPGConfigurationError.unknown_palette`. |
| `_spaced_palette` | `(palette_values, n_colors) -> list[str]` | Private. Spreads `n_colors` picks evenly across the palette (with collision nudging) when `n_colors <= len(palette)`; falls back to cyclic repetition otherwise. `n_colors == 1` returns the middle color. |
| `resolve_theme_context` | `(theme: str = "dpg", palette: str = "default") -> dict[str, Any]` | The public entry point. |

## The theme context dict

`resolve_theme_context` always returns a dict with exactly these keys:

| Key | Type | Notes |
|---|---|---|
| `theme` | `str` | `"dpg"` or `"legacy"`. |
| `palette` | `str` | The lowercased palette name as requested. |
| `colors` | `dict[str, str]` | `DPG_COLORS` plus a synthesized `class_fill = DPG_COLORS["orange"]`, or `LEGACY_COLORS` verbatim. |
| `mpl_style` | `dict[str, Any]` | rcParams overlay; applied via `plt.rcParams.update(...)`. |
| `class_palette` | `list[str]` | Ordered hex list for class-colored marks. |
| `predicate_palette` | `list[str]` | 8 hexes; olive uses a `sand`-substituted variant, legacy uses a fixed `["#1F77B4", "#2CA02C", "#9467BD", "#17BECF"]`. |
| `sequential_cmap` | colormap | `brand_sequential_cmap()` / `Blues`. |
| `edge_cmap` | colormap | `edge_sequential_cmap()` / `Greys`. |
| `community_cmap` | colormap | `ListedColormap(class_palette)` / `tab20`. |
| `class_cmap` | `Callable[[int], colormap]` | **A callable, not a colormap** — call it with the class count. |
| `feature_color_map` | `Callable[[list[str]], dict[str, RGBA]]` | Callable. |
| `predicate_line_color_map` | `Callable[[list[str]], dict[str, RGBA]]` | Callable. |

## Resolution rules

1. `theme` and `palette` are coerced with `str(x or default).lower()`, so `None` and `""` fall back
   to `"dpg"` / `"default"`.
2. `theme == "legacy"` returns the legacy branch immediately. It builds its class palette from
   `_palette_values("default")` regardless of the requested palette — **the `palette` argument is
   recorded in the returned dict but otherwise ignored under `legacy`**, and an invalid palette name
   is never validated on that path.
3. Any other `theme` besides `"dpg"` raises `DPGConfigurationError.unknown_theme` ("Expected one of:
   dpg, legacy.").
4. Under `"dpg"`, `_palette_values(palette)` validates the palette and raises
   `DPGConfigurationError.unknown_palette` ("Expected one of: default, extended, olive.") for
   anything unrecognized. Note `brand` is accepted by the code but absent from that message.

## utils.py — shared helpers

| Function | Signature | Purpose |
|---|---|---|
| `highlight_class_node` | `(dot: Digraph, dpg_config: dict \| None = None) -> Digraph` | Rewrites every line of `dot.body` containing `label="Class` to carry `fillcolor`/`shape`/`style` from `dpg_config["dpg"]["visualization"]["class_node"]`. Returns the same (mutated) `Digraph`. |
| `change_node_color` | `(graph: Digraph, node_id: str, new_color: str) -> None` | Strips an existing `fillcolor=` from matching body lines and appends `<node_id> [fillcolor="…"]`. Raises `DPGValidationError.graph_node_not_found` if the id appears nowhere. |
| `delete_folder_contents` | `(folder_path: str) -> None` | Removes every file, symlink and subdirectory inside a folder (the folder itself survives). Raises `DPGValidationError.directory_required` if the path is not a directory. |

# Behavior

- `highlight_class_node` rejects non-`Digraph` input with
  `DPGValidationError.graphviz_type_required()`. When `dpg_config` is `None` it falls back to
  reading `config.yaml` **relative to the current working directory**; `FileNotFoundError` degrades
  silently to `{}`, malformed YAML raises `DPGConfigurationError.invalid_yaml`. Defaults when
  nothing is configured: `fillcolor="#a4c2f4"`, `shape=box`, `style="rounded, filled"`. Attribute
  rewriting is regex surgery on DOT source text, not a Graphviz API call.
- `delete_folder_contents` swallows per-item exceptions and prints `Failed to delete …` so one bad
  entry does not abort the sweep.

# Gotchas

- **`change_node_color` is defined twice.** `dpg/visualizer.py` defines its own
  `change_node_color(dot, node_id, fillcolor)` which calls `dot.node(...)` and additionally picks a
  contrasting `fontcolor` via perceptual brightness (`white` below 100). Every call site inside
  `visualizer.py` binds that local version — `utils.change_node_color` has no in-repo caller.
- **`highlight_class_node` has no caller either.** Only its definition appears in the codebase;
  class-node fills are applied through the visualizer path instead.
- Of the three utils, only `delete_folder_contents` is actually imported by `dpg/visualizer.py`, and
  its single use site there is commented out.
- **Three context values are callables.** `class_cmap`, `feature_color_map` and
  `predicate_line_color_map` must be invoked; passing them straight to matplotlib fails.
- `class_fill` only exists natively in `LEGACY_COLORS`. Under the `dpg` theme it is injected as
  `orange`; visualizer code defensively reads it as
  `colors.get("class_fill", colors["success"])`.
- The module-level `feature_color_map` / `predicate_line_color_map` functions use plain cyclic
  `discrete_palette` ordering, while the same-named context entries use `_spaced_palette` spreading.
  They produce different colors for the same input — prefer the context versions.
- `_spaced_palette(values, 1)` returns the *middle* palette entry, not the first, so single-class
  plots are not gold.

# Examples

```python
from dpg.themes import resolve_theme_context

ctx = resolve_theme_context()                        # theme="dpg", palette="default"
colors = ctx["colors"]
plt.rcParams.update(ctx["mpl_style"])
cmap = ctx["class_cmap"](3)                          # callable → ListedColormap
fmap = ctx["feature_color_map"](["petal length", "petal width"])

olive = resolve_theme_context(theme="dpg", palette="olive")
old   = resolve_theme_context(theme="legacy")        # palette arg is ignored here

resolve_theme_context(theme="solarized")             # DPGConfigurationError: unknown theme
resolve_theme_context(palette="neon")                # DPGConfigurationError: unknown palette
```

```python
from dpg.utils import delete_folder_contents, highlight_class_node

dot = highlight_class_node(
    dot,
    dpg_config={"dpg": {"visualization": {"class_node": {"fillcolor": "#FA6800"}}}},
)
delete_folder_contents("temp")                       # empties, does not remove, "temp"
```

Consumed by `/modules/dpg-visualizer.md`, which imports `resolve_theme_context` and
`delete_folder_contents` and threads `theme=` / `palette=` keyword arguments down from the
`DPGExplainer.plot*` wrappers.
