---
type: Python Module
title: dpg.visualizer
description: Every rendering function in DPG — Graphviz graph plots and Matplotlib analytics plots — plus the shared theme, save/show and label-formatting conventions.
resource: https://github.com/viniferraria/DPG/blob/main/dpg/visualizer.py
tags: [visualization, graphviz, matplotlib, plotting, theming, dpg]
generated: { by: claude_code/claude-opus-5, at: 2026-08-02T00:00:00Z }
status: stable
---

# Responsibility

`dpg/visualizer.py` (2722 lines) owns *all* rendering in the project. Nothing else in `dpg/` draws.
It splits into two families:

1. **Graphviz-backed graph renderers** — take an already-built `graphviz.Digraph` (`explanation.dot`) plus
   the node/edge metric DataFrames, recolor nodes and edges in place, pipe DOT to PNG, and embed the PNG in
   a Matplotlib figure via PIL.
2. **Matplotlib analytics plots** — take a `DPGExplanation` (or a derived table) and draw bar charts,
   scatter plots, heatmaps and range panels directly.

The module also exposes the *data-preparation* helpers those plots consume
(`lrc_predicate_scores`, `sample_bc_weights`, `class_feature_predicate_counts`,
`classwise_feature_bounds_from_communities`, `class_feature_predicate_positions`,
`dataset_feature_bounds_by_class`, `class_lookup_from_target_names`), the predicate parsers
(`parse_predicate_parts`, `parse_feature_from_predicate`), and the low-level DOT mutators
(`change_node_color`, `change_edge_color`, `normalize_data`).

# API

## Public plot functions (11)

| Name | Key parameters | Renders | Backend |
|---|---|---|---|
| `plot_dpg` | `plot_name, dot, df, df_edges, save_dir="results/", attribute, clusters, threshold_clusters, class_flag, layout_template, graph_style/node_style/edge_style, fig_size, dpi, pdf_dpi, show, export_pdf, theme, palette, label_mode="full", readability="normal", title` | The full DPG. Nodes colored uniformly, by one node-metric column (`attribute`, adds a horizontal colorbar), or by a `clusters` mapping. Edges colored/widened by `Weight`. Returns `None`. | graphviz → matplotlib |
| `plot_dpg_communities` | `plot_name, dot, df, dpg_metrics, save_dir, class_flag, df_edges, layout/style args, fig_size, dpi, pdf_dpi, show, export_pdf, theme, palette, label_mode="wrapped", readability="presentation", title` | The DPG with nodes colored by community index, read from `dpg_metrics["Communities"]` or `["Clusters"]`; unmatched nodes go light gray. Returns `None`. | graphviz → matplotlib |
| `plot_dpg_local_paths_aggregate` | `plot_name, dot, df, df_edges, paths_node_ids, path_confidences, sample_id, true_class_label, obtained_class_label, sample_metrics, save_dir, class_flag=True, layout/style args, show, export_pdf, theme, palette, label_mode="wrapped", readability="presentation", title` | The global DPG with one sample's traced paths highlighted: visited nodes/edges darkened and widened by summed confidence, everything else subdued; true/predicted class nodes get distinct fills; title carries sample id, pred/true and selected metrics. Returns the `Figure`. | graphviz → matplotlib |
| `plot_dpg_reg` | `plot_name, dot, df, df_dpg, save_dir="examples/", attribute, communities, leaf_flag, theme, palette` | Regression DPG (`"Pred …"` leaves). Optional coloring by `attribute` (with colorbar) or by community. Writes `{plot_name}_REG.png`. Returns `None`. | graphviz (`dot.render`) → matplotlib |
| `plot_dpg_constraints_overview` | `normalized_constraints, feature_names, class_colors_list, output_path, title, original_sample, original_class, target_class, theme, palette` | Horizontal per-feature bars showing each class's `{min, max}` constraint band, one-sided bounds as dashed rays, the original sample as a marker, and non-overlapping ("★") discriminative features highlighted. Returns the `Figure`, or `None` when there is nothing to draw. | matplotlib |
| `plot_lrc_vs_rf_importance` | `explanation, model, X_df, top_k=10, dataset_name, save_path, show, theme, palette` | Two side-by-side horizontal bar charts: top-k predicates by Local reaching centrality vs top-k `model.feature_importances_`, sharing one feature color legend. Returns the `Figure`. | matplotlib |
| `plot_lec_vs_rf_importance` | `*args, **kwargs` | Deprecated typo alias; emits `DeprecationWarning` and delegates to `plot_lrc_vs_rf_importance`. | matplotlib |
| `plot_top_lrc_predicate_splits` | `explanation, X_df, y, top_predicates=5, top_features=2, dataset_name, class_names, save_path, show, theme, palette` | Scatter of the two highest-LRC features colored by class, overlaid with `axvline`/`axhline` split lines for the top-LRC predicates (`<=` dashed, `>` solid). Returns the `Figure`, or `None` if fewer than two features resolve against `X_df.columns`. | matplotlib |
| `plot_sample_using_bc_weights` | `explanation, X_df, y, top_k=10, dataset_name, class_names, save_path, show, theme, palette` | 2-component PCA scatter of `X_df`, colored by class, marker size scaled by `sample_bc_weights` (sum of betweenness centrality of satisfied top-k predicates). Returns the `Figure`. | matplotlib (+ sklearn PCA) |
| `plot_class_feature_complexity` | `heat_df, dataset_name, class_names, top_n_features=10, save_prefix, show, theme, palette` | Two figures: a class×feature predicate-count heatmap with a class color strip and annotated cells, and a grouped bar chart of the same counts. Returns `(fig_heat, fig_bar)`; with `save_prefix` writes `*_heatmap.png` and `*_bars.png`. | matplotlib |
| `plot_dpg_class_bounds_vs_dataset_feature_ranges` | `explanation, X_df, y, dataset_name, top_features=4, feature_cols_per_row=4, class_lookup, predicate_positions, class_bounds, class_filter, density_tol_ratio, predicate_alpha, dataset_range_lw, save_path, show, theme, palette` | Grid of per-(class, feature) panels comparing the empirical dataset range against the DPG community-derived bound, with `<` / `>` arrow markers for infinite bounds and triangle markers for clustered predicate-threshold density. Returns the `Figure`, or `None` when no class/feature survives filtering. | matplotlib |

## Public non-plot helpers

| Name | Returns |
|---|---|
| `parse_predicate_parts(label)` | `(feature, operator, threshold)` or `None` |
| `parse_feature_from_predicate(label)` | feature name, else the original label |
| `lrc_predicate_scores(explanation, top_k)` | DataFrame `[predicate, feature, operator, threshold, lrc]` |
| `sample_bc_weights(explanation, X_df, top_k)` | Series of per-sample bottleneck exposure |
| `class_feature_predicate_counts(explanation)` | class × feature count DataFrame (input to `plot_class_feature_complexity`) |
| `classwise_feature_bounds_from_communities(explanation)` | `[class_name, community_id, feature, lower_bound, upper_bound, range_width]` |
| `class_feature_predicate_positions(explanation)` | `[class_name, community_id, feature, operator, threshold]` |
| `dataset_feature_bounds_by_class(X_df, y, class_names, class_lookup)` | `[class_name, feature, ds_lower_bound, ds_upper_bound]` |
| `class_lookup_from_target_names(target_names)` | `{class name: index}` |
| `change_node_color(dot, node_id, fillcolor)` | `None`; sets fill plus a brightness-derived font color |
| `change_edge_color(graph, source_id, target_id, new_color, new_width)` | `None`; string-edits the first matching `dot.body` line |
| `normalize_data(df, attribute, colormap)` | `{node: hex color}` |

# Behavior

**Theming.** Every public function starts with `resolve_theme_context(theme=..., palette=...)` from
[`/modules/dpg-themes-utils.md`](/modules/dpg-themes-utils.md) (`theme="dpg"` default, `"legacy"` supported)
and then calls `_apply_matplotlib_theme`, which writes the theme's style dict into the **global**
`plt.rcParams`. Graph renderers additionally apply `_apply_dpg_graphviz_skin`, a `_LAYOUT_TEMPLATES` preset
(`default` / `compact` / `vertical` / `wide`), caller `graph_style` / `node_style` / `edge_style` overrides,
and a `_READABILITY_PRESETS` entry (`compact` / `normal` / `presentation`) that also supplies the label wrap
width. Unknown `readability` or `label_mode` raise `DPGValidationError`.

**Graphviz dependency.** The system `dot` binary must be on `PATH`. `plot_dpg`, `plot_dpg_communities` and
`plot_dpg_local_paths_aggregate` render through `_pipe_graph_png_with_fallback`, which pipes
`dot.source` in memory; `ExecutableNotFound` is converted to `DPGVisualizationError.graphviz_not_found()`,
and any *other* rendering failure is retried once with `_sanitize_dot_source` before the original exception
is re-raised. `plot_dpg_reg` is the exception: it calls `dot.render()` and writes
`{save_dir}/{plot_name}_temp.gv` and `.gv.png` to disk. Tests guard graph plots with
`shutil.which("dot") is None → pytest.skip`.

**Save vs. show.** The two families behave differently and this trips people up:

- Graph renderers (`plot_dpg`, `plot_dpg_communities`, `plot_dpg_local_paths_aggregate`) **always** write
  `{save_dir}/{plot_name}.png` (`os.makedirs(..., exist_ok=True)`), optionally a `.pdf` when
  `export_pdf=True`, and never call `plt.show()`. `show=False` only means "close the figure".
- Analytics plots save **only** when `save_path` (or `save_prefix`) is given, and `show=True` calls
  `plt.show()` while `show=False` calls `plt.close(fig)`.
- `plot_dpg_constraints_overview` uses `output_path` and always returns the figure without closing it.

**Filename mutation.** `plot_dpg` appends `_{attribute}` or `_clusters_{threshold_clusters}` to
`plot_name`; `plot_dpg_communities` appends `_communities`; `plot_dpg_reg` appends `_{attribute}` or
`_communities` and then suffixes `_REG.png`. The on-disk name is therefore not always what you passed in.

**Explainer wrappers.** The `DPGExplainer.plot*` methods in
[`/modules/dpg-explainer.md`](/modules/dpg-explainer.md) are thin pass-throughs that only add "compute the
explanation if you didn't give me one" and unpack `explanation.dot` / `.node_metrics` / `.edge_metrics`:

| Method | Delegates to |
|---|---|
| `plot` | `plot_dpg` |
| `plot_communities` | `plot_dpg_communities` |
| `plot_local_on_dpg` | `plot_dpg_local_paths_aggregate` |
| `plot_lrc_importance` | `plot_lrc_vs_rf_importance` |
| `plot_top_lrc_splits` | `plot_top_lrc_predicate_splits` |
| `plot_class_feature_complexity` | `class_feature_predicate_counts` → `plot_class_feature_complexity` |
| `plot_sample_using_bc_weights` | `plot_sample_using_bc_weights` |
| `plot_class_bounds_vs_dataset_ranges` | `class_lookup_from_target_names` → `plot_dpg_class_bounds_vs_dataset_feature_ranges` |

# Gotchas

- **The DOT object is mutated in place.** `change_node_color` and `change_edge_color` append/rewrite lines in
  `dot.body`, so `plot_dpg` and `plot_dpg_communities` permanently restyle the `Digraph` you hand them.
  Reusing `explanation.dot` across several plots accumulates styling; `examples/render_plot_gallery.py`
  deep-copies a `base_dot` for each renderer. Only `plot_dpg_local_paths_aggregate` deep-copies internally.
- **Class-node detection is substring matching on the label text.** Most code paths use
  `'Class' in row['Label']` or `str.contains('Class')`; the local-paths renderer uses
  `label.startswith("Class ")`. A *feature* whose name contains "Class" would be misclassified. See
  [`/conventions/label-contract.md`](/conventions/label-contract.md).
- **Predicate parsing is regex-driven.** `_PREDICATE_PATTERN` (`(.+?)\s*(<=|>)\s*<number>`) backs
  `parse_predicate_parts`, `lrc_predicate_scores`, `sample_bc_weights`, `_predicate_node_lookup` and the
  short/wrapped label modes. Predicate *selection* in `lrc_predicate_scores` and `sample_bc_weights` is even
  looser — a `str.contains("<=") | str.contains(">")` mask over `Label`. Any change to the label format in
  `core.generate_dot` silently changes what these plots pick up.
- **`_sanitize_dot_source` is a text pass over the DOT source**, escaping quotes/brackets and optionally
  reformatting labels. Its second substitution, `re.sub(r'label=([^\\s\\]]+)', r'label="\\1"', source)`,
  is written with escaped backslashes in both pattern and replacement, so it does not re-insert the captured
  group — treat the first substitution as the one that does the real work.
- **`change_edge_color` matches raw text** (`f'{source_id} -> {target_id}' in line`) and stops at the first
  hit, appending attributes by replacing the final `]`. It silently no-ops if the edge line is absent or
  formatted differently.
- **Global state side effects.** The module sets `Image.MAX_IMAGE_PIXELS = 500000000` at import time, and
  `_apply_matplotlib_theme` writes process-wide `plt.rcParams` on every call.
- **Matplotlib backend.** Because analytics plots call `plt.show()` by default, headless callers must force
  a non-interactive backend: `tests/test_visualizations_api.py` does
  `os.environ.setdefault("MPLBACKEND", "Agg")` and `examples/render_plot_gallery.py` does
  `matplotlib.use("Agg")` before importing `dpg`.
- **Soft failures.** `plot_dpg_constraints_overview` prints a `WARNING:` and returns `None` rather than
  raising when constraints are empty or non-finite; `plot_top_lrc_predicate_splits` and
  `plot_dpg_class_bounds_vs_dataset_feature_ranges` return `None` when selection yields nothing; `_class_mask`
  prints when a class name cannot be resolved. Check for `None` before touching the result.
- **Hard failures** come from `DPGValidationError`: `plot_dpg` with both `attribute` and `clusters`,
  `plot_dpg_communities` with missing metrics/communities or no label overlap, `plot_lrc_vs_rf_importance`
  with no predicate labels or a model lacking `feature_importances_`, `sample_bc_weights` with a non-DataFrame
  `X_df`, and `plot_class_feature_complexity` with an empty `heat_df`.
- **`plot_dpg_class_bounds_vs_dataset_feature_ranges` clamps the left axis limit with `max(0.0, …)`**, so
  features with negative values are cropped at zero.
- **Export surface is uneven.** `dpg/__init__.py` re-exports every plot function *except*
  `plot_dpg_communities`, which must be imported as `from dpg.visualizer import plot_dpg_communities`.

# Examples

```python
import matplotlib
matplotlib.use("Agg")  # required before plt.show() paths run headless

from dpg import DPGExplainer, plot_dpg
from dpg.visualizer import class_feature_predicate_counts, plot_class_feature_complexity

explainer = DPGExplainer(model, feature_names=X.columns.tolist(), target_names=target_names)
explanation = explainer.explain_global(X.values, communities=True)

# Direct call: always writes results/iris_dpg.png, show=False just closes the figure.
plot_dpg(
    "iris_dpg",
    explanation.dot,
    explanation.node_metrics,
    explanation.edge_metrics,
    save_dir="results/",
    class_flag=True,
    label_mode="wrapped",
    readability="presentation",
    show=False,
)

# Analytics plot: file written only because save_path was given.
fig = explainer.plot_lrc_importance(X_df=X, explanation=explanation,
                                    top_k=8, save_path="lrc_vs_rf.png", show=False)

# Two-step: derive the table, then render both figures.
heat = class_feature_predicate_counts(explanation)
fig_heat, fig_bars = plot_class_feature_complexity(
    heat_df=heat, class_names=target_names, top_n_features=3,
    save_prefix="community_complexity", show=False,
)  # writes community_complexity_heatmap.png and community_complexity_bars.png
```
