---
type: Side Effect
title: node_metrics CSV schema changed for MONK, ACIC and COMPAS
description: Since the suites moved onto experiments/causal_accuracy.py, the MONK's, ACIC 2019 and COMPAS node_metrics CSVs use the reference header (node_idx, in_degree, out_degree, split; experiment first). Older result files keep their old names and column order, so reading old and new together needs a rename; a plain union silently yields NULL columns.
resource: https://github.com/viniferraria/DPG/blob/main/experiments/causal_accuracy.py
tags: [experiments, csv, schema, node-metrics, migration, duckdb, pandas]
generated: { by: claude_code/claude-sonnet-5-5, at: 2026-10-17T00:00:00Z }
status: stable
---

# What happens

`run_monk.py`, `run_acic2019.py` and `run_compas.py` used to write `node_metrics_<ts>.csv` through a local
helper that kept the frame's own column names (snake-cased by a local `to_snake_case`) and put the run
identifiers at the end. They now call `node_metrics_rows` from the
[shared module](/modules/experiments-causal-accuracy.md), which writes the **reference runner's header**.
The reference's own header did not change. The `causal_accuracy_<ts>.csv` header was already the same in
every suite and is unchanged.

Files already in `results/` are not rewritten, so the tree now holds two layouts per suite.

# Headers, old and new

Old MONK's/ACIC (2026-10-15 files; checked by reading the first line):

```
node_index,node,degree,in_degree_nodes,out_degree_nodes,betweenness_centrality,local_reaching_centrality,closeness_centrality,harmonic_centrality,collective_influence,local_clustering_coefficient,percolation_centrality,label,experiment,split_idx,processing_time
```

Old COMPAS adds `seed` between `split_idx` and `processing_time`. MONK's files from 2026-08 are older still:
no `collective_influence`, `local_clustering_coefficient` or `percolation_centrality`.

New (all four suites):

```
experiment,split,node,degree,in_degree,out_degree,betweenness_centrality,local_reaching_centrality,closeness_centrality,harmonic_centrality,collective_influence,local_clustering_coefficient,percolation_centrality,node_idx,label,processing_time[,seed]
```

`seed` exists only in COMPAS and is the last column.

## Rename mapping (old MONK's/ACIC/COMPAS to new)

| Old column | New column | Position change |
|---|---|---|
| `node_index` | `node_idx` | from first to after the metric columns |
| `in_degree_nodes` | `in_degree` | |
| `out_degree_nodes` | `out_degree` | |
| `split_idx` | `split` | from after `experiment` at the end to second |
| `experiment` | `experiment` | from after `label` to first |
| `processing_time` | `processing_time` | last in MONK's/ACIC (unchanged); in COMPAS it was last and is now followed by `seed` |
| `seed` (COMPAS) | `seed` | was before `processing_time`, now last |
| metric columns | same names | same frame order |

Causal-synthetic files from before 2026-10 have 10 or 11 columns (for example `node_metrics_4.csv`:
`experiment,split,node,degree,in_degree,out_degree,betweenness_centrality,local_reaching_centrality,node_idx,label,processing_time`).
They already use the new names but lack the later metric columns; the 2026-10-15 reference file is
identical to the new header.

# Effect

- **Positional readers break.** Anything that reads by column index (`cut -f`, `df.iloc[:, 0]`) now gets a
  different column for MONK/ACIC/COMPAS.
- **By-name readers break on the four renamed columns.** `df["node_index"]` is a `KeyError` on a new file.
- **A naive union hides the break.** DuckDB's `union_by_name` and `pd.concat` align by name, so mixing an old
  and a new file does not fail; the renamed columns come out as two half-NULL columns each. Executed on
  the old MONK's file `node_metrics_2026-10-15T05-17-45.csv` plus the reference file
  `node_metrics_2026-10-15T05-08-31.csv`:

  ```sql
  SELECT count(*) FROM read_csv(['experiments/monks/results/node_metrics_2026-10-15T05-17-45.csv',
                                 'experiments/causal_synthetic_scenarios/results/node_metrics_2026-10-15T05-08-31.csv'],
                                union_by_name = true);
  -- 79216 rows, no error: node_index is NULL on the 70255 reference rows, node_idx on the 8961 MONK's rows
  ```

- The DuckDB rebuild SQL reads `split` and `node_idx`, so it works on new files of any suite and on
  causal-synthetic files, but **not** on old MONK/ACIC/COMPAS files without the rename below.

# Reading old and new files together

Rename on read, then union. DuckDB (executed on the same two files; first rows came back as
`monks-1_RandomForest,0,0,28,a1_1 > 0.5`):

```sql
SELECT experiment, split, node_idx, in_degree, label FROM (
  SELECT experiment, split_idx AS split, node_index AS node_idx,
         in_degree_nodes AS in_degree, out_degree_nodes AS out_degree, label
  FROM read_csv('experiments/monks/results/node_metrics_2026-10-15T05-17-45.csv')   -- old layout
  UNION ALL BY NAME
  SELECT experiment, split, node_idx, in_degree, out_degree, label
  FROM read_csv('experiments/causal_synthetic_scenarios/results/node_metrics_2026-10-15T05-08-31.csv')  -- new layout
);
```

pandas equivalent for an old file:

```python
OLD_TO_NEW = {"node_index": "node_idx", "in_degree_nodes": "in_degree",
              "out_degree_nodes": "out_degree", "split_idx": "split"}
df = pd.read_csv(path).rename(columns=OLD_TO_NEW)
```

Column order differs between the layouts; select by name, never by position.

# Why it is not fixed by rewriting old files

Old results are artifacts (see the experiment pages: `results/` is output, not source), and rewriting them
would erase the evidence of what the previous runner produced. The new header is the reference's, so there
is now one layout going forward.

# Related

- [Shared scoring module](/modules/experiments-causal-accuracy.md)
- [CSV line endings](csv-line-endings-lf.md)
- [Rebuilding causal_accuracy with DuckDB](/workflows/reconstruct-causal-accuracy.md)
