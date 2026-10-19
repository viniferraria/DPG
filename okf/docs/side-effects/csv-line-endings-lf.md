---
type: Side Effect
title: Experiment CSVs are now written with LF line endings
description: append_csv writes through pandas, which ends rows with LF. The old csv.DictWriter writers produced CRLF for every causal_accuracy file and for the causal-synthetic node_metrics file, so new files are not byte-identical to old ones even when the content is equal.
resource: https://github.com/viniferraria/DPG/blob/main/experiments/causal_accuracy.py
tags: [experiments, csv, line-endings, crlf, byte-identical, diff]
generated: { by: claude_code/claude-sonnet-5-5, at: 2026-10-17T00:00:00Z }
status: stable
---

# What happens

`append_csv` in the [shared module](/modules/experiments-causal-accuracy.md) calls `DataFrame.to_csv`, whose
default line terminator on this platform is `\n`. The writers it replaced used `csv.DictWriter`, whose
default `lineterminator` is `\r\n`. This was not separately requested; it came with the move to pandas and
is kept and documented.

Which old files were CRLF (checked by counting lines containing `\r` in the first 3000 bytes):

| Old file | CR lines |
|---|---|
| `causal_synthetic_scenarios/results/node_metrics_2026-10-15T05-08-31.csv` | 16 |
| `causal_synthetic_scenarios/results/causal_accuracy_2026-10-15T05-08-31.csv` | 36 |
| `monks/results/causal_accuracy_2026-10-15T05-17-45.csv` | 25 |
| `acic2019/results/causal_accuracy_2026-10-15T05-21-47.csv` | 38 |
| `compas/results/causal_accuracy_2026-10-15T05-18-09.csv` | 27 |
| `monks/results/node_metrics_2026-08-03T03-39-15.csv` | 0 |

The MONK's, ACIC and COMPAS `node_metrics` files were already written by pandas, so they were LF before
and are unchanged in this respect (the 2026-10-15 ones show `\n` in the last header bytes). The change is
visible for every `causal_accuracy` file and for the causal-synthetic `node_metrics` file.

# Example (executed)

The same two rows written both ways:

```python
r = causal_accuracy_rows(df, ["F1"], "demo", 0).head(2)
append_csv(r, new_path)                                        # pandas
with open(old_path, "w", newline="") as f:                     # old writer
    w = csv.DictWriter(f, fieldnames=list(r.columns)); w.writeheader(); w.writerows(r.to_dict("records"))

pd.read_csv(new_path).equals(pd.read_csv(old_path))            # True
new_path.read_bytes() == old_path.read_bytes()                 # False
new_path.read_bytes() == old_path.read_bytes().replace(b"\r\n", b"\n")   # True
```

Content equal, bytes differ only by `\r`.

# Effect

- **Tools that parse CSV are unaffected.** pandas, DuckDB `read_csv`, and `csv.reader` accept both. The
  DuckDB rebuild in [the workflow](/workflows/reconstruct-causal-accuracy.md) reads either.
- **Byte comparison and text diff are affected.** `cmp`, `sha256sum` and `git diff` of a re-run against
  a committed old file report every line changed. Normalise first: `tr -d '\r' < old.csv | cmp - new.csv`.
- **Mixed-ending append.** `append_csv` opens the file in append mode. Appending to an old CRLF file
  would produce a file whose first rows are CRLF and later rows LF. Runners always create a new
  timestamped file per run, so this only happens if a path is reused by hand.
- **A raw-text splitter breaks.** `line.split(",")` on the last field of an old file leaves a trailing
  `\r` in `processing_time` or `recall`; `float("0.5\r")` still parses, but string equality does not.

# Why not restore CRLF

No consumer in the repo depends on CRLF, and LF is what pandas wrote for the node-metrics files of three
suites already. Restoring it would need `lineterminator="\r\n"` in `append_csv`; that is a one-keyword
change if byte parity with old files is ever wanted.

# Related

- [node_metrics schema change](node-metrics-schema-unification.md)
- [Shared scoring module](/modules/experiments-causal-accuracy.md)
