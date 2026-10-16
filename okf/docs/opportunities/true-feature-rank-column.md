---
type: Opportunity
title: True-feature rank column
description: Report the best and mean rank of the true features alongside precision and recall, so that near misses can be told apart when there is one true feature; proposed, untested, not in any runner.
resource: https://github.com/viniferraria/DPG/blob/main/experiments/causal_synthetic_scenarios/run_dpg_causal_synthetic.py
tags: [opportunity, reporting, rank, precision, recall]
generated: { by: claude_code/claude-sonnet-5-5, at: 2026-10-08T00:00:00Z }
status: proposed
---

# The idea

Add columns for the best and the mean rank of the true features (for example `best_true_rank` and
`mean_true_rank`) to the causal-accuracy CSV, next to `precision` and `recall`.

# Why

The reference reports precision and recall on the top-`TOP_K` set. With one true feature and `TOP_K = 3`,
precision cannot exceed 1/3, and the pair (precision, recall) cannot tell a true feature in 1st place from
one in 3rd, or a 4th place from a miss by a wide margin. A rank would show how close a miss was.

# Status

Proposed, untested, not in any runner. An earlier ACIC runner wrote such columns; they were removed so
that its output matches the reference columns (see [ACIC 2019](/experiments/acic2019.md)). Open questions
for the test: how to rank (per node label, or per feature, which is the question of
[per-feature score aggregation](per-feature-score-aggregation.md)), how to treat ties and features the
graph never splits on, and whether it changes any conclusion drawn from precision and recall.
