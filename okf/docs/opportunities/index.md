# Opportunities

An opportunity is an untested or partly tested idea for scoring or reporting DPG experiments. It is
recorded here instead of in a runner. Each page states the idea, the evidence so far with its source,
and the open test. The test runs separately (scratchpad or a standalone analysis script) against the
reference protocol in `experiments/causal_synthetic_scenarios/run_dpg_causal_synthetic.py`, on the same
runs, and its result is reported. An opportunity is adopted into a runner only after the user approves.

Until then, every suite scores as the reference does: rank predicate nodes by each metric in `METRICS`,
take the top `TOP_K` node labels, map each to its feature name, then compute precision and recall.

## Proposed

* [per-feature-score-aggregation](per-feature-score-aggregation.md) - Combine a feature's node scores by sum, mean or max instead of taking the reference's top-k node labels. Partly tested; sum looks biased toward features with many nodes on one ACIC dataset.
* [true-feature-rank-column](true-feature-rank-column.md) - Report the best and mean rank of the true features next to precision and recall. Proposed, untested, not in any runner.
* [class-separation-gain](class-separation-gain.md) - C1: traffic-weighted information gain of each split on the leaf classes, from the trace log, summed per feature. Tested once on the four original scenarios; tied in-degree at the ceiling.
* [intervention-contrast](intervention-contrast.md) - C2: difference in class-absorption probabilities between the two sides of a split, from an absorbing Markov chain over the DPG, summed per feature. Tested once; slightly below in-degree.
* [combined-centrality-score](combined-centrality-score.md) - Combine LRC, betweenness and out-degree into one node score. Correlation check on 2026-10-15: metrics form three near-redundant groups (reach, flow, local); precision and recall untested.
* [role-guided-node-ablation](role-guided-node-ablation.md) - Label nodes by metric profile (hub, choke point, near-root), then score by role-percentile or by reach-to-sink lost when the node is removed. Tested once (2026-10-15): ablation scores lead on causal synthetic (P 0.42 vs 0.29 best existing), driven by ExtraTrees; role-percentile adds nothing.
* [feature-drop-and-relearn](feature-drop-and-relearn.md) - Retrain without a feature, rebuild the DPG, score the change (Bénard & Josse, arXiv 2308.03369). Untested.

## See also

* [Experiment suites](/experiments/index.md) - The runners that follow the reference protocol.
* [Side effects and limitations](/side-effects/index.md) - Verified behaviors, as opposed to ideas.
