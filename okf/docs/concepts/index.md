# Concepts

Domain objects and measures that outlive any single module.

* [Explanation dataclasses](explanation-dataclasses.md) - Field-level reference for DPGExplanation, DPGLocalExplanation and DPGTreePathExplanation, the three result containers returned by DPGExplainer.
* [Faithfulness evaluation](faithfulness-evaluation.md) - How DPGExplainer.evaluate_faithfulness scores local DPG explanations against the fitted ensemble, and how the sample-confidence diagnostics that feed it are computed.
* [Faithfulness evaluation on regressors](faithfulness-regressors.md) - Why the "Class "-gated voting and evidence logic in DPGExplainer never fires for regressor "Pred " leaves, making majority_vote always None and output_fidelity always 0.0 for a pure regressor.
* [Causal forest support](causal-forest-support.md) - How econml.grf.CausalForest is traced into a Decision Predicate Graph, why only n_relevant_outputs_ == 1 is supported, and worked examples of the supported and rejected cases.

## See also

* [dpg.explainer](/modules/dpg-explainer.md) - The API that produces both.
* [Event label contract](/conventions/label-contract.md) - Why some fields hold `"Class 0"` and others hold `"0"`.
