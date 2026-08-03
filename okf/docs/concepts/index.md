# Concepts

Domain objects and measures that outlive any single module.

* [Explanation dataclasses](explanation-dataclasses.md) - Field-level reference for DPGExplanation, DPGLocalExplanation and DPGTreePathExplanation, the three result containers returned by DPGExplainer.
* [Faithfulness evaluation](faithfulness-evaluation.md) - How DPGExplainer.evaluate_faithfulness scores local DPG explanations against the fitted ensemble, and how the sample-confidence diagnostics that feed it are computed.

## See also

* [dpg.explainer](/modules/dpg-explainer.md) - The API that produces both.
* [Event label contract](/conventions/label-contract.md) - Why some fields hold `"Class 0"` and others hold `"0"`.
