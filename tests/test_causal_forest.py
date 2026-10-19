"""Tests for CausalForest support in DecisionPredicateGraph and DPGExplainer."""

import re

import numpy as np
import pytest

econml = pytest.importorskip("econml")

from econml.grf import CausalForest

from dpg import DPGExplainer
from dpg.core import DEFAULT_DPG_CONFIG, DecisionPredicateGraph, is_regression_model
from dpg.exceptions import DPGModelError

SEED = 160898


@pytest.fixture(scope="module")
def causal_data():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 4))
    T = rng.integers(0, 2, size=300)
    y = X[:, 0] * T + rng.normal(scale=0.1, size=300)
    feature_names = ["f0", "f1", "f2", "f3"]
    return X, T, y, feature_names


@pytest.fixture(scope="module")
def causal_forest(causal_data):
    X, T, y, _ = causal_data
    cf = CausalForest(n_estimators=5, max_depth=3, min_samples_leaf=10, inference=False, random_state=42)
    cf.fit(X, T, y)
    return cf


def test_causal_forest_is_regression_model(causal_forest):
    assert is_regression_model(causal_forest)


def test_global_graph_has_pred_leaves(causal_data, causal_forest):
    X, _, _, feature_names = causal_data
    dpg = DecisionPredicateGraph(
        causal_forest, feature_names, dpg_config=DEFAULT_DPG_CONFIG
    )
    dot = dpg.fit(X)
    graph, nodes = dpg.to_networkx(dot)

    label_re = re.compile(r"^(f[0-3] (<=|>) -?\d+(\.\d+)?|Pred -?\d+(\.\d+)?)$")
    labels = [label for node_id, label in nodes if "->" not in node_id]
    for label in labels:
        assert label_re.match(label), f"Unexpected label format: {label!r}"

    assert any(label.startswith("Pred ") for label in labels)
    assert not any(label.startswith("Class ") for label in labels)
    assert graph.number_of_edges() > 0


def test_explainer_global_and_local(causal_data, causal_forest):
    X, _, _, feature_names = causal_data
    explainer = DPGExplainer(causal_forest, feature_names, dpg_config=DEFAULT_DPG_CONFIG)

    explanation = explainer.explain_global(X)
    assert len(explanation.nodes) > 0

    local = explainer.explain_local(X[0])
    assert len(local.tree_paths) == 5
    for path in local.tree_paths:
        assert path.labels[-1].startswith("Pred ")


def test_leaf_value_matches_tree(causal_data, causal_forest):
    X, _, _, feature_names = causal_data
    explainer = DPGExplainer(causal_forest, feature_names, dpg_config=DEFAULT_DPG_CONFIG)

    tree_ = causal_forest.estimators_[0].tree_
    i = 0
    while True:
        left = tree_.children_left[i]
        right = tree_.children_right[i]
        if left == right:
            break
        i = left if X[0][tree_.feature[i]] <= tree_.threshold[i] else right
    expected = f"Pred {round(float(tree_.value[i][0][0]), 2)}"

    local = explainer.explain_local(X[0], X=X)
    assert local.tree_paths[0].labels[-1] == expected


def test_multi_treatment_raises(causal_data):
    X, _, y, feature_names = causal_data
    rng = np.random.default_rng(SEED)
    T2 = rng.integers(0, 2, size=(300, 2))
    cf2 = CausalForest(n_estimators=5, max_depth=3, min_samples_leaf=10, inference=False, random_state=42)
    cf2.fit(X, T2, y)

    assert cf2.n_relevant_outputs_ == 2
    with pytest.raises(DPGModelError, match="n_relevant_outputs_=2"):
        DecisionPredicateGraph(cf2, feature_names, dpg_config=DEFAULT_DPG_CONFIG)


def test_multi_outcome_raises(causal_data):
    # econml 0.17's CausalForest derives `n_relevant_outputs_` solely from
    # `T.shape[1]` (see `CausalForest._get_n_outputs_decomposition`), and its
    # `LinearMomentGRFCriterion` unconditionally rejects any `y` with more
    # than one column ("LinearMomentGRFCriterion currently only supports a
    # scalar y"), regardless of `T`'s shape or other hyperparameters. So a
    # multi-column `y` never reaches `n_relevant_outputs_ != 1`: `fit()`
    # itself raises first. This confirms DecisionPredicateGraph can only ever
    # observe `n_relevant_outputs_ != 1` via a multi-column `T`
    # (see test_multi_treatment_raises), not via a multi-column `y`.
    X, T, y, _ = causal_data
    y2 = np.column_stack([y, -y])
    cf2 = CausalForest(n_estimators=5, max_depth=3, min_samples_leaf=10, inference=False, random_state=42)
    with pytest.raises(AttributeError, match="scalar y"):
        cf2.fit(X, T, y2)


def test_dpg_leaf_mean_equals_tree_average_not_predict(causal_data, causal_forest):
    # DPG reads `tree_.value` per tree, so averaging the Pred leaves of one sample
    # reproduces `predict_tree_average`, NOT `predict`, which solves the moment
    # equation on the *averaged* alpha/J (pinv(mean J) @ mean alpha).
    X, _, _, feature_names = causal_data
    explainer = DPGExplainer(causal_forest, feature_names, dpg_config=DEFAULT_DPG_CONFIG)
    sample = X[:20]
    dpg_mean = np.array([
        np.mean([float(p.labels[-1].split()[1]) for p in explainer.explain_local(x, X=X).tree_paths])
        for x in sample
    ])

    tree_avg = causal_forest.predict_tree_average(sample).ravel()
    predict = causal_forest.predict(sample).ravel()
    assert np.allclose(dpg_mean, tree_avg, atol=0.006)  # Pred labels are rounded to 2 decimals
    assert np.abs(dpg_mean - predict).max() > 0.006


def test_causal_forest_dml_rejected_but_inner_forest_accepted():
    CausalForestDML = pytest.importorskip("econml.dml").CausalForestDML
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 3))
    T = rng.integers(0, 2, size=300)
    y = X[:, 0] * T + rng.normal(scale=0.1, size=300)
    names = ["f0", "f1", "f2"]
    est = CausalForestDML(
        n_estimators=4, max_depth=3, min_samples_leaf=10, discrete_treatment=True, random_state=0
    ).fit(y, T, X=X)

    with pytest.raises(DPGModelError):
        DecisionPredicateGraph(est, names, dpg_config=DEFAULT_DPG_CONFIG)

    inner = est.model_cate.estimators_[0]
    assert np.allclose(est.effect(X).ravel(), inner.predict(X).ravel())
    dpg = DecisionPredicateGraph(inner, names, dpg_config=DEFAULT_DPG_CONFIG)
    dpg.fit(X)
    _, nodes = dpg.to_networkx()
    assert any(label.startswith("Pred ") for _, label in nodes)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_dml_inner_forest_ranks_effect_modifier_above_prognostic(seed):
    # X0 modifies the effect (tau = 1 + 2*1[X0>0]); X1 is prognostic only (3*X1 in Y).
    # On the DML inner forest DPG's summed per-feature LRC and in-degree both rank X0 first.
    # (Default CausalForest criterion="mse" ranks X1 first: see causal-forest-support.md.)
    CausalForestDML = pytest.importorskip("econml.dml").CausalForestDML
    rng = np.random.default_rng(seed)
    n = 600
    X = rng.normal(size=(n, 5))
    T = rng.integers(0, 2, size=n)
    y = 3 * X[:, 1] + T * (1 + 2 * (X[:, 0] > 0)) + rng.normal(scale=0.5, size=n)
    names = [f"x{i}" for i in range(5)]
    est = CausalForestDML(
        n_estimators=20, max_depth=4, min_samples_leaf=20, discrete_treatment=True, random_state=seed
    ).fit(y, T, X=X)

    nm = DPGExplainer(est.model_cate.estimators_[0], names, dpg_config=DEFAULT_DPG_CONFIG).explain_global(X).node_metrics
    nm = nm[~nm["Label"].str.startswith(("Pred ", "Class "))].copy()
    nm["feature"] = nm["Label"].str.split(" ").str[0]
    per_feature = nm.groupby("feature")[["Local reaching centrality", "In degree nodes"]].sum()

    assert per_feature.loc["x0", "Local reaching centrality"] > per_feature.loc["x1", "Local reaching centrality"]
    assert per_feature.loc["x0", "In degree nodes"] > per_feature.loc["x1", "In degree nodes"]
