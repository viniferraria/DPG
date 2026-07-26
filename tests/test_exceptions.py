"""Regression tests for the DPG exception hierarchy and messages."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from dpg import (
    DPGConfigurationError,
    DPGDatasetError,
    DPGError,
    DPGGraphError,
    DPGModelError,
    DPGNotFittedError,
    DPGValidationError,
)
from dpg.core import DecisionPredicateGraph
from dpg.explainer import DPGExplainer
from dpg.sklearn_dpg import select_dataset
from dpg.sklearn_dpg import test_dpg as run_dpg


def _iris_model() -> tuple[RandomForestClassifier, np.ndarray]:
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=2, random_state=42)
    model.fit(iris.data, iris.target)
    return model, iris.data


def test_validation_errors_are_dpg_and_value_errors():
    model, _ = _iris_model()

    with pytest.raises(DPGValidationError, match="Feature names cannot be empty"):
        DecisionPredicateGraph(model=model, feature_names=[])

    with pytest.raises(ValueError):
        DecisionPredicateGraph(model=model, feature_names=[])


def test_model_error_explains_invalid_model():
    with pytest.raises(DPGModelError, match="tree-based ensemble"):
        DecisionPredicateGraph(model=object(), feature_names=["feature"])

    with pytest.raises(DPGError):
        DecisionPredicateGraph(model=object(), feature_names=["feature"])


def test_configuration_error_identifies_invalid_graph_mode():
    model, _ = _iris_model()

    with pytest.raises(DPGConfigurationError, match="Unsupported graph construction mode"):
        DecisionPredicateGraph(
            model=model,
            feature_names=["sepal length (cm)"] * 4,
            dpg_config={"dpg": {"graph_construction": {"mode": "invalid"}}},
        )


def test_graph_error_identifies_filter_configuration():
    model, _ = _iris_model()
    dpg = DecisionPredicateGraph(
        model=model,
        feature_names=["sepal length (cm)"] * 4,
        dpg_config={
            "dpg": {
                "default": {"perc_var": 1.0, "decimal_threshold": 6, "n_jobs": 1}
            }
        },
    )

    empty_log = dpg._extract_trace_log(np.empty((0, 4)))
    with pytest.raises(DPGGraphError, match="No decision paths remain"):
        dpg.discover_dfg(dpg.filter_log(empty_log))


def test_not_fitted_error_preserves_value_error_compatibility():
    model, _ = _iris_model()
    explainer = DPGExplainer(model=model, feature_names=["feature"] * 4)

    with pytest.raises(DPGNotFittedError, match=r"Call fit\(X\)"):
        explainer.explain_global()

    with pytest.raises(ValueError):
        explainer.explain_global()


def test_dataset_and_model_validation_errors_are_specific():
    with pytest.raises(DPGDatasetError, match="Failed to load dataset"):
        select_dataset("does-not-exist.csv")

    with pytest.raises(DPGValidationError, match="Number of learners must be positive"):
        run_dpg(datasets="iris", n_learners=0)

    with pytest.raises(DPGModelError, match="Unsupported model"):
        run_dpg(datasets="iris", model_name="UnknownModel")
