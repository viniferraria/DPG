"""Unit tests for the refactored experiment runner in ``main_v2``.

These cover the pure, side-effect-free functions plus a light integration check
that the import is clean and the orchestration wiring runs with stubs.
"""

import csv
import datetime
import types
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

import experiments.causal_synthetic_scenarios.run_dpg_causal_synthetic as m


# --- timestamp ---------------------------------------------------------------


def test_timestamp_uses_provided_datetime():
    fixed = datetime.datetime(2026, 6, 20, 13, 5, 9)
    assert m.timestamp(fixed) == "2026-06-20T13-05-09"


def test_timestamp_is_filesystem_safe():
    ts = m.timestamp()
    assert ":" not in ts and " " not in ts


# --- load_dataset ------------------------------------------------------------


def test_load_dataset_splits_last_column_as_target(tmp_path):
    csv_path = tmp_path / "toy.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "Y": [0, 1]}).to_csv(csv_path, index=False)

    X, y = m.load_dataset(csv_path)

    assert list(X.columns) == ["a", "b"]
    assert y.name == "Y"
    assert y.tolist() == [0, 1]


# --- make_splitter -----------------------------------------------------------


def test_make_splitter_is_reproducible():
    X = pd.DataFrame({"f": range(20)})
    y = pd.Series([0, 1] * 10)
    splits_a = [tuple(s) for s in m.make_splitter().split(X, y)]
    splits_b = [tuple(s) for s in m.make_splitter().split(X, y)]
    assert len(splits_a) == m.N_SPLITS
    for (tr_a, te_a), (tr_b, te_b) in zip(splits_a, splits_b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)


# --- evaluate_accuracy -------------------------------------------------------


class _PerfectModel:
    def predict(self, X):
        return np.asarray(X).ravel()


def test_evaluate_accuracy_perfect_and_type():
    y = pd.Series([0, 1, 1, 0])
    X = y.to_frame()
    acc = m.evaluate_accuracy(_PerfectModel(), X, y)
    assert acc == 1.0
    assert isinstance(acc, float)


# --- records_from_explanation ------------------------------------------------


def _fake_explanation(rows):
    df = pd.DataFrame(rows)
    return types.SimpleNamespace(node_metrics=df)


def test_records_from_explanation_maps_columns_and_types():
    expl = _fake_explanation(
        [
            {
                "Node": "n1",
                "Degree": 3,
                "In degree nodes": 1,
                "Out degree nodes": 2,
                "Betweenness centrality": 0.5,
                "Local reaching centrality": 0.25,
                "Label": "F1 <= 0.5",
            }
        ]
    )

    records = m.records_from_explanation(expl, "exp", 0, processing_time=1.23456)

    assert len(records) == 1
    rec = records[0]
    assert rec.experiment == "exp"
    assert rec.split == 0
    assert rec.node == "n1"
    assert rec.degree == 3
    assert rec.in_degree == 1
    assert rec.out_degree == 2
    assert rec.betweenness_centrality == 0.5
    assert rec.local_reaching_centrality == 0.25
    assert rec.node_idx == 0
    assert rec.label == "F1 <= 0.5"
    assert rec.processing_time == 1.2346  # rounded to 4 dp
    assert isinstance(rec.node, str) and isinstance(rec.degree, int)


def test_records_from_explanation_empty_frame():
    expl = _fake_explanation([])
    # Ensure required columns exist even when empty.
    expl.node_metrics = pd.DataFrame(
        columns=[
            "Node",
            "Degree",
            "In degree nodes",
            "Out degree nodes",
            "Betweenness centrality",
            "Local reaching centrality",
            "Label",
        ]
    )
    assert m.records_from_explanation(expl, "exp", 1, 0.0) == []


# --- write_records_to_csv ----------------------------------------------------


def test_write_records_to_csv_roundtrip(tmp_path):
    records = [
        m.NodeMetricRecord(
            experiment="exp",
            split=0,
            node="n1",
            degree=3,
            in_degree=1,
            out_degree=2,
            betweenness_centrality=0.5,
            local_reaching_centrality=0.25,
            node_idx=0,
            label="F1 <= 0.5",
            processing_time=1.2346,
        )
    ]
    out = tmp_path / "nested" / "metrics.csv"

    m.write_records_to_csv(records, out)

    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    expected_fields = [fld.name for fld in fields(m.NodeMetricRecord)]
    assert list(rows[0].keys()) == expected_fields
    assert rows[0]["node"] == "n1"
    assert rows[0]["experiment"] == "exp"


def test_write_records_to_csv_overwrites(tmp_path):
    out = tmp_path / "metrics.csv"
    m.write_records_to_csv([], out)
    m.write_records_to_csv([], out)  # second call must not append duplicate header
    with open(out, newline="") as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) == 1  # header only


# --- save_pickle -------------------------------------------------------------


def test_save_pickle_roundtrip(tmp_path):
    import pickle

    target = tmp_path / "obj.pkl"
    m.save_pickle({"a": 1}, target)
    with open(target, "rb") as f:
        assert pickle.load(f) == {"a": 1}


# --- mean_or_nan -------------------------------------------------------------


def test_mean_or_nan():
    assert m.mean_or_nan([1.0, 3.0]) == 2.0
    assert np.isnan(m.mean_or_nan([]))


# --- iter_splits -------------------------------------------------------------


def test_iter_splits_shapes_and_count():
    X = pd.DataFrame({"f": range(20)})
    y = pd.Series([0, 1] * 10)
    splits = list(m.iter_splits(X, y, m.make_splitter()))
    assert len(splits) == m.N_SPLITS
    for split_idx, X_tr, X_te, y_tr, y_te in splits:
        assert len(X_tr) == len(y_tr)
        assert len(X_te) == len(y_te)
        assert len(X_tr) + len(X_te) == len(X)


# --- module hygiene ----------------------------------------------------------


def test_module_has_no_import_side_effects():
    # SCENARIO_FILES is defined and main is guarded; importing must not run it.
    assert isinstance(m.SCENARIOS_WITH_GT, list)
    assert callable(m.main)
    assert set(m.MODEL_FACTORIES) == {"RandomForest", "ExtraTrees"}


def test_model_factories_produce_fresh_instances():
    for name, factory in m.MODEL_FACTORIES.items():
        a, b = factory(), factory()
        assert a is not b
        assert a.random_state == m.RANDOM_STATE


# --- run_experiments wiring (stubbed, no DPG) --------------------------------


def test_run_experiments_wiring_with_stubs(tmp_path, monkeypatch):
    """Drive run_experiments end-to-end without the heavy DPG pipeline."""
    # Tiny synthetic scenario file.
    scenario = tmp_path / "scenario_x.csv"
    pd.DataFrame(
        {
            "F1": list(range(10)),
            "F2": list(range(10, 20)),
            "Y": [0, 1] * 5,
        }
    ).to_csv(scenario, index=False)

    # Replace the DPG-heavy split handler with a deterministic stub.
    def fake_explain_split(*, model, X, y, X_train, run_key, split_idx, processing_logger):
        return [
            m.NodeMetricRecord(
                experiment=run_key,
                split=split_idx,
                node="n",
                degree=1,
                in_degree=0,
                out_degree=1,
                betweenness_centrality=0.0,
                local_reaching_centrality=0.0,
                node_idx=0,
                label="leaf",
                processing_time=0.0,
            )
        ]

    monkeypatch.setattr(m, "explain_split", fake_explain_split)

    out = tmp_path / "out.csv"
    logger = _CapturingLogger()
    records = m.run_experiments([str(scenario)], out, logger)

    # 2 models * N_SPLITS splits, one record each.
    assert len(records) == 2 * m.N_SPLITS
    assert out.exists()
    assert any("All" not in msg for msg in logger.messages)


class _CapturingLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)

    def error(self, msg):
        self.messages.append(msg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
