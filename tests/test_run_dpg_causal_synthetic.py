"""Unit tests for the refactored experiment runner in ``main_v2``.

These cover the pure, side-effect-free functions plus a light integration check
that the import is clean and the orchestration wiring runs with stubs.
"""

import csv
import datetime
import types
from dataclasses import fields

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import experiments.causal_synthetic_scenarios.run_dpg_causal_synthetic as m
from metrics.nodes import NodeMetrics

# --- timestamp ---------------------------------------------------------------


def test_timestamp_uses_provided_datetime():
    fixed = datetime.datetime(2026, 6, 20, 13, 5, 9, tzinfo=datetime.timezone.utc)
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
                "Closeness centrality": 0.1,
                "Harmonic centrality": 0.2,
                "Collective influence": 4.0,
                "Local clustering coefficient": 0.3,
                "Percolation centrality": 0.4,
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
    assert rec.closeness_centrality == 0.1
    assert rec.harmonic_centrality == 0.2
    assert rec.collective_influence == 4.0
    assert rec.local_clustering_coefficient == 0.3
    assert rec.percolation_centrality == 0.4
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
            "Closeness centrality",
            "Harmonic centrality",
            "Collective influence",
            "Local clustering coefficient",
            "Percolation centrality",
            "Label",
        ]
    )
    assert m.records_from_explanation(expl, "exp", 1, 0.0) == []


def test_records_from_real_node_metrics_have_all_runner_columns():
    """A wrong metric/column name would raise KeyError, swallowed by the runner's per-run except."""
    G = nx.DiGraph()
    G.add_weighted_edges_from(
        [("a", "b", 3), ("a", "c", 2), ("b", "d", 3), ("c", "d", 1), ("c", "e", 1)]
    )
    nodes_list = [
        ["a", "F1 <= 0.5"],
        ["b", "F2 <= 1.5"],
        ["c", "F2 > 1.5"],
        ["d", "Class 0"],
        ["e", "Class 1"],
    ]

    df = NodeMetrics.extract_node_metrics(G, nodes_list)

    for name in m.METRICS:
        assert name in df.columns
    records = m.records_from_explanation(
        types.SimpleNamespace(node_metrics=df), "exp", 0, processing_time=0.0
    )
    assert len(records) == len(nodes_list)


# --- write_records_to_csv ----------------------------------------------------


@pytest.mark.parametrize(
    ("drop_duplicates", "expected"),
    [(False, ["F2", "F2", "F1"]), (True, ["F2", "F1", "F3"])],
)
def test_extract_top_k_features_duplicates(drop_duplicates, expected):
    explanation = pd.DataFrame(
        {
            "Label": ["F2 <= 0.5", "F2 > 0.5", "F1 <= 1.0", "F3 > 2.0", "Class 0"],
            "Percolation centrality": [0.9, 0.8, 0.7, 0.6, 1.0],
        }
    )
    top = m.extract_top_k_features(
        explanation,
        top_k=3,
        metric="Percolation centrality",
        drop_duplicates=drop_duplicates,
    )
    assert top == expected


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
            closeness_centrality=0.0,
            harmonic_centrality=0.0,
            collective_influence=0.0,
            local_clustering_coefficient=0.0,
            percolation_centrality=0.0,
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


def test_main_creates_results_and_states_dirs(tmp_path, monkeypatch):
    # write_causal_accuracy_to_csv does not mkdir; main() must create results/.
    monkeypatch.chdir(tmp_path)  # main() opens its log file in the CWD
    monkeypatch.setattr(m, "STATES_DIR", tmp_path / "states")
    monkeypatch.setattr(m, "OUTPUT_PATH", tmp_path / "results")
    monkeypatch.setattr(m, "run_experiments_with_ground_truth", lambda *a: None)

    m.main()

    assert (tmp_path / "states").is_dir()
    assert (tmp_path / "results").is_dir()


def test_module_has_no_import_side_effects():
    # SCENARIO_FILES is defined and main is guarded; importing must not run it.
    assert isinstance(m.SCENARIOS_WITH_GT, list)
    assert callable(m.main)
    assert set(m.MODEL_FACTORIES) == {"RandomForest", "ExtraTrees"}


def test_model_factories_produce_fresh_instances():
    for factory in m.MODEL_FACTORIES.values():
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
                closeness_centrality=0.0,
                harmonic_centrality=0.0,
                collective_influence=0.0,
                local_clustering_coefficient=0.0,
                percolation_centrality=0.0,
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
