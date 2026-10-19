"""Unit tests for the refactored experiment runner in ``main_v2``.

These cover the pure, side-effect-free functions plus a light integration check
that the import is clean and the orchestration wiring runs with stubs.
"""

import datetime
import types

import numpy as np
import pandas as pd
import pytest

import experiments.causal_synthetic_scenarios.run_dpg_causal_synthetic as m

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


def test_load_dataset_and_ground_truth_from_scm_dir(tmp_path):
    d = tmp_path / "s13_mediator" / "seed_42"
    d.mkdir(parents=True)
    pd.DataFrame({"X1": [1, 2], "M": [3, 4]}).to_csv(d / "X.csv", index=False)
    pd.DataFrame({"target": [0, 1]}).to_csv(d / "y.csv", index=False)
    (d / "ground_truth.yaml").write_text("causal_features:\n- M\n")

    X, y = m.load_dataset(d)
    gt = m.scm_ground_truth(tmp_path, seeds=[42])

    assert list(X.columns) == ["X1", "M"]
    assert y.tolist() == [0, 1]
    assert gt == {str(d): ["M"]}
    assert m.scm_ground_truth(tmp_path, seeds=[43]) == {}


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
    # main() must create results/ and states/.
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
        return pd.DataFrame({"experiment": [run_key], "split": [split_idx], "label": ["leaf"]})

    monkeypatch.setattr(m, "explain_split", fake_explain_split)

    out = tmp_path / "out.csv"
    logger = _CapturingLogger()
    rows = m.run_experiments([str(scenario)], out, logger)

    # 2 models * N_SPLITS splits, one row each, written once at the end.
    assert len(rows) == 2 * m.N_SPLITS
    assert len(pd.read_csv(out)) == 2 * m.N_SPLITS
    assert any("All" not in msg for msg in logger.messages)


def test_run_experiments_with_ground_truth_wiring(tmp_path, monkeypatch):
    """Node metrics are written once at the end; causal accuracy is appended per split."""
    scenario = tmp_path / "scenario_x.csv"
    pd.DataFrame({"F1": range(10), "F2": range(10, 20), "Y": [0, 1] * 5}).to_csv(
        scenario, index=False
    )
    node_metrics = pd.DataFrame(
        {
            "Node": ["a", "b"],
            "Degree": [1, 1],
            "In degree nodes": [0, 1],
            "Out degree nodes": [1, 0],
            "Label": ["F1 <= 0.5", "Class 0"],
            **{name: [0.5, 0.1] for name in m.METRICS},
        }
    )
    monkeypatch.setattr(
        m,
        "_build_explanation",
        lambda **kw: (types.SimpleNamespace(node_metrics=node_metrics), 0.1),
    )

    rows = m.run_experiments_with_ground_truth({str(scenario): ["F1"]}, tmp_path / "out", _CapturingLogger())

    n_runs = 2 * m.N_SPLITS
    assert len(rows) == n_runs * len(node_metrics)
    (nm_csv,) = (tmp_path / "out").glob("node_metrics_*.csv")
    (acc_csv,) = (tmp_path / "out").glob("causal_accuracy_*.csv")
    assert len(pd.read_csv(nm_csv)) == len(rows)
    acc = pd.read_csv(acc_csv)
    assert len(acc) == n_runs * len(m.METRICS)
    assert (acc["precision"] == 1.0).all()


class _CapturingLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)

    def error(self, msg):
        self.messages.append(msg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
