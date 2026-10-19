#!/usr/bin/env -S uv run python
# from __future__ import annotations
"""Run DPG explainer experiments across scenario datasets.

Refactor of ``main.py`` favouring small, pure, testable functions composed by a
thin imperative shell (``main``). Importing this module has no side effects, so
each function can be unit-tested in isolation.
"""

import datetime
import logging
import pickle
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from dpg import DPGExplainer
from dpg.explainer import DPGExplanation
from experiments.causal_accuracy import (
    append_csv,
    causal_accuracy_rows,
    node_metrics_rows,
)


def get_logger(name, log_file=None):
    """
    Factory function to create and configure a logger.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional log file path. If provided, logs will also be written to this file.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# --- Configuration -----------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 2
TEST_SIZE = 0.2

SUITE_DIR = Path(__file__).parent
STATES_DIR = SUITE_DIR / "states"
OUTPUT_PATH = SUITE_DIR / "results"

SCENARIO_GROUND_TRUTH: dict[str, list[str]] = {
    str(SUITE_DIR / "datasets/scenario_1.csv"): ["F1"],
    str(SUITE_DIR / "datasets/scenario_2_updated.csv"): ["F1"],
    str(SUITE_DIR / "datasets/scenario_3.csv"): ["F1", "F2"],
    str(SUITE_DIR / "datasets/scenario_5.csv"): ["F1", "F2"],
}
SCENARIOS_WITH_GT: list[str] = list(SCENARIO_GROUND_TRUTH)
SCM_ROOT = SUITE_DIR / "datasets/scm"  # <scenario>/seed_<n>/{X.csv,y.csv,ground_truth.yaml}

CONFIG_PATH = ((Path(__file__).parent).parent) / "../config.yaml"

# Each entry maps a model name to a zero-arg factory producing a fresh estimator.
ModelFactory = Callable[[], Any]
MODEL_FACTORIES: dict[str, ModelFactory] = {
    "RandomForest": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
    "ExtraTrees": lambda: ExtraTreesClassifier(random_state=RANDOM_STATE),
}


# --- Pure helpers ------------------------------------------------------------


def timestamp(now: datetime.datetime | None = None) -> str:
    """Return a filesystem-safe timestamp string."""
    return (now or datetime.datetime.now(tz=datetime.timezone.utc)).strftime("%Y-%m-%dT%H-%M-%S")


def make_splitter() -> StratifiedShuffleSplit:
    """Return a fresh, reproducible train/test splitter using the module config."""
    return StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV scenario into a feature frame ``X`` and target series ``y``.

    The last column is treated as the target, all preceding columns as features.
    A directory is read as the generator layout instead: ``X.csv`` and ``y.csv``.
    """
    if Path(path).is_dir():
        return pd.read_csv(Path(path) / "X.csv"), pd.read_csv(Path(path) / "y.csv").iloc[:, 0]
    df = pd.read_csv(path)
    return df.iloc[:, :-1], df.iloc[:, -1]


def scm_ground_truth(
    root: Path = SCM_ROOT,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, list[str]]:
    """Map each ``<root>/<scenario>/seed_<n>`` dir to its ``causal_features``."""
    return {
        str(d): yaml.safe_load((d / "ground_truth.yaml").read_text())["causal_features"]
        for sc in sorted(root.iterdir())
        if scenarios is None or sc.name in scenarios
        for d in sorted(sc.glob("seed_*"))
        if seeds is None or int(d.name.split("_")[1]) in seeds
    }


def evaluate_accuracy(model: Any, X_test: Any, y_test: Any) -> float:
    """Return accuracy of a fitted ``model`` on a held-out split."""
    return float(accuracy_score(y_test, model.predict(X_test)))


def save_pickle(obj: Any, path: Path) -> None:
    """Pickle ``obj`` to ``path``."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def mean_or_nan(values: list[float]) -> float:
    """Mean of ``values``, or NaN when empty."""
    return sum(values) / len(values) if values else float("nan")


# --- Orchestration (impure shell) -------------------------------------------


def iter_splits(
    X: pd.DataFrame, y: pd.Series, splitter: StratifiedShuffleSplit
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """Yield ``(split_idx, X_train, X_test, y_train, y_test)`` for each fold."""
    for split_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X.to_numpy(), y.to_numpy())
    ):
        yield (
            split_idx,
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
        )


def _build_explanation(
    model: Any,
    X_train: pd.DataFrame,
    y: pd.Series,
    run_key: str,
    split_idx: int,
    processing_logger: Any,
) -> tuple[DPGExplanation, float]:
    """Fit a DPG explainer on one split, persist artifacts, return its explanation."""
    explainer = DPGExplainer(
        model=model,
        feature_names=X_train.columns.tolist(),
        target_names=np.unique(y).astype(str).tolist(),
        config_file=str(CONFIG_PATH.resolve(strict=True)),
    )
    # Fit on training data only to avoid test-data leak.
    explainer.fit(X_train.values)

    ts = timestamp()
    pkl_explainer = STATES_DIR / f"{run_key}_split{split_idx}_dpg_explainer_{ts}.pkl"
    processing_logger.info(f"Saving DPGExplainer → {pkl_explainer}")
    save_pickle(explainer, pkl_explainer)

    processing_logger.info(
        f"Generating global explanation for {run_key} split={split_idx}..."
    )
    start = time.time()
    explanation = explainer.explain_global(communities=True)
    processing_time = time.time() - start
    processing_logger.info(f"Global explanation generated in {processing_time:.2f}s")

    pkl_explanation = STATES_DIR / f"{run_key}_split{split_idx}_explanation_{ts}.pkl"
    processing_logger.info(f"Saving explanation → {pkl_explanation}")
    save_pickle(explanation, pkl_explanation)

    processing_logger.info(
        f"Collected {len(explanation.node_metrics)} node records for {run_key} split={split_idx}"
    )
    return explanation, processing_time


def explain_split(
    *,
    model: Any,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    y: pd.Series,
    run_key: str,
    split_idx: int,
    processing_logger: Any,
) -> pd.DataFrame:
    """Fit a DPG explainer on one split and return its node-metric rows.

    ``X`` is the held-out split for this fold; the explanation itself is built
    from ``X_train`` only, to avoid test-data leak.
    """
    processing_logger.info(
        f"Explaining {run_key} split={split_idx} against {len(X)} held-out rows"
    )
    explanation, processing_time = _build_explanation(
        model=model,
        X_train=X_train,
        y=y,
        run_key=run_key,
        split_idx=split_idx,
        processing_logger=processing_logger,
    )
    return node_metrics_rows(explanation, run_key, split_idx, processing_time)


def run_experiments(
    scenario_files: list[str],
    output_path: Path,
    logger: Any,
) -> pd.DataFrame:
    """Run every (scenario, model, split) combination and return their node-metric rows."""
    frames: list[pd.DataFrame] = []
    splitter = make_splitter()

    for scenario_file in scenario_files:
        logger.info(f"Processing {scenario_file}...")
        scenario_name = Path(scenario_file).stem
        X, y = load_dataset(scenario_file)

        for model_name, model_factory in MODEL_FACTORIES.items():
            run_key = f"{scenario_name}_{model_name}"
            split_accuracies: list[float] = []

            for split_idx, X_train, X_test, y_train, y_test in iter_splits(
                X, y, splitter
            ):
                model = model_factory()
                model.fit(X_train, y_train)

                accuracy = evaluate_accuracy(model, X_test, y_test)
                split_accuracies.append(accuracy)

                try:
                    frames.append(
                        explain_split(
                            model=model,
                            X=X_test,
                            X_train=X_train,
                            y=y,
                            run_key=run_key,
                            split_idx=split_idx,
                            processing_logger=logger,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - keep batch resilient
                    logger.error(f"Failed: {run_key} split={split_idx}: {exc}")
                    traceback.print_exc()

            mean_accuracy = mean_or_nan(split_accuracies)
            logger.info(
                f"[{run_key}] mean accuracy over {len(split_accuracies)} splits: "
                f"{mean_accuracy:.4f}"
            )

    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frames:
        append_csv(all_rows, output_path)
    return all_rows


def run_experiments_with_ground_truth(
    scenarios_with_gt: dict[str, list[str]],
    output_dir: Path,
    logger: Any,
) -> pd.DataFrame:
    """Run experiments per scenario, additionally tracking causal accuracy against
    each scenario's ground-truth causal features.

    Node-metric rows (written once at the end) and causal-accuracy rows (appended
    per split) each go to one CSV per run under ``output_dir``, sharing the
    run's timestamp.
    """
    frames: list[pd.DataFrame] = []
    splitter = make_splitter()
    ts = timestamp()
    accuracy_path = output_dir / f"causal_accuracy_{ts}.csv"

    for scenario_file, gt_features in scenarios_with_gt.items():
        logger.info(f"Processing {scenario_file}...")
        p = Path(scenario_file)
        scenario_name = f"{p.parent.name}_{p.name}" if p.is_dir() else p.stem
        X, y = load_dataset(scenario_file)

        for model_name, model_factory in MODEL_FACTORIES.items():
            run_key = f"{scenario_name}_{model_name}"
            split_accuracies: list[float] = []

            for split_idx, X_train, X_test, y_train, y_test in iter_splits(
                X, y, splitter
            ):
                model = model_factory()
                model.fit(X_train, y_train)

                accuracy = evaluate_accuracy(model, X_test, y_test)
                split_accuracies.append(accuracy)

                try:
                    explanation, processing_time = _build_explanation(
                        model=model,
                        X_train=X_train,
                        y=y,
                        run_key=run_key,
                        split_idx=split_idx,
                        processing_logger=logger,
                    )
                    frames.append(
                        node_metrics_rows(
                            explanation, run_key, split_idx, processing_time
                        )
                    )
                    append_csv(
                        causal_accuracy_rows(
                            explanation.node_metrics, gt_features, run_key, split_idx
                        ),
                        accuracy_path,
                    )

                except Exception as exc:  # noqa: BLE001 - keep batch resilient
                    logger.error(f"Failed: {run_key} split={split_idx}: {exc}")
                    traceback.print_exc()

            mean_accuracy = mean_or_nan(split_accuracies)
            logger.info(
                f"[{run_key}] mean accuracy over {len(split_accuracies)} splits: "
                f"{mean_accuracy:.4f}"
            )

    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frames:
        append_csv(all_rows, output_dir / f"node_metrics_{ts}.csv")
    return all_rows


def main(include_scm: bool = True) -> None:
    """Entry point: configure logging and run all experiments.

    ``include_scm`` (``--scm`` on the command line) adds every dataset under ``SCM_ROOT``.
    """
    STATES_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.mkdir(exist_ok=True)
    logger = get_logger(__name__, log_file=str(SUITE_DIR / f"dpg_explainer_{timestamp()}.log"))

    logger.info("Starting DPG explainer experiments...")
    scenarios = {**SCENARIO_GROUND_TRUTH, **(scm_ground_truth() if include_scm else {})}
    run_experiments_with_ground_truth(scenarios, OUTPUT_PATH, logger)
    logger.info("All experiments completed.")


if __name__ == "__main__":
    main(include_scm="--scm" in sys.argv[1:])
