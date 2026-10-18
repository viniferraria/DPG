#!/usr/bin/env -S uv run python
"""Run DPG explainer experiments on the ProPublica COMPAS violent-recidivism score.

COMPAS is real data, so there is no structural causal model. The reference used
here is the *qualitative* finding of Vuk, Ihlo & Behr (arXiv:2512.11081v2,
Sec. 5.3): ``priors`` and ``age`` are the globally most important features of a
random forest fitted to the COMPAS score, with ethnicity ranked lower. See
``okf/docs/experiments/compas.md`` for why this is a reference ranking, not a
ground truth.
"""

import argparse
import datetime
import logging
import pickle
import time
import traceback
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dpg import DPGExplainer
from dpg.explainer import DPGExplanation
from experiments.causal_accuracy import (
    append_csv,
    causal_accuracy_rows,
    node_metrics_rows,
)


def get_logger(name, log_file=None):
    """Create a console logger, optionally also writing to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        return logger
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


# --- Configuration -----------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]

STATES_DIR = Path(__file__).parent / "states"
OUTPUT_PATH = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "datasets/compas-scores-two-years.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

# Reference ranking (not a ground truth): the paper's globally most important
# features, expressed in this runner's feature names.
REFERENCE_FEATURES: list[str] = ["priors_count", "age"]
CATEGORICAL = ["c_charge_degree", "race", "sex"]
FEATURES = ["age", "priors_count", "c_charge_degree", "race", "sex"]

# Explicit so results do not depend on the CWD (mirrors the repo config.yaml).
DPG_CONFIG: dict[str, Any] = {
    "dpg": {
        "default": {"perc_var": 0.0001, "decimal_threshold": 3, "n_jobs": -1},
        "graph_construction": {"mode": "aggregated_transitions", "context_order": 1},
        "metrics": {"backend": "igraph"},
    }
}

ModelFactory = Callable[[int], Any]
MODEL_FACTORIES: dict[str, ModelFactory] = {
    "RandomForest": lambda seed: RandomForestClassifier(
        n_estimators=100, min_samples_leaf=5, random_state=seed
    ),
    "ExtraTrees": lambda seed: ExtraTreesClassifier(
        n_estimators=100, min_samples_leaf=5, random_state=seed
    ),
}

# --- Pure helpers ------------------------------------------------------------


def timestamp(now: datetime.datetime | None = None) -> str:
    """Return a filesystem-safe timestamp string."""
    return (now or datetime.datetime.now(tz=datetime.timezone.utc)).strftime(
        "%Y-%m-%dT%H-%M-%S"
    )


def download_dataset(path: Path = DATASET_PATH) -> Path:
    """Fetch ProPublica's CSV if it is not already on disk (it is not committed)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, path)
    return path


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load COMPAS and apply ProPublica's filters plus the paper's restriction.

    Filters: ``days_b_screening_arrest`` within +-30, ``is_recid != -1``,
    ``c_charge_degree != 'O'``, ``v_score_text`` in Low/Medium/High (pandas
    reads the literal "N/A" as NaN, so a ``!= 'N/A'`` test would keep it); then only
    African-American and Caucasian defendants. The target is 1 when the violent
    recidivism score text is Medium or High (ProPublica's "high risk" grouping;
    the paper only says "high (= 1) and low (= 0)"). Identifiers (name, dob...)
    are never loaded into ``X``.
    """
    df = pd.read_csv(path)
    df = df[
        df["days_b_screening_arrest"].between(-30, 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & df["v_score_text"].isin(["Low", "Medium", "High"])
        & df["race"].isin(["African-American", "Caucasian"])
    ]
    y = (df["v_score_text"] != "Low").astype(int).reset_index(drop=True)
    X = df[FEATURES].reset_index(drop=True)
    return X, y


def pre_process_dataset(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    """One-hot encode the categorical columns across several frames at once.

    Columns become e.g. ``race_African-American`` (bool). All frames are encoded
    together so they share one column set; only the vocabulary is shared.
    """
    combined = pd.concat(frames, keys=range(len(frames)))
    encoded = pd.get_dummies(combined, columns=CATEGORICAL, dtype=bool)
    return [encoded.xs(position) for position in range(len(frames))]


def base_feature_name(label: str) -> str:
    """Map a one-hot column back to its source feature ("race_Caucasian" -> "race")."""
    for column in CATEGORICAL:
        if str(label).startswith(f"{column}_"):
            return column
    return str(label)


def evaluate_accuracy(model: Any, X_test: Any, y_test: Any) -> float:
    """Return accuracy of a fitted ``model`` on a held-out split."""
    return float(accuracy_score(y_test, model.predict(X_test)))


def save_pickle(obj: Any, path: Path) -> None:
    """Pickle ``obj`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# --- Orchestration (impure shell) -------------------------------------------


def _build_explanation(
    model: Any,
    X_train: pd.DataFrame,
    y: pd.Series,
    run_key: str,
    split_idx: int,
    states_dir: Path,
    logger: Any,
) -> tuple[DPGExplanation, float]:
    """Fit a DPG explainer on one split, persist artifacts, return its explanation."""
    explainer = DPGExplainer(
        model=model,
        feature_names=X_train.columns.tolist(),
        target_names=np.unique(y).astype(str).tolist(),
        dpg_config=DPG_CONFIG,
    )
    # Fit on training data only to avoid test-data leak.
    explainer.fit(X_train.values)

    ts = timestamp()
    pkl_explainer = states_dir / f"{run_key}_split{split_idx}_dpg_explainer_{ts}.pkl"
    logger.info(f"Saving DPGExplainer → {pkl_explainer}")
    save_pickle(explainer, pkl_explainer)

    logger.info(f"Generating global explanation for {run_key} split={split_idx}...")
    start = time.time()
    explanation = explainer.explain_global(communities=True)
    processing_time = time.time() - start
    logger.info(f"Global explanation generated in {processing_time:.2f}s")

    pkl_explanation = states_dir / f"{run_key}_split{split_idx}_explanation_{ts}.pkl"
    logger.info(f"Saving explanation → {pkl_explanation}")
    save_pickle(explanation, pkl_explanation)
    return explanation, processing_time


def run_experiments(
    seeds: list[int],
    model_names: list[str],
    output_dir: Path,
    states_dir: Path,
    logger: Any,
) -> None:
    """Fit each model per seed on a stratified 80/20 split and score the DPG ranking."""
    X_raw, y = load_dataset(download_dataset())
    # Encoding before the split shares only the category vocabulary, not the target.
    (X,) = pre_process_dataset(X_raw)
    ts = timestamp()
    node_metrics_path = output_dir / f"node_metrics_{ts}.csv"
    accuracy_path = output_dir / f"causal_accuracy_{ts}.csv"

    for split_idx, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        for model_name in model_names:
            run_key = f"compas_{model_name}"
            model = MODEL_FACTORIES[model_name](seed)
            model.fit(X_train, y_train)
            accuracy = evaluate_accuracy(model, X_test, y_test)
            try:
                explanation, processing_time = _build_explanation(
                    model, X_train, y_train, run_key, split_idx, states_dir, logger
                )
                append_csv(
                    node_metrics_rows(
                        explanation,
                        run_key,
                        split_idx,
                        processing_time,
                        seed=seed,
                    ),
                    node_metrics_path,
                )
                append_csv(
                    causal_accuracy_rows(
                        explanation.node_metrics,
                        REFERENCE_FEATURES,
                        run_key,
                        split_idx,
                        feature_name=base_feature_name,
                    ),
                    accuracy_path,
                )
            except Exception as exc:  # noqa: BLE001 - keep batch resilient
                logger.error(f"Failed: {run_key}: {exc}")
                traceback.print_exc()
            logger.info(
                f"[{run_key}] seed={seed} accuracy on held-out "
                f"({len(X_test)} rows): {accuracy:.4f}"
            )


def main(argv: list[str] | None = None) -> None:
    """Entry point: configure logging and run all experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_FACTORIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--states-dir", type=Path, default=STATES_DIR)
    args = parser.parse_args(argv)

    args.states_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__, log_file=str(Path(__file__).parent / f"dpg_explainer_{timestamp()}.log"))
    logger.info("Starting DPG explainer experiments...")
    run_experiments(args.seeds, args.models, args.output_dir, args.states_dir, logger)
    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()
