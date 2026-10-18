"""Run DPG causal-accuracy experiments on ACIC 2019 Data Challenge datasets.

Mirrors ``experiments/monks/run_monk.py``. Each dataset is a treatment-effect
problem ``(Y, A, V1..Vp)``; a ``CausalForest`` is fitted on ``X=V``, ``T=A``,
``y=Y`` and its DPG is ranked against the feature(s) that drive the
heterogeneity of the treatment effect in the published data-generating process.
Fetch the data first with ``fetch_acic2019.py``.
"""

import argparse
import datetime
import logging
import pickle
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from econml.grf import CausalForest

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

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# --- Configuration -----------------------------------------------------------
RANDOM_STATE = 42
N_DATASETS = 5  # of the 100 datasets drawn from each DGP; see --n-datasets

SUITE_DIR = Path(__file__).parent
DATASETS_DIR = SUITE_DIR / "datasets"
STATES_DIR = SUITE_DIR / "states"
OUTPUT_PATH = SUITE_DIR / "results"

# (DGP id, ground-truth causal features, files merged per dataset). DGP ids are the ``DGPid`` column of
# the official ``lowDim_trueATE.csv``; ``fetch_acic2019.py`` stores each DGP's
# 100 files under ``datasets/dgp<id>/``.
#
# DGP 64 = student performance 2 (UCI student-por), continuous outcome,
# modification 4 ("scenario 4" in Benard & Josse 2023). From
# ``low_studentperf2_cont_sim.R``:
#   Y ~ N(2*A + 0.5*A*rank(age)/100 + lp(X), .)
# so the treatment effect varies with age alone. age is the 3rd covariate
# after the generator drops ``address`` (the treatment) and G1..G3, i.e. V3.
# Only the effect modifier is ground truth; confounders are deliberately not.
#
# DGP 39 = spam email (UCI spambase), continuous outcome, 500 x 24 files,
# "Mod 3" in ``generate_simSpam_cont.R``, whose heterogeneous effect is
#   3 * (1 - d[,21] + 2.2 * d[,10])    (line 159)
# with ``d`` = (Y, A, V1..V22), so d[,21] = V19 and d[,10] = V8: the paper's
# X(8) and X(19). (The code's "Mod 4", DGP 40, modifies on V11, V12, V15, V20.)
# Files are 500-row bootstrap draws, so 20 are concatenated per dataset to reach
# the paper's n = 10000.
SCENARIO_GROUND_TRUTH: list[tuple[int, list[str], int]] = [
    (64, ["V3"], 1),
    (39, ["V8", "V19"], 20),
]

# Passed explicitly: node identity depends on decimal_threshold, so the run must
# not depend on the CWD-relative config.yaml. Values match the repo config.yaml.
DPG_CONFIG: dict[str, Any] = {
    "dpg": {
        "default": {"perc_var": 0.0001, "decimal_threshold": 3, "n_jobs": -1},
        "graph_construction": {"mode": "aggregated_transitions", "context_order": 1},
        "metrics": {"backend": "igraph"},
        "visualization": {},
    }
}

# Each entry maps a model name to a zero-arg factory producing a fresh estimator.
# criterion="het" (not the default "mse") so splits target effect heterogeneity
# rather than prognostic signal; see okf/docs/concepts/causal-forest-support.md.
ModelFactory = Callable[[], Any]
MODEL_FACTORIES: dict[str, ModelFactory] = {
    "CausalForest": lambda: CausalForest(
        n_estimators=100, criterion="het", random_state=RANDOM_STATE, n_jobs=-1
    ),
}

# --- Pure helpers ------------------------------------------------------------


def timestamp(now: datetime.datetime | None = None) -> str:
    """Return a filesystem-safe timestamp string."""
    return (now or datetime.datetime.now(tz=datetime.timezone.utc)).strftime(
        "%Y-%m-%dT%H-%M-%S"
    )


logger = get_logger(__name__, log_file=str(SUITE_DIR / f"dpg_explainer_{timestamp()}.log"))


def load_dataset(
    paths: str | Path | list[Path],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load ACIC 2019 CSV(s) (``Y, A, V1..Vp``), concatenated, into ``(X, A, Y)``."""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df.drop(columns=["Y", "A"]), df["A"], df["Y"]


def list_datasets(
    dgp_id: int,
    n_datasets: int,
    datasets_dir: Path = DATASETS_DIR,
    n_merge: int = 1,
) -> list[list[Path]]:
    """Return ``n_datasets`` groups of ``n_merge`` files of a DGP, in numeric filename order.

    Each group is concatenated into one dataset by :func:`load_dataset`.
    """
    folder = datasets_dir / f"dgp{dgp_id}"
    files = sorted(folder.glob("low*.csv"), key=lambda p: int(p.stem[3:]))
    if not files:
        raise FileNotFoundError(f"No datasets in {folder}; run fetch_acic2019.py first.")
    groups = [files[i : i + n_merge] for i in range(0, len(files) - n_merge + 1, n_merge)]
    return groups[:n_datasets]


def true_cate_dgp64(X: pd.DataFrame) -> np.ndarray:
    """True CATE of DGP 64: ``2 + 0.5 * rank(age) / 100`` (age is ``V3``).

    ``rank`` is the generator's R ``rank`` (average ties), taken over the
    dataset's own rows.
    """
    return 2 + 0.5 * X["V3"].rank(method="average").to_numpy() / 100


def true_cate_dgp39(X: pd.DataFrame) -> np.ndarray:
    """True CATE of DGP 39: ``3 * (1 - V19 + 2.2 * V8)``."""
    return 3 * (1 - X["V19"].to_numpy() + 2.2 * X["V8"].to_numpy())


TRUE_CATE: dict[int, Callable[[pd.DataFrame], np.ndarray]] = {
    64: true_cate_dgp64,
    39: true_cate_dgp39,
}


def evaluate_cate_rmse(model: Any, X: pd.DataFrame, true_cate: np.ndarray) -> float:
    """Return RMSE of the forest's CATE estimates against the true CATE."""
    estimate = np.asarray(model.predict(X.values)).ravel()
    return float(np.sqrt(np.mean((estimate - true_cate) ** 2)))


def save_pickle(obj: Any, path: Path) -> None:
    """Pickle ``obj`` to ``path``."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# --- Orchestration (impure shell) -------------------------------------------


def _build_explanation(
    model: Any,
    X_train: pd.DataFrame,
    run_key: str,
    split_idx: int,
    states_dir: Path,
) -> tuple[DPGExplanation, float]:
    """Fit a DPG explainer on one dataset, persist artifacts, return its explanation."""
    explainer = DPGExplainer(
        model=model,
        feature_names=X_train.columns.tolist(),
        dpg_config=DPG_CONFIG,
    )
    explainer.fit(X_train.values)

    ts = timestamp()
    pkl_explainer = states_dir / f"{run_key}_split{split_idx}_dpg_explainer_{ts}.pkl"
    logger.info(f"Saving DPGExplainer → {pkl_explainer}")
    save_pickle(explainer, pkl_explainer)

    logger.info(f"Generating global explanation for {run_key} split={split_idx}...")
    start = time.time()
    # communities=False: community extraction raises on a regression DPG (Pred sinks).
    explanation = explainer.explain_global(communities=False)
    processing_time = time.time() - start
    logger.info(f"Global explanation generated in {processing_time:.2f}s")

    pkl_explanation = states_dir / f"{run_key}_split{split_idx}_explanation_{ts}.pkl"
    logger.info(f"Saving explanation → {pkl_explanation}")
    save_pickle(explanation, pkl_explanation)

    logger.info(
        f"Collected {len(explanation.node_metrics)} node records for {run_key} split={split_idx}"
    )
    return explanation, processing_time


def run_experiments_with_ground_truth(
    scenarios_with_gt: list[tuple[int, list[str], int]],
    output_dir: Path,
    states_dir: Path,
    n_datasets: int,
    logger: Any,
    datasets_dir: Path = DATASETS_DIR,
):
    """Run experiments per DGP, tracking causal accuracy against its ground truth.

    Each of the first ``n_datasets`` groups of a DGP's files is one split
    (``split_idx`` = the first file's numeric id; a group of ``n_merge`` files
    is concatenated). The forest is fitted on the whole
    dataset: there is no held-out set because the target is the explanation,
    not generalisation; CATE RMSE against the known true effect is logged as
    the model-quality check instead.
    """
    ts = timestamp()
    node_metrics_path = output_dir / f"node_metrics_{ts}.csv"
    accuracy_path = output_dir / f"causal_accuracy_{ts}.csv"

    for dgp_id, gt_features, n_merge in scenarios_with_gt:
        for paths in list_datasets(dgp_id, n_datasets, datasets_dir, n_merge):
            name = paths[0].name if len(paths) == 1 else f"{paths[0].name}..{paths[-1].name}"
            logger.info(f"Processing DGP {dgp_id}: {name}...")
            split_idx = int(paths[0].stem[3:])
            X, A, Y = load_dataset(paths)

            for model_name, model_factory in MODEL_FACTORIES.items():
                run_key = f"dgp{dgp_id}_{model_name}"

                model = model_factory()
                model.fit(X.values, A.values, Y.values)
                rmse = evaluate_cate_rmse(model, X, TRUE_CATE[dgp_id](X))

                try:
                    explanation, processing_time = _build_explanation(
                        model=model,
                        X_train=X,
                        run_key=run_key,
                        split_idx=split_idx,
                        states_dir=states_dir,
                    )
                    append_csv(
                        node_metrics_rows(explanation, run_key, split_idx, processing_time),
                        node_metrics_path,
                    )
                    append_csv(
                        causal_accuracy_rows(
                            explanation.node_metrics, gt_features, run_key, split_idx
                        ),
                        accuracy_path,
                    )

                except Exception as exc:  # noqa: BLE001 - keep batch resilient
                    logger.error(f"Failed: {run_key}: {exc}")
                    traceback.print_exc()

                logger.info(f"[{run_key}] CATE RMSE on {name}: {rmse:.4f}")


def main() -> None:
    """Entry point: configure logging and run all experiments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-datasets", type=int, default=N_DATASETS)
    parser.add_argument("--out-dir", type=Path, default=SUITE_DIR)
    args = parser.parse_args()

    states_dir = args.out_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__, log_file=str(SUITE_DIR / f"dpg_explainer_{timestamp()}.log"))

    logger.info("Starting DPG explainer experiments...")
    run_experiments_with_ground_truth(
        SCENARIO_GROUND_TRUTH,
        args.out_dir / "results",
        states_dir,
        args.n_datasets,
        logger,
    )
    logger.info("All experiments completed.")


if __name__ == "__main__":
    main()
