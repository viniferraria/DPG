1. **MONK's-1 and MONK-3** (classification, categorical, explicit distractors) — near-zero prep.
2. **Madelon 10-column subset** (5 informative + 5 probes) — high-value false-positive test at scenario scale.
3. **IHDP** (regression counterpart) and **Energy Efficiency** (real-world regression, weak-distractor X6) once the pipeline supports regression targets — note `main.py` currently uses classifiers + `StratifiedShuffleSplit`, so Tier 3/4 regression datasets need a regressor branch first.
4. **bnlearn networks via pgmpy** — ASIA (8 nodes), INSURANCE (27), ALARM (37); DAGs published at https://www.bnlearn.com/bnrepository/
   ```python
   from pgmpy.utils import get_example_model
   df = get_example_model("asia").simulate(n_samples=1000)
   df.to_csv("test_datasets/asia.csv", index=False)  # pick e.g. "dysp" as target
   ```
   Richer than a causal-subset label: the full DAG lets DPG's recovered predicate paths be scored against true edges, not just feature sets. ASIA fits the 5–12-feature range exactly; all-categorical is a good tree-ensemble stress test.
   *(LUCAS/LUCAP — 11-node lung-cancer BN with published CPTs + explicit probe variables — is conceptually the ideal fit but hosted only at causality.inf.ethz.ch, not Kaggle/UCI.)*

5. **Energy Efficiency (ENB2012)** — UCI id 242: 768×8, Ecotect building-physics simulation, 2 regression targets; orientation (X6) is a documented near-irrelevant feature — the closest thing to a natural distractor in this tier.
6. **Concrete Compressive Strength** — UCI id 165: 1030×8, regression (Yeh 1998).
7. Airfoil Self-Noise (id 291, 1503×5), Combined Cycle Power Plant (id 294, 9568×4), Yacht Hydrodynamics (id 243, 308×6) — usable but below/at the low end of the feature range.
8. **Twins, ACIC** — no credible Kaggle/UCI mirrors (GitHub/Synapse only).
9.  **Tübingen cause-effect pairs** — bivariate only; wrong shape for feature-subset recovery. Not on UCI (webdav.tuebingen.mpg.de / Zenodo).
10. **Sachs protein signaling** — excellent interventional ground truth but