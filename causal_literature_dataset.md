The user chose analysis-only (no code), so this is a research deliverable rather than an implementation plan — I'll present the findings directly. I've written the full write-up to /Users/vinicius/.claude/plans/is-it-possible-to-prancy-firefly.md. Summary:

Q1 — Keep only a NetworkX structure in memory? Feasible, with one caveat.

The graphviz.Digraph plays two separate roles:

- Intermediate representation — generate_dot() → to_networkx() is a wasteful round-trip: the dfg dict {(src_label, tgt_label): count} is serialized to DOT and then regex-parsed back into nx.DiGraph (core.py:484-512). You can build the nx.DiGraph + _nodes list directly from dfg, deleting all DOT parsing. Clean win.
- Styling/render carrier — the whole visualizer.py is bound to the stored _dot object: it mutates dot.body strings and renders via Source(dot.source).pipe("png"). To drop graphviz from memory entirely, you'd regenerate the Digraph lazily from the nx graph at render time.

So: networkx-only-in-memory is achievable; it removes the round-trip cheaply but pushes Digraph construction into the visualizer. Three invariants must hold: SHA-1 node ids, weight edge attribute, and the _nodes = sorted([id,label]) list.

Q2 — Replace with igraph's to_graphviz? No.

python-igraph has no to_graphviz() returning a graphviz.Digraph. It only has the lossy Graph.write_dot() (no reliable per-node fillcolor, rankdir, edge penwidth/label) plus Cairo plotting. The visualizer relies on graphviz-specific dot.body/.source/.pipe/.attr behavior that igraph can't supply. igraph is already a dep (pyproject.toml:40) but only as a fast centrality engine in metrics/nodes.py — keep it there. Keep graphviz as the render backend.

Q3 — Properties generate_dot produces (full table in the file)

- Graph: bgcolor, rankdir (config), overlap, fontsize (hardcoded) → re-applied by visualizer layout templates.
- Nodes: id = str(int(sha1(activity),16)) (the system-wide key, used in every metrics module + recomputed in explainer._label_to_node_id); label = escaped activity (id↔label maps, node_lookup); style=filled, fontsize=20; fillcolor (config, mutated by change_node_color/highlight_class_node); shape.
- Edges: source/target id (SHA-1); label = frequency → recovered as float weight, consumed by all metrics + penwidth scaling + Markov calcs in metrics/graph.py; penwidth=1 (overwritten by change_edge_color); fontsize=18.
- Fragile DOT-format couplings a refactor would remove: the label="..." regex and -> parsing in to_networkx, plus dot.body line mutations in utils.py and visualizer.py:2710-2715.

I'm not calling ExitPlanMode since no code implementation was requested — if you later want me to actually do the networkx-only refactor (the recommended direction), say the word and I'll plan the implementation.

---

# Real/semi-synthetic datasets with explicit causal features (Kaggle + UCI survey, 2026-07-10)

Goal: extend the `main.py` causal-feature-recovery analysis beyond the synthetic `test_datasets/scenario_*.csv` files. Selection criteria mirror the scenario design: single CSV, one target column, a **documented causal feature subset** (from a published DGP, RCT, or physical mechanism) plus **explicit irrelevant distractors** to test false positives, sized for sklearn RandomForest/ExtraTrees.

Three parallel web-research passes (Kaggle, UCI, causal benchmark suites); all URLs verified live unless noted.

## Tier 1 — hard ground truth from a known generator, with explicit distractors (closest analog to the scenarios)

1. **MONK's Problems** — UCI id 70, https://archive.ics.uci.edu/dataset/70/monk+s+problems
   432 rows × 6 categorical attrs, binary target. Target is an exact published Boolean formula per variant (MONK-1: `(a1=a2) OR (a5=1)` → only a1/a2/a5 causal; MONK-3 adds 5% label noise). a3/a6 are pure distractors. Loadable via `ucimlrepo.fetch_ucirepo(id=70)`; needs OHE (pipeline already handles OHE columns). **Best drop-in for the classification pipeline.**
2. **Waveform Database Generator v2** — UCI id 107 family
   5000 rows × 40 continuous attrs (21 signal + **19 explicitly-labeled pure-noise columns**), 3-class. CART-book generator. Ships as C source/.Z archives — one-time conversion to CSV needed.
3. **LED Display Domain +17** — UCI id 57, https://archive.ics.uci.edu/dataset/57/led+display+domain
   Deterministic generator: 7 causal segment attrs (10% flip noise, Bayes-optimal 74%) + **17 injected uniform-noise attrs** in the `led-creator-+17.c` variant. Requires compiling the bundled C generator (`ucimlrepo` id 57 only gives the base 7-attr version). Textbook irrelevant-feature stress test.
4. **Madelon** — UCI id 171, https://archive.ics.uci.edu/dataset/171/madelon (Kaggle mirror: `sayroy1997/madelon`)
   4400 rows × 500 features: 5 real informative + 15 linear combos + **480 documented probes**, binary target. Strongest false-positive test, but needs column subsetting (e.g. 5 informative + 5 probes) to match the 5–12-feature scenario scale. `ucimlrepo` id 171. (Arcene/Gisette/Dorothea are the same NIPS-2003 family but even higher-dimensional — fallbacks only.)

## Tier 2 — full causal DAG ground truth (generate-to-CSV, not a static Kaggle/UCI file)

5. **bnlearn networks via pgmpy** — ASIA (8 nodes), INSURANCE (27), ALARM (37); DAGs published at https://www.bnlearn.com/bnrepository/
   ```python
   from pgmpy.utils import get_example_model
   df = get_example_model("asia").simulate(n_samples=1000)
   df.to_csv("test_datasets/asia.csv", index=False)  # pick e.g. "dysp" as target
   ```
   Richer than a causal-subset label: the full DAG lets DPG's recovered predicate paths be scored against true edges, not just feature sets. ASIA fits the 5–12-feature range exactly; all-categorical is a good tree-ensemble stress test.
   *(LUCAS/LUCAP — 11-node lung-cancer BN with published CPTs + explicit probe variables — is conceptually the ideal fit but hosted only at causality.inf.ethz.ch, not Kaggle/UCI.)*

## Tier 3 — RCT / semi-synthetic treatment-effect data (Kaggle; causal ground truth for the treatment column, established externally)

6. **IHDP** — `kaggle datasets download -d konradb/ihdp-data`
   747 rows × 25 covariates + treatment + simulated outcomes (Hill 2011 response-surface DGP; only a covariate subset enters the true surface → real distractors). Continuous target fits the regression scenarios. DGP details live in the literature (Hill 2011, Shalit et al.), not on the Kaggle page.
7. **Hillstrom E-Mail Analytics** — `bofulee/kevin-hillstrom-minethatdata-e-mailanalytics`
   64k rows × ~9 cols, real randomized 3-arm email campaign; treatment causal by design, clean interpretable covariates. Also via `scikit-uplift fetch_hillstrom()`.
8. **LaLonde/NSW** — `samuelzakouri/lalonde`: 614 rows, real job-training RCT, published ATE. Small n limits DPG path statistics.
9. **MegaFon Uplift** — `mrmorj/megafon-uplift-competition`: 600k × 50 anonymized features, randomized treatment (p=0.5). Scale is good for false-positive testing but anonymization blocks feature-level causal claims.

## Tier 4 — mechanistic/physics ground truth (all features causal → true-positive checks only, no distractors)

- **Energy Efficiency (ENB2012)** — UCI id 242: 768×8, Ecotect building-physics simulation, 2 regression targets; orientation (X6) is a documented near-irrelevant feature — the closest thing to a natural distractor in this tier.
- **Concrete Compressive Strength** — UCI id 165: 1030×8, regression (Yeh 1998).
- Airfoil Self-Noise (id 291, 1503×5), Combined Cycle Power Plant (id 294, 9568×4), Yacht Hydrodynamics (id 243, 308×6) — usable but below/at the low end of the feature range.

## Ruled out

- **Tübingen cause-effect pairs** — bivariate only; wrong shape for feature-subset recovery. Not on UCI (webdav.tuebingen.mpg.de / Zenodo).
- **Sachs protein signaling** — excellent interventional ground truth but no Kaggle/UCI mirror (bnlearn.com / Zenodo / `pgmpy get_example_model("sachs")` if the hosting constraint is relaxed).
- **Twins, ACIC** — no credible Kaggle/UCI mirrors (GitHub/Synapse only).
- **Criteo uplift v2.1 Kaggle mirror** — real RCT but 3.25 GB and the mirror is undocumented/unofficial.
- **`davinwijaya/customer-retention`** — RCT status is tutorial folklore, no citable primary source.
- **REGED/SIDO/CINA/MARTI** — causal graphs withheld (competition data), not on Kaggle/UCI.

## Recommended next additions to `main.py` scenarios

1. **MONK's-1 and MONK-3** (classification, categorical, explicit distractors) — near-zero prep.
2. **ASIA via pgmpy** (classification, full-DAG scoring) — one-line generation, commit the CSV to `test_datasets/`.
3. **Madelon 10-column subset** (5 informative + 5 probes) — high-value false-positive test at scenario scale.
4. **IHDP** (regression counterpart) and **Energy Efficiency** (real-world regression, weak-distractor X6) once the pipeline supports regression targets — note `main.py` currently uses classifiers + `StratifiedShuffleSplit`, so Tier 3/4 regression datasets need a regressor branch first.