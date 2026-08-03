# Top 5 Causal Datasets in the arXiv / Causal-ML Literature

The 5 most widely used datasets that carry **explicit causal features and ground truth**, ranked by citation frequency across arXiv causal-inference and causal-discovery papers. They fall into two canonical camps: **treatment-effect estimation** (ground truth = counterfactual outcomes / true ATE) and **causal discovery** (ground truth = a known causal graph / edge direction).

_Compiled 2026-07-10 from a literature/benchmark survey; see Sources at the end._

---

## 1. IHDP (Infant Health and Development Program)

- **Type:** Treatment-effect estimation (CATE/ATE), semi-synthetic.
- **Size / target:** 747 rows × 25 covariates + binary treatment; continuous outcome (child cognitive test score).
- **Causal ground truth:** Covariates and treatment come from a real RCT (home visits for premature infants); **outcomes are simulated from a published response surface** (Hill 2011, "NPCI" setup B), so true individual treatment effects and ATE are known. Only a subset of covariates enters the true surface → the rest are genuine distractors.
- **Why it's #1:** The single most-cited semi-synthetic benchmark in the treatment-effect literature (CFR/TARNet, CEVAE, Dragonnet, and nearly every CATE paper report on it).
- **Access:** Kaggle `konradb/ihdp-data` (CSV); NPCI generator (`vdorie/npci`); standard 100/1000-replication splits from Shalit et al.
- **Reference:** Hill (2011), *J. Comput. Graph. Stat.*

## 2. Twins

- **Type:** Treatment-effect estimation, semi-synthetic from real records.
- **Size / target:** ~11,400 same-sex twin pairs × ~39 covariates; binary outcome (first-year mortality).
- **Causal ground truth:** Built from US twin births 1989–1991. "Treatment" = being the heavier twin; because **both twins are observed, both potential outcomes are available**, giving exact individual-level ground truth. Confounding is introduced synthetically by selectively hiding one twin based on covariates.
- **Why it's popular:** The go-to companion to IHDP for CATE evaluation — real covariates, real outcomes, and true counterfactuals in one dataset.
- **Access:** Louizos et al. CEVAE repo (`AMLab-Amsterdam/CEVAE`); NBER linked birth/infant-death files. No clean Kaggle/UCI mirror.
- **Reference:** Louizos et al. (2017), *NeurIPS* (CEVAE); Almond et al. (2005).

## 3. Jobs / LaLonde (NSW + PSID)

- **Type:** Treatment-effect estimation, real RCT + observational control.
- **Size / target:** ~614 rows (NSW experimental) up to ~2,675 with PSID controls; ~8–10 covariates; outcome = 1978 earnings (continuous) or employment (binary).
- **Causal ground truth:** The National Supported Work Demonstration was a **genuine randomized trial**, so the treatment effect on the experimental sample is a trusted benchmark (Dehejia–Wahba ATE ≈ $1,794). Widely used to test whether observational estimators recover the experimental answer.
- **Why it's popular:** The classic real-RCT causal benchmark from econometrics; a standard "policy value / ATT" evaluation set in causal-ML papers.
- **Access:** NBER `users.nber.org/~rdehejia/data`; R packages `cobalt`, `qte`, `lalonde`; Kaggle `samuelzakouri/lalonde`.
- **Reference:** LaLonde (1986), *Am. Econ. Rev.*; Dehejia & Wahba (1999/2002).

## 4. Sachs Protein-Signaling Network

- **Type:** Causal discovery (structure learning), real experimental data.
- **Size / target:** 7,466 cells × 11 continuous protein/phospholipid measurements.
- **Causal ground truth:** A **consensus causal graph (11 nodes, ~17–20 edges)** established by Sachs et al. 2005 from **interventional** flow-cytometry experiments (each protein perturbed directly). This is the standard *real-world* ground-truth DAG for benchmarking structure-learning algorithms.
- **Why it's #1 for discovery:** Appears in essentially every causal-structure-learning paper (PC/GES, NOTEARS, DAG-GNN, GraN-DAG, and successors) as the real-data benchmark.
- **Access:** `bnlearn.com` (`sachs.data.txt.gz`, `sachs.interventional.txt.gz`); R `bnlearn` `data(sachs)`; Python `pgmpy.utils.get_example_model("sachs")`.
- **Reference:** Sachs et al. (2005), *Science*.

## 5. Tübingen Cause-Effect Pairs

- **Type:** Causal discovery (bivariate causal direction), real data.
- **Size / target:** 108 cause-effect pairs drawn from 37 real datasets (meteorology, biology, medicine, engineering, economics); each instance is a variable pair `(X, Y)`.
- **Causal ground truth:** The **true causal direction** (X→Y or Y→X) is annotated for every pair, with per-pair weights to avoid over-counting related pairs. The reference benchmark for testing whether a method infers the correct arrow; best published accuracy ≈ 83%.
- **Why it's popular:** The canonical benchmark for the bivariate/cause-effect-direction subfield (ANM, IGCI, RCC, and LLM-based causal reasoning papers all report on it).
- **Access:** `webdav.tuebingen.mpg.de/cause-effect/` (pairs + `pairmeta.txt`); Zenodo mirrors.
- **Reference:** Mooij et al. (2016), *JMLR*.

---

## At a glance

| # | Dataset | Task | Ground truth | Rows × features | Target |
|---|---------|------|--------------|-----------------|--------|
| 1 | IHDP | Treatment effect | Simulated outcomes (known ITE/ATE) | 747 × 25 | Continuous |
| 2 | Twins | Treatment effect | Both potential outcomes observed | 11,400 × 39 | Binary |
| 3 | Jobs / LaLonde | Treatment effect | Real RCT (known ATE) | ~614 × ~9 | Earnings / employment |
| 4 | Sachs | Causal discovery | Consensus DAG from interventions | 7,466 × 11 | 11-node graph |
| 5 | Tübingen pairs | Causal discovery | Annotated causal direction | 108 pairs | Direction (X→Y) |

**Honorable mention:** **ACIC 2016/2017/2018** data challenges — 77+ semi-synthetic datasets (~4,800 obs × ~55 covariates) with fully known DGPs; increasingly a standard CATE benchmark, hosted on Synapse rather than Kaggle/UCI.

**Fit note for this repo's DPG pipeline:** Datasets 1–3 target treatment-effect estimation (one causal treatment column + confounder/distractor covariates) — usable as single-CSV, single-target inputs, but `main.py` currently assumes classifiers with `StratifiedShuffleSplit`, so the continuous-outcome ones need a regressor branch. Datasets 4–5 are structure-discovery benchmarks whose ground truth is a graph/edge-direction rather than a feature-target CSV, so they suit scoring DPG's recovered causal *edges* rather than a plain classification run.

## Sources

- [Hill 2011 IHDP / benchmark discussion (Adversarial balancing, DMKD)](https://link.springer.com/article/10.1007/s10618-021-00759-3)
- [Causal inference benchmark datasets (IHDP/Jobs/Twins/News) — arXiv 2301.11351](https://arxiv.org/pdf/2301.11351)
- [Dynamic Inter-treatment Information Sharing (IHDP/Twins/Jobs) — arXiv 2305.15984](https://arxiv.org/html/2305.15984v3)
- [awesome-causality-data index (GitHub)](https://github.com/rguo12/awesome-causality-data)
- [ACIC 2016 description — arXiv 2401.02154](https://arxiv.org/pdf/2401.02154)
- [Sachs & Tübingen pairs as causal-discovery benchmarks — arXiv 2506.01361 (TimeGraph)](https://arxiv.org/html/2506.01361v1)
- [Tübingen cause-effect pairs, Mooij et al. — arXiv 2209.08579](https://arxiv.org/pdf/2209.08579)
- [Causal reasoning & LLMs (Tübingen pairs benchmark) — arXiv 2305.00050](https://arxiv.org/pdf/2305.00050)
