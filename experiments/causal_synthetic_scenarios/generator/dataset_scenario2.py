#!/usr/bin/env -S uv run

"""
Synthetic Dataset — Scenario 2
================================
DGP:
    Y = β₁·F1 + ε

    F1 ~ N(0, 1)                                    [continuous, CAUSAL]
    [F2, F3] ~ MVN(0, Σ),  ρ(F2,F3) = 0.30         [continuous, IRRELEVANT, weakly correlated]
    F4 ∈ {A, B, C}                                  [categorical, IRRELEVANT] → OHE → F4_B, F4_C
    F5 ∈ {low, high}, P(high|F4=A/B/C)={0.35,0.50,0.65}
                                                    [categorical, IRRELEVANT, weak dep. on F4]
                                                    → OHE → F5_low
    ε  ~ N(0, σ²),  σ = 1.5

Signal-to-Noise Ratio:
    SNR = β₁² · Var(F1) / σ²  =  9 / 2.25  =  4.0  (moderate)

Post-OHE feature matrix columns: [F1, F2, F3, F4_B, F4_C, F5_low]  → 6 cols from 5 raw features

Challenge:
    1. Moderate noise (SNR=4) vs. near-perfect signal in Sc.1 (SNR=36).
    2. F2–F3 Pearson ρ=0.30 → MDI bias: correlated splits inflate irrelevant importance.
    3. F4–F5 Cramér's V ≈ 0.245 → weak categorical dependence, zero path to Y.
    4. Population truth: {F2, F3, F4, F5} are causally inert.
       MDI will show mild inflation on F2/F3; permutation importance corrects this.

Dependencies:
    numpy>=1.26.4
    pandas>=2.2.2
    scikit-learn>=1.5.0
    scipy>=1.13.0
"""

# ── 1. Imports & config ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED      = 42
N         = 1000
BETA1     = 3.0
SIGMA     = 1.5        # σ;  SNR = BETA1² / SIGMA² = 4.0
RHO       = 0.30       # Pearson ρ(F2, F3)
TEST_SIZE = 0.20

rng = np.random.default_rng(SEED)

# ── 2. Feature generation ──────────────────────────────────────────────────────

# F1 — continuous, causal; independent of all other features by construction
F1 = rng.standard_normal(N)

# F2, F3 — continuous, irrelevant, weakly correlated via bivariate normal
# Σ = [[1, ρ], [ρ, 1]]
# Cholesky:  L = [[1, 0], [ρ, √(1-ρ²)]]
# Draw z ~ N(0,I), then [F2,F3] = L·z  →  exact ρ(F2,F3) = RHO in population
cov = np.array([[1.0, RHO],
                [RHO, 1.0]])
F2_F3 = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=N)
F2, F3 = F2_F3[:, 0], F2_F3[:, 1]

# F4 — categorical, irrelevant, 3 levels, uniform marginal
F4_raw = rng.choice(["A", "B", "C"], size=N)

# F5 — categorical, irrelevant, 2 levels, weak conditional dependence on F4
# P(high | F4=A) = 0.35,  P(high | F4=B) = 0.50,  P(high | F4=C) = 0.65
# Theoretical Cramér's V:
#   χ² ≈ 4 × [(N/3)(Δp)² / ((N/3·0.5)/N)] = 4 × [(N/3)(0.15²) / 0.5]
#   With Δp=0.15, N=1000 → χ² ≈ 60  →  V = √(60/1000) ≈ 0.245
p_high_map = {"A": 0.35, "B": 0.50, "C": 0.65}
p_high_vec = np.vectorize(p_high_map.get)(F4_raw).astype(float)
F5_raw = np.where(rng.uniform(size=N) < p_high_vec, "high", "low")

# ── 3. Target generation ───────────────────────────────────────────────────────
epsilon = rng.normal(0.0, SIGMA, N)
Y = BETA1 * F1 + epsilon   # only F1 enters the DGP

# ── 4. One-hot encoding ────────────────────────────────────────────────────────
# sklearn sorts categories alphabetically before applying drop='first'
# F4: ['A','B','C'] → drop 'A' → keep F4_B, F4_C
ohe_f4 = OneHotEncoder(drop="first", sparse_output=False, dtype=np.float64)
F4_ohe = ohe_f4.fit_transform(F4_raw.reshape(-1, 1))
f4_cols = [f"F4_{c}" for c in ohe_f4.categories_[0][1:]]   # ['B', 'C']

# F5: ['high','low'] → drop 'high' → keep F5_low
ohe_f5 = OneHotEncoder(drop="first", sparse_output=False, dtype=np.float64)
F5_ohe = ohe_f5.fit_transform(F5_raw.reshape(-1, 1))
f5_cols = [f"F5_{c}" for c in ohe_f5.categories_[0][1:]]   # ['low']

# ── 5. Assemble DataFrame ──────────────────────────────────────────────────────
df = pd.DataFrame({
    "F1": F1,
    "F2": F2,
    "F3": F3,
    **dict(zip(f4_cols, F4_ohe.T)),
    **dict(zip(f5_cols, F5_ohe.T)),
    "Y": Y,
})

feature_cols = ["F1", "F2", "F3"] + f4_cols + f5_cols   # 6 columns

X = df[feature_cols]
y = df["Y"]

# ── 6. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

# ── 7. Diagnostic helpers ──────────────────────────────────────────────────────
def compute_vif(X_df: pd.DataFrame) -> pd.Series:
    """
    Variance Inflation Factor:  VIF_i = 1 / (1 - R²_i)
    R²_i from regressing feature i on all remaining features.

    Interpretation:
        VIF = 1.0      → zero multicollinearity
        1.0 < VIF < 5  → acceptable
        5 ≤ VIF < 10   → moderate — investigate
        VIF ≥ 10       → severe — likely problematic for linear models

    Complexity: O(p² · n) — negligible for p < 200, n < 10⁶.
    Note: VIF is meaningful for continuous features; OHE columns
          will show mild inflation by construction (complementary dummies).
    """
    vifs = {}
    cols = X_df.columns.tolist()
    for col in cols:
        X_others = X_df.drop(columns=[col]).values
        y_col    = X_df[col].values
        r2 = LinearRegression().fit(X_others, y_col).score(X_others, y_col)
        vifs[col] = round(1.0 / (1.0 - r2) if r2 < 1.0 else np.inf, 4)
    return pd.Series(vifs)


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cramér's V — symmetric nominal association measure.
    V ∈ [0,1]:  < 0.10 negligible | 0.10–0.30 weak | 0.30–0.50 moderate | > 0.50 strong
    χ² with Yates correction disabled (correction=False) for consistency with population V.
    """
    ct   = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n    = int(ct.values.sum())
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def mdi_inflation_check(
    imp_series: pd.Series,
    causal_col: str = "F1",
    threshold: float = 0.05
) -> list[str]:
    """Flag non-causal features whose MDI exceeds `threshold`."""
    return [
        feat for feat in imp_series.index
        if feat != causal_col and imp_series[feat] > threshold
    ]


# ── 8. Sanity checks ───────────────────────────────────────────────────────────
SEP = "=" * 62
print(SEP)
print("SCENARIO 2  |  SNR=4.0  |  Moderate Noise  |  Weak Correlations")
print(SEP)

# Shape & dtypes
print(f"\n[Shape]  X: {X.shape}  |  y: {y.shape}")
print(f"\n[Dtypes]\n{X.dtypes.to_string()}")

# Target distribution
expected_std = np.sqrt(BETA1**2 + SIGMA**2)
print("\n[Target Y distribution]")
print(f"  mean = {Y.mean():.4f}   (expected ≈ 0)")
print(f"  std  = {Y.std():.4f}   (expected ≈ {expected_std:.4f}  =  √(β₁²+σ²))")

# Pearson r with Y — population truth: only F1 ≠ 0
expected_r_f1 = BETA1 / expected_std
print(f"\n[Pearson r with Y]  (expected r(F1,Y) = β₁/√(β₁²+σ²) = {expected_r_f1:.4f})")
for col in feature_cols:
    r, p = stats.pearsonr(X[col], y)
    tag = " <-- CAUSAL" if col == "F1" else ""
    print(f"  {col:<12} r = {r:+.4f}   p = {p:.2e}{tag}")

# Empirical inter-feature correlations
r_f2f3, p_f2f3 = stats.pearsonr(F2, F3)
print(f"\n[F2–F3 Pearson ρ]   empirical = {r_f2f3:.4f}   target = {RHO}   p = {p_f2f3:.2e}")

cv = cramers_v(F4_raw, F5_raw)
print(f"[F4–F5 Cramér's V]  empirical = {cv:.4f}   theoretical ≈ 0.245")

# VIF — quantifies multicollinearity in the feature matrix
print("\n[Variance Inflation Factors]")
print("  (1.0=no collinearity | >5=moderate | >10=severe)")
vif = compute_vif(X)
for feat, v in vif.items():
    flag = "  *** HIGH" if v > 5 else ("  * mild" if v > 2 else "")
    print(f"  {feat:<12} VIF = {v:.4f}{flag}")

# RandomForest — MDI
print("\n[RandomForest — MDI Importances]")
rf = RandomForestRegressor(
    n_estimators=300,
    max_features="sqrt",   # √p randomisation; reduces MDI bias vs. 'auto'
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
imp_mdi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_mdi.items():
    bar = "█" * int(score * 60)
    print(f"  {feat:<12} {score:.4f}  {bar}")

# Check for MDI inflation on non-causal features
inflated = mdi_inflation_check(imp_mdi)
if inflated:
    print(f"  [!] MDI inflation detected: {inflated}")
    print(f"      Expected — ρ(F2,F3)={RHO} introduces correlated splits.")

# RandomForest — Permutation importance (test set; less biased)
print("\n[RandomForest — Permutation Importances  (test set, n_repeats=30)]")
perm_rf = permutation_importance(
    rf, X_test, y_test, n_repeats=30, random_state=SEED, n_jobs=-1
)
imp_perm = pd.Series(perm_rf.importances_mean, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_perm.items():
    se = perm_rf.importances_std[list(feature_cols).index(feat)] / np.sqrt(30)
    bar = "█" * max(0, int(score * 15))
    print(f"  {feat:<12} {score:.4f} ± {se:.4f}  {bar}")

# GradientBoosting — MDI cross-check
# GBM fits sequentially on residuals → less susceptible to MDI correlation bias than RF
print("\n[GradientBoosting — MDI Importances  (cross-check)]")
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=SEED,
)
gb.fit(X_train, y_train)
imp_gb = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_gb.items():
    bar = "█" * int(score * 60)
    print(f"  {feat:<12} {score:.4f}  {bar}")

# Hard assertions — both metrics must rank F1 first
assert imp_mdi.index[0]  == "F1", f"MDI FAILED: top feature = {imp_mdi.index[0]}"
assert imp_perm.index[0] == "F1", f"Permutation FAILED: top feature = {imp_perm.index[0]}"
print("\n[PASS] F1 is top-ranked by both MDI and permutation importance.")

# Soft warning — non-zero permutation importance on irrelevants at 2-sigma
noisy_irrelevants = [
    f for f in feature_cols if f != "F1"
    and imp_perm[f] > 2 * (perm_rf.importances_std[list(feature_cols).index(f)] / np.sqrt(30))
]
if noisy_irrelevants:
    print(f"[WARN] Spurious permutation signal on: {noisy_irrelevants}")
    print("       Likely finite-sample noise — rerun with larger N to confirm.")

# ── 9. Export ──────────────────────────────────────────────────────────────────
out_path = "scenario_02_moderate_noise_weak_corr.csv"
df.to_csv(out_path, index=False)
print(f"\n[Saved] {out_path}  ({N} rows × {len(feature_cols) + 1} cols)")
print(SEP)