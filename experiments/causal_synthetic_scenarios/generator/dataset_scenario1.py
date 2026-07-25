#!/usr/bin/env -S uv run
"""
Synthetic Dataset — Scenario 1
================================
DGP:
    Y = β₁·F1 + ε

    F1 ~ N(0, 1)              [continuous, CAUSAL]
    F2 ∈ {low, mid, high}     [categorical, IRRELEVANT]  → OHE → F2_mid, F2_high
    F3 ~ N(0, 1)              [continuous, IRRELEVANT]
    ε  ~ N(0, σ²), σ = 0.5

Signal-to-Noise Ratio:
    SNR = β₁² · Var(F1) / σ²
        = 3² · 1 / 0.25
        = 36  (high → F1 should dominate any importance ranking)

Challenge:
    Sanity check. A tree ensemble must recover F1 as the sole driver.
    F2 and F3 carry zero population-level information about Y.

Dependencies:
    numpy==1.26.4
    pandas==2.2.2
    scikit-learn==1.5.0
    scipy==1.13.0
"""

# ── 1. Imports & config ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

SEED   = 42
N      = 1000
BETA1  = 3.0
SIGMA  = 0.5          # noise std;  SNR = (BETA1**2 * 1) / SIGMA**2 = 36
TEST_SIZE = 0.2

rng = np.random.default_rng(SEED)

# ── 2. Feature generation ─────────────────────────────────────────────────────

# F1 — continuous, causal
F1 = rng.standard_normal(N)

# F2 — categorical, irrelevant; uniform across 3 levels
F2_raw = rng.choice(["low", "mid", "high"], size=N)

# F3 — continuous, irrelevant
F3 = rng.standard_normal(N)

# ── 3. Target generation ──────────────────────────────────────────────────────
epsilon = rng.normal(loc=0.0, scale=SIGMA, size=N)
Y = BETA1 * F1 + epsilon

# ── 4. One-hot encode F2 ──────────────────────────────────────────────────────
# drop='first' → removes F2_low; avoids the dummy variable trap.
# Remaining cols: F2_high, F2_mid  (sklearn sorts categories alphabetically)
ohe = OneHotEncoder(drop="first", sparse_output=False, dtype=np.float64)
F2_ohe = ohe.fit_transform(F2_raw.reshape(-1, 1))
ohe_cols = [f"F2_{cat}" for cat in ohe.categories_[0][1:]]  # skip dropped level

# ── 5. Assemble DataFrame ─────────────────────────────────────────────────────
df_ohe  = pd.DataFrame(F2_ohe, columns=ohe_cols)
df = pd.DataFrame({"F1": F1, "F3": F3})
df = pd.concat([df, df_ohe], axis=1)
df["Y"] = Y

feature_cols = ["F1", "F3"] + ohe_cols   # explicit column order

# ── 6. Train / test split ─────────────────────────────────────────────────────
X = df[feature_cols]
y = df["Y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

# ── 7. Sanity checks ──────────────────────────────────────────────────────────
print("=" * 55)
print("SCENARIO 1 — SANITY CHECK")
print("=" * 55)

print(f"\n[Shape]  X: {X.shape}  |  y: {y.shape}")
print(f"\n[Dtypes]\n{X.dtypes.to_string()}")

print(f"\n[Target Y]")
print(f"  mean = {Y.mean():.4f}  (expected ≈ 0)")
print(f"  std  = {Y.std():.4f}  (expected ≈ sqrt(BETA1²+σ²) = {np.sqrt(BETA1**2 + SIGMA**2):.4f})")

# Pearson r between Y and each feature (OHE cols should be ≈ 0)
print("\n[Pearson r with Y]")
for col in feature_cols:
    r, p = stats.pearsonr(X[col], y)
    print(f"  {col:<12} r = {r:+.4f}   p = {p:.2e}")

# Quick RF to validate importance ordering
print("\n[RandomForest feature importances — MDI]")
rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)

imp_mdi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_mdi.items():
    bar = "█" * int(score * 60)
    print(f"  {feat:<12} {score:.4f}  {bar}")

# Permutation importance on test set (less biased than MDI for categoricals)
print("\n[RandomForest feature importances — Permutation on test set]")
perm = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=SEED, n_jobs=-1)
imp_perm = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_perm.items():
    bar = "█" * max(0, int(score * 20))
    print(f"  {feat:<12} {score:.4f}  {bar}")

# Assert F1 is ranked #1 by both metrics
assert imp_mdi.index[0] == "F1",  "MDI FAILED: F1 not top feature"
assert imp_perm.index[0] == "F1", "Permutation FAILED: F1 not top feature"
print("\n[PASS] F1 is the top-ranked feature by both MDI and permutation importance.")

# ── 8. Export ─────────────────────────────────────────────────────────────────
out_path = "scenario_01_sanity_check.csv"
df[["F1", "F3"] + ohe_cols + ["Y"]].to_csv(out_path, index=False)
print(f"\n[Saved] {out_path}  ({N} rows x {len(feature_cols)+1} cols)")
print("=" * 55)

