#!/usr/bin/env -S uv run
"""
Synthetic Dataset — Scenario 5  (Non-linear additive, Binary Y)
================================================================
DGP:
    s = sin(F1) + exp(F2) + ε              [latent non-linear additive score]
    Y = 1{ s > median(s) }                  [binary outcome, balanced by construction]

    F1 ~ U(-2π, 2π)              [continuous, CAUSAL — periodic contribution]
    F2 ~ U(-1.5, 1.5)            [continuous, CAUSAL — exponential contribution]
    F3..F6 ~ N(0, 1)            [continuous, IRRELEVANT]
    F7 ∈ {A, B, C}             [categorical, IRRELEVANT] → OHE → F7_A, F7_B, F7_C
    F8 ∈ {X, Y, Z}             [categorical, IRRELEVANT] → OHE → F8_X, F8_Y, F8_Z
    ε  ~ N(0, σ²)              [zero-mean Gaussian, σ set from target SNR]

Signal-to-Noise Ratio (latent scale):
    SNR = Var(sin(F1) + exp(F2)) / σ²
    σ is derived empirically so that the realised SNR equals TARGET_SNR.

    Decomposition of the deterministic signal variance:
        Var(sin(F1)) ≈ 0.5          (F1 spans full periods → sin centred, var 1/2)
        Var(exp(F2)) ≈ 1.33         (F2 ~ U(-1.5,1.5))
        Var(signal)  ≈ 1.83

Design notes:
    - F1 drawn over [-2π, 2π] so the model sees multiple sine cycles; a single
      monotone segment would be indistinguishable from a linear effect.
    - F2 bounded to [-1.5, 1.5] so exp(F2) stays in [0.22, 4.48] — avoids a
      heavy right tail dominating the signal variance.
    - Binary label via the *median* of the latent score → exactly balanced
      classes, removing class-imbalance as a confounder for importance metrics.

Challenge for DPG / tree ensembles:
    Recover F1 and F2 as the causal pair when their effect is non-linear and
    additive, while {F3..F8} carry zero population information about Y. Tree
    predicates must tile the sine/exp surfaces with axis-aligned thresholds.

Dependencies:
    numpy>=1.26.4
    pandas>=2.2.2
    scikit-learn>=1.5.0
    scipy>=1.13.0
    matplotlib>=3.8.0
"""

# ── 1. Imports & config ────────────────────────────────────────────────────────
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED       = 42
N          = 1000
TARGET_SNR = 5.0          # moderate; σ derived empirically to hit this exactly
TEST_SIZE  = 0.20

rng = np.random.default_rng(SEED)

# ── 2. Feature generation ──────────────────────────────────────────────────────

# F1 — continuous, causal; uniform over ±2π exposes multiple sine periods
F1 = rng.uniform(-2 * np.pi, 2 * np.pi, size=N)

# F2 — continuous, causal; bounded range keeps exp(F2) well-conditioned
F2 = rng.uniform(-1.5, 1.5, size=N)

# F3..F6 — continuous, irrelevant
F3 = rng.standard_normal(N)
F4 = rng.standard_normal(N)
F5 = rng.standard_normal(N)
F6 = rng.standard_normal(N)

# F7, F8 — categorical, irrelevant (non-uniform marginals)
F7_raw = rng.choice(["A", "B", "C"], size=N, p=[0.40, 0.35, 0.25])
F8_raw = rng.choice(["X", "Y", "Z"], size=N, p=[0.30, 0.40, 0.30])

# ── 3. Target generation ───────────────────────────────────────────────────────
# Deterministic non-linear additive signal
signal = np.sin(F1) + np.exp(F2)

# Noise variance adjusted to control the signal-to-noise ratio:
#   σ² = Var(signal) / TARGET_SNR
sigma   = float(np.sqrt(signal.var() / TARGET_SNR))
epsilon = rng.normal(0.0, sigma, size=N)

latent_score = signal + epsilon                     # s = sin(F1) + exp(F2) + ε

# Binary label at the median → exactly balanced classes
threshold = np.median(latent_score)
Y = (latent_score > threshold).astype(int)

# ── 4. One-hot encoding ────────────────────────────────────────────────────────
# Categoricals kept with all levels (drop=None) — irrelevant features, and DPG
# consumes the explicit per-level indicators.
ohe_f7  = OneHotEncoder(sparse_output=False, dtype=np.float64)
F7_ohe  = ohe_f7.fit_transform(F7_raw.reshape(-1, 1))
f7_cols = [f"F7_{c}" for c in ohe_f7.categories_[0]]

ohe_f8  = OneHotEncoder(sparse_output=False, dtype=np.float64)
F8_ohe  = ohe_f8.fit_transform(F8_raw.reshape(-1, 1))
f8_cols = [f"F8_{c}" for c in ohe_f8.categories_[0]]

# ── 5. Assemble DataFrame ──────────────────────────────────────────────────────
df = pd.DataFrame({
    "F1": F1, "F2": F2, "F3": F3, "F4": F4, "F5": F5, "F6": F6,
    **dict(zip(f7_cols, F7_ohe.T)),
    **dict(zip(f8_cols, F8_ohe.T)),
    "Y": Y,
})

feature_cols = ["F1", "F2", "F3", "F4", "F5", "F6"] + f7_cols + f8_cols

X = df[feature_cols]
y = df["Y"]

# ── 6. Train / test split (stratified to preserve class balance) ───────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

# ── 7. Diagnostics ─────────────────────────────────────────────────────────────
SEP = "=" * 66
print(SEP)
print(f"SCENARIO 5 (Binary Y)  |  Y = sin(F1)+exp(F2)+ε  |  SNR={TARGET_SNR}")
print(SEP)

print(f"\n[Shape]  X: {X.shape}  |  y: {y.shape}")
print(f"\n[Dtypes]\n{X.dtypes.to_string()}")

# Noise calibration
realised_snr = signal.var() / sigma**2
print("\n[Signal / noise]")
print(f"  Var(signal) = {signal.var():.4f}")
print(f"  σ (noise)   = {sigma:.4f}")
print(f"  realised SNR = {realised_snr:.4f}  (target {TARGET_SNR})")

# Class balance
n1, n0 = int(Y.sum()), int((1 - Y).sum())
print("\n[Class balance]")
print(f"  Y=1 : {n1} ({100*n1/N:.1f}%)   Y=0 : {n0} ({100*n0/N:.1f}%)   (median split → ≈50/50)")

# Point-biserial r with Y (continuous features) — captures *linear* association.
# F1's effect is periodic → near-zero linear r despite being causal (expected).
print("\n[Point-biserial r with Y]  (linear screen — misses F1's periodicity)")
for col in ["F1", "F2", "F3", "F4", "F5", "F6"]:
    r_pb, p_val = stats.pointbiserialr(Y, X[col].values)
    tag = " <-- CAUSAL" if col in ("F1", "F2") else ""
    print(f"  {col:<8} r = {r_pb:+.4f}   p = {p_val:.2e}{tag}")

# RandomForest — MDI
print("\n[RandomForest — MDI Importances]")
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    random_state=SEED, n_jobs=-1, class_weight="balanced",
)
rf.fit(X_train, y_train)
imp_mdi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_mdi.items():
    bar = "█" * int(score * 60)
    print(f"  {feat:<8} {score:.4f}  {bar}")

y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(f"\n  RF AUC-ROC  : {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"  RF Log-loss : {log_loss(y_test, y_prob_rf):.4f}")

# Permutation importance — scored on AUC-ROC (robust to non-linearity)
print("\n[RandomForest — Permutation Importances  (test set, scoring=roc_auc)]")
perm_rf = permutation_importance(
    rf, X_test, y_test, scoring="roc_auc",
    n_repeats=30, random_state=SEED, n_jobs=-1,
)
imp_perm = pd.Series(perm_rf.importances_mean, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_perm.items():
    se  = perm_rf.importances_std[list(feature_cols).index(feat)] / np.sqrt(30)
    bar = "█" * max(0, int(score * 200))
    print(f"  {feat:<8} {score:.4f} ± {se:.4f}  {bar}")

# GradientBoosting — cross-check
print("\n[GradientBoosting — MDI Importances  (cross-check)]")
gb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, random_state=SEED,
)
gb.fit(X_train, y_train)
imp_gb = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_gb.items():
    bar = "█" * int(score * 60)
    print(f"  {feat:<8} {score:.4f}  {bar}")

y_prob_gb = gb.predict_proba(X_test)[:, 1]
print(f"\n  GB AUC-ROC  : {roc_auc_score(y_test, y_prob_gb):.4f}")
print(f"  GB Log-loss : {log_loss(y_test, y_prob_gb):.4f}")

# Stratified 5-fold CV AUC
cv_auc = cross_val_score(
    RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    scoring="roc_auc", n_jobs=-1,
)
print("\n[Stratified 5-fold CV AUC — RandomForest]")
print(f"  folds : {np.round(cv_auc, 4)}")
print(f"  mean  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Assertions — the two causal features must top both ensembles' MDI ranking
top2_rf = set(imp_mdi.index[:2])
top2_gb = set(imp_gb.index[:2])
assert top2_rf == {"F1", "F2"}, f"RF MDI top-2 = {top2_rf}, expected {{F1, F2}}"
assert top2_gb == {"F1", "F2"}, f"GB MDI top-2 = {top2_gb}, expected {{F1, F2}}"
print("\n[PASS] F1 & F2 are the top-2 features by both RF and GB MDI.")

# ── 8. PCA visualisation (styled to match the paper figure) ────────────────────
# Project over the continuous features only. Standardised one-hot indicators
# would tile the plane into discrete categorical blocks and hide the signal
# geometry; the categoricals are causally inert, so the continuous subspace is
# the meaningful view. Standardise first — F1 (±2π) and exp(F2) otherwise dwarf
# the unit-variance features and collapse the projection onto F1.
pca_cols = ["F1", "F2", "F3", "F4", "F5", "F6"]
X_std    = StandardScaler().fit_transform(X[pca_cols].values)
pca      = PCA(n_components=2, random_state=SEED)
X_pca    = pca.fit_transform(X_std)
ev       = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(5.2, 4.6))
styles = {
    0: {"marker": "o", "color": "black",   "label": "0"},
    1: {"marker": "s", "color": "#E6A817", "label": "1"},   # goldenrod
}
for cls, st in styles.items():
    idx = (Y == cls)
    ax.scatter(
        X_pca[idx, 0], X_pca[idx, 1],
        marker=st["marker"], c=st["color"], label=st["label"],
        s=42, edgecolors="black", linewidths=0.4, alpha=0.95,
    )
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var.)", fontsize=12)
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var.)", fontsize=12)
leg = ax.legend(title="y", frameon=False, loc="upper left", fontsize=11)
leg.get_title().set_fontsize(11)
fig.tight_layout()

fig_dir  = Path(__file__).parent.parent / "docs"
fig_path = fig_dir / "scenario_5_pca.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"\n[Saved] {fig_path}  (PC1 {ev[0]*100:.1f}% / PC2 {ev[1]*100:.1f}%)")

# ── 9. Export ──────────────────────────────────────────────────────────────────
out_cols = feature_cols + ["Y"]
out_path = Path(__file__).parent.parent / "test_datasets" / "scenario_5.csv"
df[out_cols].to_csv(out_path, index=False)
print(f"[Saved] {out_path}  ({N} rows × {len(out_cols)} cols)")
print(SEP)
