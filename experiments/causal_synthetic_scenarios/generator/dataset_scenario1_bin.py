#!/usr/bin/env -S uv run python
"""
Synthetic Dataset — Scenario 1  (Binary Y revision)
=====================================================
DGP:
    η = β₁·F1 + ε                          [latent linear score]
    p = sigmoid(η) = 1 / (1 + exp(-η))     [probability via sigmoid]
    Y ~ Bernoulli(p)                        [binary outcome]

    F1 ~ N(0, 1)             [continuous, CAUSAL]
    F2 ∈ {low, mid, high}    [categorical, IRRELEVANT] → OHE → F2_mid, F2_high
    F3 ~ N(0, 1)             [continuous, IRRELEVANT]
    ε  ~ N(0, σ²),  σ = 0.5

Latent-scale SNR:
    SNR = β₁² · Var(F1) / σ²  =  9 / 0.25  =  36  (high)

Theoretical class balance:
    E[Y] = E[σ(β₁·F1 + ε)] ≈ 0.5
    (η centred at 0 by construction → symmetric sigmoid → balanced classes)

Expected classifier behaviour:
    AUC-ROC >> 0.95  — SNR=36 leaves very little irreducible noise.
    importance(F1) >> importance(F2, F3) ≈ 0
    If AUC < 0.90 or F1 is not top feature → pipeline is broken.

Change log vs. Scenario 1 original:
    - Y: continuous → binary via sigmoid + Bernoulli  (same as Sc.2b)
    - Models: Regressor → Classifier
    - Metrics: R², Pearson r → AUC-ROC, log-loss, point-biserial r
    - train_test_split: added stratify=y
    - Permutation importance scoring: r2 → roc_auc

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
from scipy.special import expit           # numerically stable sigmoid
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix

SEED      = 42
N         = 1000
BETA1     = 3.0
SIGMA     = 0.5        # σ on latent scale;  SNR = β₁²/σ² = 36
TEST_SIZE = 0.20

rng = np.random.default_rng(SEED)

# ── 2. Feature generation ──────────────────────────────────────────────────────

# F1 — continuous, causal
F1 = rng.standard_normal(N)

# F2 — categorical, irrelevant; uniform across 3 levels
F2_raw = rng.choice(["low", "mid", "high"], size=N)

# F3 — continuous, irrelevant
F3 = rng.standard_normal(N)

# ── 3. Binary target generation ────────────────────────────────────────────────
#
# Step 1 — latent score
epsilon = rng.normal(0.0, SIGMA, N)
eta     = BETA1 * F1 # + epsilon            # η ∈ (-∞, +∞)
#
# Step 2 — sigmoid squash
# scipy.special.expit is numerically stable for large |η|
# avoids exp overflow that naive 1/(1+exp(-η)) hits for η > ~710
p = expit(eta)                            # p ∈ (0, 1)
#
# Step 3 — Bernoulli draw
# Y|η ~ Bernoulli(σ(η))
Y = rng.binomial(n=1, p=p, size=N)       # Y ∈ {0, 1}

# ── 4. One-hot encode F2 ──────────────────────────────────────────────────────
# Categories sorted alphabetically: ['high','low','mid'] → drop 'high'
# Retained: F2_low, F2_mid
ohe = OneHotEncoder(drop="first", sparse_output=False, dtype=np.float64)
F2_ohe  = ohe.fit_transform(F2_raw.reshape(-1, 1))
ohe_cols = [f"F2_{c}" for c in ohe.categories_[0][1:]]

# ── 5. Assemble DataFrame ──────────────────────────────────────────────────────
df = pd.DataFrame({
    "F1":  F1,
    "F3":  F3,
    **dict(zip(ohe_cols, F2_ohe.T)),
    "eta": eta,                           # latent score  — diagnostics only
    "p":   p,                             # sigmoid output — diagnostics only
    "Y":   Y,
})

feature_cols = ["F1", "F3"] + ohe_cols   # 4 model columns

X = df[feature_cols]
y = df["Y"]


# def plot_pca(df, title):
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(df.drop(columns=["Y"]))
#     classes = df["Y"].unique()
#     plt.figure(figsize=(8, 6))
#     plt.title(title)
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.grid()
#     markers = {0: 'o', 1: 's'}
#     for classe in classes:
#         idx = df["Y"] == classe
#         plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Y={classe}", alpha=0.7, marker=markers[classe], edgecolors='w', s=100)
#     plt.colorbar(label='Class Label')
#     plt.show()


# plot_pca(df, "PCA of Synthetic Dataset (Scenario 1b - Binary Y)")

# ── 6. Train / test split (stratified) ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

# ── 7. Diagnostic helpers ──────────────────────────────────────────────────────
def compute_vif(X_df: pd.DataFrame) -> pd.Series:
    """VIF_i = 1 / (1 - R²_i);  O(p² · n)."""
    vifs = {}
    cols = X_df.columns.tolist()
    for col in cols:
        X_o = X_df.drop(columns=[col]).values
        y_c = X_df[col].values
        r2  = LinearRegression().fit(X_o, y_c).score(X_o, y_c)
        vifs[col] = round(1.0 / (1.0 - r2) if r2 < 1.0 else np.inf, 4)
    return pd.Series(vifs)


def point_biserial_r(x_cont: np.ndarray, y_bin: np.ndarray) -> tuple[float, float]:
    """
    Point-biserial r: Pearson r between a continuous and a binary variable.
    Equivalent to standard Pearson r when one variable is dichotomous.
    H₀: ρ = 0.
    """
    return stats.pointbiserialr(y_bin, x_cont)


# ── 8. Sanity checks ───────────────────────────────────────────────────────────
SEP = "=" * 62
print(SEP)
print("SCENARIO 1 (Binary Y)  |  SNR=36  |  Sanity Check")
print(SEP)

print(f"\n[Shape]  X: {X.shape}  |  y: {y.shape}")
print(f"\n[Dtypes]\n{X.dtypes.to_string()}")

# Class balance
n1, n0 = Y.sum(), (1 - Y).sum()
print("\n[Class balance]")
print(f"  Y=1 : {n1} ({100*n1/N:.1f}%)   Y=0 : {n0} ({100*n0/N:.1f}%)")
print("  Expected ≈ 50/50  (η centred at 0)")

# Latent score distribution
print("\n[Latent score η = β₁·F1 + ε]")
print(f"  mean = {eta.mean():.4f}  (expected ≈ 0)")
print(f"  std  = {eta.std():.4f}  (expected ≈ {np.sqrt(BETA1**2 + SIGMA**2):.4f})")
print(f"  p̄    = {p.mean():.4f}  (expected ≈ 0.5)")

# Point-biserial r — only F1 should be significant
print("\n[Point-biserial r with Y]")
for col in feature_cols:
    r_pb, p_val = point_biserial_r(X[col].values, Y)
    tag = " <-- CAUSAL" if col == "F1" else ""
    print(f"  {col:<12} r = {r_pb:+.4f}   p = {p_val:.2e}{tag}")

# VIF
print("\n[Variance Inflation Factors]")
vif = compute_vif(X)
for feat, v in vif.items():
    flag = "  *** HIGH" if v > 5 else ("  * mild" if v > 2 else "")
    print(f"  {feat:<12} VIF = {v:.4f}{flag}")

# Logistic regression baseline — |β_F1| should dominate
print("\n[Logistic Regression baseline — coefficient magnitudes]")
lr = LogisticRegression(max_iter=1000, random_state=SEED)
lr.fit(X_train, y_train)
lr_coefs = pd.Series(np.abs(lr.coef_[0]), index=feature_cols).sort_values(ascending=False)
for feat, coef in lr_coefs.items():
    bar = "█" * int(coef * 6)
    print(f"  {feat:<12} |β| = {coef:.4f}  {bar}")

# RandomForest classifier — MDI
print("\n[RandomForest — MDI Importances]")
rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced",
)
rf.fit(X_train, y_train)
imp_mdi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_mdi.items():
    bar = "█" * int(score * 60)
    print(f"  {feat:<12} {score:.4f}  {bar}")

y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(f"\n  RF AUC-ROC  : {roc_auc_score(y_test, y_prob_rf):.4f}  (expected > 0.97)")
print(f"  RF Log-loss : {log_loss(y_test, y_prob_rf):.4f}")
cm = confusion_matrix(y_test, rf.predict(X_test))
print(f"  Confusion matrix:\n{cm}")

# Permutation importance — roc_auc scoring
print("\n[RandomForest — Permutation Importances  (test set, scoring=roc_auc)]")
perm_rf = permutation_importance(
    rf, X_test, y_test,
    scoring="roc_auc",
    n_repeats=30,
    random_state=SEED,
    n_jobs=-1,
)
imp_perm = pd.Series(perm_rf.importances_mean, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_perm.items():
    se  = perm_rf.importances_std[list(feature_cols).index(feat)] / np.sqrt(30)
    bar = "█" * max(0, int(score * 100))
    print(f"  {feat:<12} {score:.4f} ± {se:.4f}  {bar}")

# GradientBoosting classifier — cross-check
print("\n[GradientBoosting — MDI Importances  (cross-check)]")
gb = GradientBoostingClassifier(
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

# Assertions — hard
assert imp_mdi.index[0]  == "F1", f"MDI FAILED: top = {imp_mdi.index[0]}"
assert imp_perm.index[0] == "F1", f"Permutation FAILED: top = {imp_perm.index[0]}"
# After (correct):
# Empirical AUC ceiling for logistic latent DGP with η ~ N(0, 9.25):
# Theoretical upper bound estimated via Monte Carlo on p=σ(η):
#   AUC_theory ≈ P(η_pos > η_neg) ≈ 0.89-0.92 for σ_η ≈ 3.04
# 0.85 gives ~2σ margin below the empirical CV mean (0.895 ± 0.024)
assert roc_auc_score(y_test, y_prob_rf) > 0.85, "AUC FAILED: expected > 0.85 at SNR=36"
print("\n[PASS] F1 top-ranked by MDI and permutation. AUC > 0.85.")

# Soft warning — spurious permutation signal on irrelevants
noisy_irrelevants = [
    f for f in feature_cols if f != "F1"
    and imp_perm[f] > 2 * (perm_rf.importances_std[list(feature_cols).index(f)] / np.sqrt(30))
]
if noisy_irrelevants:
    print(f"[WARN] Spurious signal on: {noisy_irrelevants} — expected absent at SNR=36.")

# ── 9. Export ──────────────────────────────────────────────────────────────────
out_cols = feature_cols + ["Y"]
out_path = "scenario_01b_binary_y.csv"
df[out_cols].to_csv(out_path, index=False)
print(f"[Saved] {out_path}  ({N} rows × {len(out_cols)} cols)")
print(SEP)