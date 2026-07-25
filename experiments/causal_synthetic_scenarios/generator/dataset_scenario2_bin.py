"""
Synthetic Dataset — Scenario 2  (Binary Y revision)
======================================================
DGP:
    η = β₁·F1 + ε                          [latent linear score]
    p = sigmoid(η) = 1 / (1 + exp(-η))     [probability via sigmoid]
    Y ~ Bernoulli(p)                        [binary outcome]

    F1 ~ N(0, 1)                                    [continuous, CAUSAL]
    [F2, F3] ~ MVN(0, Σ),  ρ(F2,F3) = 0.30         [continuous, IRRELEVANT, weakly correlated]
    F4 ∈ {A, B, C}                                  [categorical, IRRELEVANT] → OHE → F4_B, F4_C
    F5 ∈ {low, high}, P(high|F4=A/B/C)={0.35,0.50,0.65}
                                                    [categorical, IRRELEVANT, weak dep. on F4]
                                                    → OHE → F5_low
    ε  ~ N(0, σ²),  σ = 1.5

Latent-scale SNR:
    SNR = β₁² · Var(F1) / σ²  =  9 / 2.25  =  4.0

Theoretical class balance:
    E[Y] = E[Bernoulli(σ(η))]
         = E[σ(β₁·F1 + ε)]
    For symmetric distributions centred at 0 → E[Y] ≈ 0.5  (balanced classes)

    Note: ε on the latent scale shifts η before squashing.
    High σ → η is noisy → p(η) ≈ 0.5 more often → harder classification.

Change log vs. Scenario 2 original:
    - Y: continuous → binary via sigmoid + Bernoulli
    - Models: Regressor → Classifier
    - Metrics: R², Pearson r → AUC-ROC, log-loss, point-biserial r
    - Permutation importance scored on roc_auc

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
from scipy.special import expit          # numerically stable sigmoid: 1/(1+exp(-x))
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix

SEED      = 42
N         = 1000
BETA1     = 3.0
SIGMA     = 1.5        # σ on latent scale;  SNR = β₁²/σ² = 4.0
RHO       = 0.30       # Pearson ρ(F2, F3)
TEST_SIZE = 0.20

rng = np.random.default_rng(SEED)

# ── 2. Feature generation ──────────────────────────────────────────────────────

# F1 — continuous, causal
F1 = rng.standard_normal(N)

# F2, F3 — continuous, irrelevant, weakly correlated
# [F2, F3] ~ MVN(0, [[1, ρ], [ρ, 1]])
cov  = np.array([[1.0, RHO], [RHO, 1.0]])
F2_F3 = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=N)
F2, F3 = F2_F3[:, 0], F2_F3[:, 1]

# F4 — categorical, irrelevant, 3 levels
F4_raw = rng.choice(["A", "B", "C"], size=N)

# F5 — categorical, irrelevant, weak conditional dependence on F4
p_high_map = {"A": 0.35, "B": 0.50, "C": 0.65}
p_high_vec = np.vectorize(p_high_map.get)(F4_raw).astype(float)
F5_raw     = np.where(rng.uniform(size=N) < p_high_vec, "high", "low")

# ── 3. Binary target generation ────────────────────────────────────────────────
#
# Step 1 — latent score (same as Sc.2 continuous Y)
epsilon = rng.normal(0.0, SIGMA, N)
eta     = BETA1 * F1 + epsilon          # η ∈ (-∞, +∞)
#
# Step 2 — sigmoid squash  (scipy.special.expit = numerically stable sigmoid)
# σ(η) = 1 / (1 + exp(-η))
# expit avoids overflow on large |η|; equivalent to torch.sigmoid()
p = expit(eta)                          # p ∈ (0, 1)
#
# Step 3 — Bernoulli draw
# Y|η ~ Bernoulli(σ(η))
# Equivalently: Y = 1{U < σ(η)},  U ~ Uniform(0,1)
Y = rng.binomial(n=1, p=p, size=N)     # Y ∈ {0, 1}

# ── 4. One-hot encoding ────────────────────────────────────────────────────────
ohe_f4  = OneHotEncoder(drop="first", sparse_output=False, dtype=np.float64)
F4_ohe  = ohe_f4.fit_transform(F4_raw.reshape(-1, 1))
f4_cols = [f"F4_{c}" for c in ohe_f4.categories_[0][1:]]    # F4_B, F4_C

ohe_f5  = OneHotEncoder(drop="first", sparse_output=False, dtype=np.float64)
F5_ohe  = ohe_f5.fit_transform(F5_raw.reshape(-1, 1))
f5_cols = [f"F5_{c}" for c in ohe_f5.categories_[0][1:]]    # F5_low

# ── 5. Assemble DataFrame ──────────────────────────────────────────────────────
df = pd.DataFrame({
    "F1": F1, "F2": F2, "F3": F3,
    **dict(zip(f4_cols, F4_ohe.T)),
    **dict(zip(f5_cols, F5_ohe.T)),
    "eta": eta, "p": p,              # latent score + probability (diagnostics only)
    "Y": Y,
})

feature_cols = ["F1", "F2", "F3"] + f4_cols + f5_cols   # 6 model features

X = df[feature_cols]
y = df["Y"]

# ── 6. Train / test split (stratified to preserve class balance) ───────────────
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
    Equivalent to Pearson r when one variable is dichotomous.
    H₀: r = 0  (no linear association with binary outcome)
    """
    return stats.pointbiserialr(y_bin, x_cont)


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    ct   = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n    = int(ct.values.sum())
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


# ── 8. Sanity checks ───────────────────────────────────────────────────────────
SEP = "=" * 65
print(SEP)
print("SCENARIO 2 (Binary Y)  |  SNR=4.0  |  Latent sigmoid DGP")
print(SEP)

print(f"\n[Shape]  X: {X.shape}  |  y: {y.shape}")
print(f"\n[Dtypes]\n{X.dtypes.to_string()}")

# Class balance
n1, n0 = Y.sum(), (1 - Y).sum()
print(f"\n[Class balance]")
print(f"  Y=1 : {n1} ({100*n1/N:.1f}%)   Y=0 : {n0} ({100*n0/N:.1f}%)")
print(f"  Expected ≈ 50/50  (symmetric η distribution centred at 0)")

# Latent score distribution
print(f"\n[Latent score η = β₁·F1 + ε]")
print(f"  mean  = {eta.mean():.4f}  (expected ≈ 0)")
print(f"  std   = {eta.std():.4f}  (expected ≈ √(β₁²+σ²) = {np.sqrt(BETA1**2+SIGMA**2):.4f})")
print(f"  p̄     = {p.mean():.4f}  (expected ≈ 0.5)")

# Point-biserial r with Y — population truth: only F1 ≠ 0
print(f"\n[Point-biserial r with Y]")
for col in feature_cols:
    r_pb, p_val = point_biserial_r(X[col].values, Y)
    tag = " <-- CAUSAL" if col == "F1" else ""
    print(f"  {col:<12} r = {r_pb:+.4f}   p = {p_val:.2e}{tag}")

# Empirical inter-feature correlations
r_f2f3, p_f2f3 = stats.pearsonr(F2, F3)
cv = cramers_v(F4_raw, F5_raw)
print(f"\n[F2–F3 Pearson ρ]   empirical = {r_f2f3:.4f}   target = {RHO}   p = {p_f2f3:.2e}")
print(f"[F4–F5 Cramér's V]  empirical = {cv:.4f}   theoretical ≈ 0.245")

# VIF
print("\n[Variance Inflation Factors]")
vif = compute_vif(X)
for feat, v in vif.items():
    flag = "  *** HIGH" if v > 5 else ("  * mild" if v > 2 else "")
    print(f"  {feat:<12} VIF = {v:.4f}{flag}")

# Logistic regression baseline (should isolate F1)
print("\n[Logistic Regression baseline — coefficient magnitudes]")
lr = LogisticRegression(max_iter=1000, random_state=SEED)
lr.fit(X_train, y_train)
lr_coefs = pd.Series(np.abs(lr.coef_[0]), index=feature_cols).sort_values(ascending=False)
for feat, coef in lr_coefs.items():
    bar = "█" * int(coef * 10)
    print(f"  {feat:<12} |β| = {coef:.4f}  {bar}")

# RandomForest classifier — MDI
print("\n[RandomForest — MDI Importances]")
rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced",    # guards against any residual class imbalance
)
rf.fit(X_train, y_train)
imp_mdi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, score in imp_mdi.items():
    bar = "█" * int(score * 60)
    print(f"  {feat:<12} {score:.4f}  {bar}")

# RF metrics
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(f"\n  RF AUC-ROC  : {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"  RF Log-loss : {log_loss(y_test, y_prob_rf):.4f}")

# Permutation importance — scored on AUC-ROC (more reliable than accuracy for binary)
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
    bar = "█" * max(0, int(score * 200))
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

# Stratified 5-fold CV AUC on full dataset
cv_auc = cross_val_score(
    RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    scoring="roc_auc", n_jobs=-1,
)
print(f"\n[Stratified 5-fold CV AUC — RandomForest]")
print(f"  folds : {np.round(cv_auc, 4)}")
print(f"  mean  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Assertions
assert imp_mdi.index[0]  == "F1", f"MDI FAILED: top = {imp_mdi.index[0]}"
assert imp_perm.index[0] == "F1", f"Permutation FAILED: top = {imp_perm.index[0]}"
print("\n[PASS] F1 is top-ranked by both MDI and permutation importance.")

# ── 9. Export  (drop latent diagnostics cols from final CSV) ───────────────────
out_cols = feature_cols + ["Y"]
out_path = "scenario_02b_binary_y.csv"
df[out_cols].to_csv(out_path, index=False)
print(f"[Saved] {out_path}  ({N} rows × {len(out_cols)} cols)")
print(SEP)