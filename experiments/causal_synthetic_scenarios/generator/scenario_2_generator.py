import numpy as np
import pandas as pd
from pathlib import Path

"""
Synthetic Dataset — Scenario 2
================================
DGP:
    Y = F1 + ε

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

# ============================================================
# Scenario 2
# 5 features
# Causal feature: F1
#
# Mechanism:
#     Y <- F1 + ε
#
# Characteristics:
# - F1 is the only causal variable.
# - Moderate Gaussian noise.
# - Non-causal features exhibit weak correlations among themselves.
# - Correlated predictors may appear predictive due to finite sampling,
#   but they do not participate in the data-generating mechanism.
#
# Goal:
# Distinguish a true cause from correlated but non-causal variables.
# ============================================================

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# Number of samples
N = 1000

# ------------------------------------------------------------
# Feature generation
# ------------------------------------------------------------

# True causal feature
F1 = rng.normal(0, 1, N)

# Generate weakly correlated non-causal continuous features
#
# Covariance matrix produces pairwise correlations ≈ 0.20
#
cov = np.array([
    [1.0, 0.2, 0.2],
    [0.2, 1.0, 0.2],
    [0.2, 0.2, 1.0]
])

F2_F3_F4 = rng.multivariate_normal(
    mean=[0, 0, 0],
    cov=cov,
    size=N
)

F2 = F2_F3_F4[:, 0]
F3 = F2_F3_F4[:, 1]
F4 = F2_F3_F4[:, 2]

# Non-causal categorical feature
F5 = rng.choice(
    ["A", "B", "C"],
    size=N,
    p=[0.4, 0.35, 0.25]
)

# ------------------------------------------------------------
# Noise term
# ------------------------------------------------------------
# Moderate noise
#
# Signal variance ≈ Var(F1) = 1
# Noise variance = 0.5² = 0.25
#
epsilon = rng.normal(
    loc=0.0,
    scale=0.05,
    size=N
)

# ------------------------------------------------------------
# Target generation
# ------------------------------------------------------------
#
# Only F1 is causal
#
latent_score = F1 + epsilon

Y = (latent_score > 0).astype(int)

# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------
df = pd.DataFrame({
    "F1": F1,
    "F2": F2,
    "F3": F3,
    "F4": F4,
    "F5": F5,
    "Y": Y
})

# ------------------------------------------------------------
# One-hot encode categorical features
# ------------------------------------------------------------
X = pd.get_dummies(
    df.drop(columns=["Y"]),
    columns=["F5"],
    drop_first=False,
    dtype=int
)

# Final dataset
dataset = pd.concat(
    [X, df["Y"]],
    axis=1
)

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------
print("Dataset shape:", dataset.shape)

print("\nClass distribution:")
print(dataset["Y"].value_counts(normalize=True))

print("\nCorrelation matrix (continuous variables):")
print(
    pd.DataFrame({
        "F1": F1,
        "F2": F2,
        "F3": F3,
        "F4": F4
    }).corr()
)

print("\nColumns:")
print(dataset.columns.tolist())

print("\nFirst rows:")
print(dataset.head())

# Optional
parent_dir = Path(__file__).parent.parent
save_path = parent_dir / "test_datasets" / "scenario_2_updated.csv"
dataset.to_csv(str(save_path.resolve()), index=False)

# def plot_pca(df, title):
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(df.drop(columns=["Y"]))
#     classes = df["Y"].unique()
#     plt.figure(figsize=(8, 6))
#     plt.title(title)
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.grid()
#     markers = {0: "o", 1: "s"}
#     for classe in classes:
#         idx = df["Y"] == classe
#         plt.scatter(
#             X_pca[idx, 0],
#             X_pca[idx, 1],
#             label=f"Y={classe}",
#             alpha=0.7,
#             marker=markers[classe],
#             edgecolors="w",
#             s=100,
#         )
#     plt.colorbar(label="Class Label")
#     plt.show()


# plot_pca(dataset, "PCA of Synthetic Dataset (Scenario 2 - Binary Y)")