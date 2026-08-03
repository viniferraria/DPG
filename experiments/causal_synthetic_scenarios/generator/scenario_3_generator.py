import numpy as np
import pandas as pd

# ============================================================
# Scenario 3
# 5 features
# Causal features: F1, F2
#
# Mechanism:
#     Y <- 0.6*F1 + 0.4*F2 + ε
#
# Characteristics:
# - Two additive causal variables.
# - F1 has a stronger effect than F2.
# - Remaining features are non-causal.
# - Moderate noise.
#
# Goal:
# Evaluate whether DPG:
#   1. Recovers both causal features.
#   2. Preserves the relative importance:
#        F1 > F2.
#   3. Remains robust when tree predicates create
#      threshold-based partitions of continuous variables.
# ============================================================

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# Number of samples
N = 1000

# ------------------------------------------------------------
# Feature generation
# ------------------------------------------------------------

# Causal features
F1 = rng.normal(loc=0.0, scale=1.0, size=N)
F2 = rng.normal(loc=0.0, scale=1.0, size=N)

# Non-causal continuous features
F3 = rng.normal(loc=0.0, scale=1.0, size=N)
F4 = rng.normal(loc=0.0, scale=1.0, size=N)

# Non-causal categorical feature
F5 = rng.choice(
    ["A", "B", "C"],
    size=N,
    p=[0.4, 0.35, 0.25]
)

# ------------------------------------------------------------
# Noise term
# ------------------------------------------------------------
#
# Signal variance:
# Var(0.6F1 + 0.4F2)
# = 0.36 + 0.16
# = 0.52
#
# Moderate noise
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
# True causal mechanism:
#
# Y <- 0.6F1 + 0.4F2 + ε
#
latent_score = (
    0.6 * F1 +
    0.4 * F2 +
    epsilon
)

# Binary classification target
Y = (latent_score > 0).astype(int)

# ------------------------------------------------------------
# Build dataset
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
# One-hot encoding
# ------------------------------------------------------------
X = pd.get_dummies(
    df.drop(columns=["Y"]),
    columns=["F5"],
    drop_first=False,
    dtype=int
)

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

print("\nColumns:")
print(dataset.columns.tolist())

print("\nFirst rows:")
print(dataset.head())

# Optional
dataset.to_csv("test_datasets/scenario_3.csv", index=False)

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


# plot_pca(dataset, "PCA of Synthetic Dataset (Scenario 3 - Binary Y)")
