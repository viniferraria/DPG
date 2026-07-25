import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# ============================================================
# Scenario 5
# 8 features
# Causal features: F1, F2
#
# Mechanism:
#     Y <- sin(F1) + exp(F2) + ε
#
# Characteristics:
# - Non-linear additive causal mechanism.
# - F1 contributes through a periodic relationship.
# - F2 contributes through an exponential relationship.
# - Remaining features are non-causal.
# - Moderate Gaussian noise.
#
# Goal:
# Evaluate whether DPG can identify causal variables
# participating in non-linear relationships and distinguish
# them from irrelevant predictors.
# ============================================================

# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

# Number of samples
N = 1000

# ------------------------------------------------------------
# Feature generation
# ------------------------------------------------------------

# Causal feature
# Uniform distribution allows observing multiple sine cycles
F1 = rng.uniform(
    low=-2 * np.pi,
    high=2 * np.pi,
    size=N
)

# Causal feature
# Restricted range prevents extreme exponential values
F2 = rng.uniform(
    low=-1.5,
    high=1.5,
    size=N
)

# Non-causal continuous features
F3 = rng.normal(0, 1, N)
F4 = rng.normal(0, 1, N)
F5 = rng.normal(0, 1, N)
F6 = rng.normal(0, 1, N)

# Non-causal categorical features
F7 = rng.choice(
    ["A", "B", "C"],
    size=N,
    p=[0.4, 0.35, 0.25]
)

F8 = rng.choice(
    ["X", "Y", "Z"],
    size=N,
    p=[0.3, 0.4, 0.3]
)

# ------------------------------------------------------------
# Noise
# ------------------------------------------------------------
#
# Signal:
#   sin(F1) ∈ [-1, 1]
#   exp(F2) ∈ [0.22, 4.48]
#
# Moderate noise
#
epsilon = rng.normal(
    loc=0.0,
    scale=0.05,
    size=N
)
# epsilon = np.zeros_like(epsilon)  # Uncomment this line to remove noise
# ------------------------------------------------------------
# Target generation
# ------------------------------------------------------------
#
# True causal mechanism:
#
# Y* = sin(F1) + exp(F2) + ε
#
latent_score = (
    np.sin(F1) +
    np.exp(F2) +
    epsilon
)

# Binary classification target
#
# Median threshold creates approximately balanced classes.
#
# threshold = np.median(latent_score)
threshold = np.percentile(latent_score, 50)

Y = (latent_score > threshold).astype(int)

# ------------------------------------------------------------
# Build dataset
# ------------------------------------------------------------
df = pd.DataFrame({
    "F1": F1,
    "F2": F2,
    "F3": F3,
    "F4": F4,
    "F5": F5,
    "F6": F6,
    "F7": F7,
    "F8": F8,
    "Y": Y
})

# ------------------------------------------------------------
# One-hot encoding
# ------------------------------------------------------------
X = pd.get_dummies(
    df.drop(columns=["Y"]),
    columns=["F7", "F8"],
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
parent_dir = Path(__file__).parent.parent
save_path = parent_dir / "test_datasets" / "scenario_5.csv"
dataset.to_csv(str(save_path.resolve()), index=False)

def plot_pca(df, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df.drop(columns=["Y"]))
    classes = df["Y"].unique()
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel(
        f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)"
    )
    plt.ylabel(
        f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)"
    )
    plt.grid()
    markers = {0: "o", 1: "s"}
    for classe in classes:
        idx = df["Y"] == classe
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=f"Y={classe}",
            alpha=0.7,
            marker=markers[classe],
            edgecolors="w",
            s=100,
        )
    plt.colorbar(label="Class Label")
    plt.show()


plot_pca(dataset, "PCA of Synthetic Dataset (Scenario 5 - Binary Y)")
print("\nPCA plot generated s5uccessfully.")