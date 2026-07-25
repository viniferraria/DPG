import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ============================================================
# Scenario 1
# 3 features
# Causal feature: F1
# Mechanism: Y <- F1 (linear, additive)
#
# Purpose:
# Sanity-check scenario where a single feature is the true cause
# and remaining features are irrelevant.
# ============================================================

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Number of samples
N = 1000

# ------------------------------------------------------------
# Feature generation
# ------------------------------------------------------------

# True causal feature
F1 = np.random.normal(loc=0.0, scale=1.0, size=N)

# Irrelevant continuous feature
F2 = np.random.normal(loc=0.0, scale=1.0, size=N)

# Irrelevant categorical feature
F3 = np.random.choice(["A", "B", "C"], size=N, p=[0.4, 0.3, 0.3])

# Target generation
# Linear additive mechanism:
#
# latent_score = F1 + ε
# Y = 1 if latent_score > 0 else 0
#
# Only F1 influences Y.
# ------------------------------------------------------------
latent_score = F1

Y = (latent_score > 0).astype(int)

# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------
df = pd.DataFrame({"F1": F1, "F2": F2, "F3": F3, "Y": Y})

# ------------------------------------------------------------
# One-hot encoding for categorical variables
# ------------------------------------------------------------
X = pd.get_dummies(df.drop(columns=["Y"]), columns=["F3"], drop_first=False, dtype=int)

# Final dataset
dataset = pd.concat([X, df["Y"]], axis=1)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("Dataset shape:", dataset.shape)
print("\nClass distribution:")
print(dataset["Y"].value_counts(normalize=True))

print("\nColumns:")
print(dataset.columns.tolist())

print("\nFirst rows:")
print(dataset.head())

# Optional: save dataset
dataset.to_csv("test_datasets/scenario_1.csv", index=False)

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


# plot_pca(dataset, "PCA of Synthetic Dataset (Scenario 1b - Binary Y)")