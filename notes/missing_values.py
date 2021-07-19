"""
=========================================
Machine learning with missing values
=========================================

Here we use simulated data to illustrate the fundamentals of statistical
learning with missing values.
"""

import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 4) # Smaller default figure size

# %%
# The fully-observed data: a toy regression problem
# ==================================================
#
# We consider a simple regression problem where X (the data) is bivariate
# gaussian, and y (the prediction target)  is a linear function of the first
# coordinate, with noise.

def generate_without_missing_values(n_samples):
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    X = np.random.multivariate_normal(mean, cov, size=n_samples)

    epsilon = 0.1 * np.random.randn(n_samples)
    y = X[:, 0] + epsilon

    return X, y

# %%
# A quick plot reveals what the data looks like

X, y = generate_without_missing_values(500)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.colorbar(label='y')

# %%
# Missing completely at random setting
# ======================================

X_full, y_full = X.copy(), y.copy()

