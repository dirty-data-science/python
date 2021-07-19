"""
=========================================
Machine learning with missing values
=========================================

Here we use simulated data to understanding the fundamentals of statistical
learning with missing values.

This notebook reveals why a HistGradientBoostingRegressor (
:class:`sklearn.ensemble.HistGradientBoostingRegressor` ) is a choice to
predict with missing values.
"""

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 4) # Smaller default figure size

# %%
# The fully-observed data: a toy regression problem
# ==================================================
#
# We consider a simple regression problem where X (the data) is bivariate
# gaussian, and y (the prediction target)  is a linear function of the first
# coordinate, with noise.
#
# The data-generating mechanism
# ------------------------------

def generate_without_missing_values(n_samples, rng=42):
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)
    X = rng.multivariate_normal(mean, cov, size=n_samples)

    epsilon = 0.1 * rng.randn(n_samples)
    y = X[:, 0] + epsilon

    return X, y

# %%
# A quick plot reveals what the data looks like

plt.figure()
X_full, y_full = generate_without_missing_values(500)
plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full)
plt.colorbar(label='y')

# %%
# Missing completely at random setting
# ======================================
#
# We now consider missing completely at random settings (a special case
# of missing at random).
#
# The missing-values mechanism
# -----------------------------

def generate_mcar(n_samples, missing_rate=0.2, rng=42):
    X, y = generate_without_missing_values(n_samples, rng=rng)
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)

    M = rng.binomial(1, missing_rate, (n_samples, 2))
    np.putmask(X, M, np.nan)

    return X, y

# %%
# A quick plot to look at the data
X, y = generate_mcar(500, missing_rate=.5)

plt.figure()
plt.scatter(X_full[:, 0], X_full[:, 1], color='.8', ec='.5',
            label='All data')
plt.colorbar(label='y')
plt.scatter(X[:, 0], X[:, 1], c=y, label='Fully observed')
plt.legend()

# %%
# We can see that the distribution of the fully-observed data is the same
# than that of the original data
#
# Building a predictive model: imputation and simple model
# --------------------------------------------------------
#
# Given that the relationship between the fully-observed X and y is a
# linear relationship, it seems natural to use a linear model for
# prediction, which must be adapted to missing values using imputation.

from sklearn.linear_model import RidgeCV # Good default linear model
from sklearn.impute import IterativeImputive # Good imputer

# %%
# Missing not at random: censoring
# ======================================
#
# We now consider missing not at random settings, in particular
# self-masking or censoring, where large values are more likely to be
# missing.
#
# The missing-values mechanism
# -----------------------------

def generate_censored(n_samples, missing_rate=0.2, rng=42):
    X, y = generate_without_missing_values(n_samples, rng=rng)
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)

    B = np.random.binomial(1, 2 * missing_rate, (n_samples, 2))
    M = (X > 0.5) * B

    np.putmask(X, M, np.nan)

    return X, y

# %%
# A quick plot to look at the data
X, y = generate_censored(500, missing_rate=.4)

plt.figure()
plt.scatter(X_full[:, 0], X_full[:, 1], color='.8', ec='.5',
            label='All data')
plt.colorbar(label='y')
plt.scatter(X[:, 0], X[:, 1], c=y, label='Fully observed')
plt.legend()

# %%
# Here the full-observed data does not reflect well at all the
# distribution of all the data

# %%
# Using a predictor for the fully-observed case
# ==============================================
#
# Let us go back to the "easy" case of the missing completely at random
# settings with plenty of data
n_samples = 10000

X, y = generate_mcar(n_samples, missing_rate=.5)

# %%
# Suppose we have been able to train a predictive model that works on
# fully-observed data:

X_full, y_full = generate_without_missing_values(n_samples)


