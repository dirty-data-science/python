"""
=========================================
Machine learning with missing values
=========================================

Here we use simulated data to understanding the fundamentals of statistical
learning with missing values.

This notebook reveals why a HistGradientBoostingRegressor (
:class:`sklearn.ensemble.HistGradientBoostingRegressor` ) is a choice to
predict with missing values.

Simulations are very useful to control the missing-value mechanism, and
inspect it's impact on predictive models. In particular, standard
imputation procedures can reconstruct missing values with distortion only
if the data is *missing at random*.

The mathematical details behind this notebook can be found in
https://arxiv.org/abs/1902.06931
"""


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

import numpy as np

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

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 4) # Smaller default figure size

plt.figure()
X_full, y_full = generate_without_missing_values(1000)
plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full)
plt.colorbar(label='y')

# %%
# Missing completely at random settings
# ======================================
#
# We now consider missing completely at random settings (a special case
# of missing at random): the missingness is completely independent from
# the values.
#
# The missing-values mechanism
# -----------------------------

def generate_mcar(n_samples, missing_rate=.5, rng=42):
    X, y = generate_without_missing_values(n_samples, rng=rng)
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)

    M = rng.binomial(1, missing_rate, (n_samples, 2))
    np.putmask(X, M, np.nan)

    return X, y

# %%
# A quick plot to look at the data
X, y = generate_mcar(1000)

plt.figure()
plt.scatter(X_full[:, 0], X_full[:, 1], color='.8', ec='.5', label='All data')
plt.colorbar(label='y')
plt.scatter(X[:, 0], X[:, 1], c=y, label='Fully observed')
plt.legend()

# %%
# We can see that the distribution of the fully-observed data is the same
# than that of the original data
#
# Conditional Imputation with the IterativeImputer
# ------------------------------------------------
#
# As the data is MAR (missing at random), an imputer can use the
# conditional dependencies between the observed and the missing values to
# impute the missing values.
#
# We'll use the IterativeImputer, a good imputer, but it needs to be enabled
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute
iterative_imputer = impute.IterativeImputer()

# %%
# Let us try the imputer on the small data used to visualize
#
# **The imputation is learned by fitting the imputer on the data with
# missing values**
iterative_imputer.fit(X)

# %%
# **The data are imputed with the transform method**
X_imputed = iterative_imputer.transform(X)

# %%
# We can display the imputed data as our previous visualization
plt.figure()
plt.scatter(X_full[:, 0], X_full[:, 1], color='.8', ec='.5',
            label='All data', alpha=.5)
plt.scatter(X_imputed[:, 0], X_imputed[:, 1], c=y, marker='X',
            label='Imputed')
plt.colorbar(label='y')
plt.legend()

# %%
# We can see that the imputer did a fairly good job of recovering the
# data distribution
#
# Supervised learning with a linear model
# ----------------------------------------
#
# Given that the relationship between the fully-observed X and y is a
# linear relationship, it seems natural to use a linear model for
# prediction, which must be adapted to missing values using imputation.
#
# To use it in supervised setting, we will pipeline it with a linear
# model, using a ridge, which is a good default linear model
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV

iterative_and_ridge = make_pipeline(impute.IterativeImputer(), RidgeCV())

# %%
# We can evaluate the model performance in a cross-validation loop
# (for better evaluation accuracy, we increase slightly the number of
# folds to 7)
from sklearn import model_selection
scores_iterative_and_ridge = model_selection.cross_val_score(
    iterative_and_ridge, X, y, cv=8)

scores_iterative_and_ridge

# %%
# **Computational cost** One drawback of the IterativeImputer to keep in
# mind is that its computational cost can become prohibitive of large
# datasets (it has a bad computation scalability).

# %%
# Mean imputation: SimpleImputer
# -------------------------------
#
# We can try a simple imputer: imputation by the mean
mean_imputer = impute.SimpleImputer()

# %%
# A quick visualization reveals a larger disortion of the distribution
X_imputed = mean_imputer.fit_transform(X)
plt.figure()
plt.scatter(X_full[:, 0], X_full[:, 1], color='.8', ec='.5',
            label='All data', alpha=.5)
plt.scatter(X_imputed[:, 0], X_imputed[:, 1], c=y, marker='X',
            label='Imputed')
plt.colorbar(label='y')

# %%
# Evaluating in prediction pipeline
mean_and_ridge = make_pipeline(impute.SimpleImputer(), RidgeCV())
scores_mean_and_ridge = model_selection.cross_val_score(
    mean_and_ridge, X, y, cv=8)

scores_mean_and_ridge

# %%
# Supervised without imputation
# -----------------------------
#
# The HistGradientBoosting models are based on trees, which can be
# adapted to model directly missing values
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
score_hist_gradient_boosting = model_selection.cross_val_score(
    HistGradientBoostingRegressor(), X, y, cv=8)

score_hist_gradient_boosting

# %%
# Recap: which pipeline predicts well on our small data?
# .......................................................
#
# Let's plot the scores to see things better
import pandas as pd
import seaborn as sns

scores = pd.DataFrame({'Mean imputation + Ridge': scores_mean_and_ridge,
             'IterativeImputer + Ridge': scores_iterative_and_ridge,
             'HistGradientBoostingRegressor': score_hist_gradient_boosting,
    })

sns.boxplot(data=scores, orient='h')
plt.title('Prediction accuracy\n linear and small data\n'
          'Missing Completely at Random')
plt.tight_layout()


# %%
# Not much difference with the more sophisticated imputer. A more thorough
# analysis would be necessary, with more cross-validation runs.
#
# Prediction performance with large datasets
# -------------------------------------------
#
# Let us consider large datasets, to compare models in such regimes

X, y = generate_mcar(n_samples=20000)

scores_mean_and_ridge = model_selection.cross_val_score(
    mean_and_ridge, X, y, cv=8)
scores_mean_and_ridge

# %%
scores_iterative_and_ridge= model_selection.cross_val_score(
    iterative_and_ridge, X, y, cv=8)
scores_iterative_and_ridge

# %%
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

score_hist_gradient_boosting = model_selection.cross_val_score(
    HistGradientBoostingRegressor(), X, y, cv=8)
score_hist_gradient_boosting

# %%
import seaborn as sns
import pandas as pd

scores = pd.DataFrame({'Mean imputation + Ridge': scores_mean_and_ridge,
             'IterativeImputer + Ridge': scores_iterative_and_ridge,
             'HistGradientBoostingRegressor': score_hist_gradient_boosting,
    })

sns.boxplot(data=scores, orient='h')
plt.tight_layout()


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


