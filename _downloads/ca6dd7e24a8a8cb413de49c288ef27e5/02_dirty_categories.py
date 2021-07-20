"""
========================================================
Dirty categories: learning with non normalized entries
========================================================

Including strings that represent categories often calls for much data
preparation. In particular categories may appear with many morphological
variants, when they have been manually input, or assembled from diverse
sources.

Including such a column in a learning pipeline as a standard categorical
colum leads to categories with very high cardinalities and can loose
information on which categories are similar.

Here we look at a dataset on wages [#]_ where the column *Employee
Position Title* contains dirty categories.

.. [#] https://catalog.data.gov/dataset/employee-salaries-2016

We investigate encodings to include such compare different categorical
encodings for the dirty column to predict the *Current Annual Salary*,
using gradient boosted trees. For this purpose, we use the dirty-cat
library ( https://dirty-cat.github.io ).

"""

# %%
#
# The data
# ========
#
# Data Importing and preprocessing
# --------------------------------
#
# We first download the dataset:
from dirty_cat.datasets import fetch_employee_salaries
employee_salaries = fetch_employee_salaries()
print(employee_salaries['DESCR'])

# %%
# Then we load it:
import pandas as pd
df = employee_salaries['data'].copy()
df

# %%
# Now, let's carry out some basic preprocessing:
df['Date First Hired'] = pd.to_datetime(df['date_first_hired'])
df['Year First Hired'] = df['Date First Hired'].apply(lambda x: x.year)
# drop rows with NaN in gender
df.dropna(subset=['gender'], inplace=True)


# %%
# First we extract the target

target_column = 'Current Annual Salary'
y = df[target_column].values.ravel()

# %%
#
# Assembling a machine-learning pipeline that encodes the data
# =============================================================
#
# The learning pipeline
# ----------------------------
#
# To build a learning pipeline, we need to assemble encoders for each
# column, and apply a supvervised learning model on top.

# %%
# The categorical encoders
# ........................
#
# A encoder is needed to turn a categorical column into a numerical
# representation
from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)

# %%
# We assemble these to be applied on the relevant columns.
# The column transformer is created by specifying a set of transformers
# alongside with the column names on which to apply them

from sklearn.compose import make_column_transformer
encoder = make_column_transformer(
    (one_hot, ['gender', 'department_name', 'assignment_category']),
    ('passthrough', ['Year First Hired']),
    # Last but not least, our dirty column
    (one_hot, ['employee_position_title']),
    remainder='drop',
   )

# %%
# Pipelining an encoder with a learner
# ....................................
#
# We will use a HistGradientBoostingRegressor, which is a good predictor
# for data with heterogeneous columns
# (for scikit-learn 0.24 we need to require the experimental feature)
from sklearn.experimental import enable_hist_gradient_boosting
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingRegressor

# We then create a pipeline chaining our encoders to a learner
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(encoder, HistGradientBoostingRegressor())

# %%
# The pipeline can be readily applied to the dataframe for prediction
pipeline.fit(df, y)

# %%
# Dirty-category encoding
# -------------------------
#
# The one-hot encoder is actually not well suited to the 'Employee
# Position Title' column, as this columns contains 400 different entries.
#
# We will now experiments with encoders for dirty columns
from dirty_cat import SimilarityEncoder, TargetEncoder, MinHashEncoder,\
    GapEncoder

similarity = SimilarityEncoder(similarity='ngram')
target = TargetEncoder(handle_unknown='ignore')
minhash = MinHashEncoder(n_components=100)
gap = GapEncoder(n_components=100)

encoders = {
    'one-hot': one_hot,
    'similarity': SimilarityEncoder(similarity='ngram'),
    'target': target,
    'minhash': minhash,
    'gap': gap}

# %%
# We now loop over the different encoding methods,
# instantiate each time a new pipeline, fit it
# and store the returned cross-validation score:

from sklearn.model_selection import cross_val_score
import numpy as np

all_scores = dict()

for name, method in encoders.items():
    encoder = make_column_transformer(
        (one_hot, ['gender', 'department_name', 'assignment_category']),
        ('passthrough', ['Year First Hired']),
        # Last but not least, our dirty column
        (method, ['employee_position_title']),
        remainder='drop',
    )

    pipeline = make_pipeline(encoder, HistGradientBoostingRegressor())
    scores = cross_val_score(pipeline, df, y)
    print('{} encoding'.format(name))
    print('r2 score:  mean: {:.3f}; std: {:.3f}\n'.format(
        np.mean(scores), np.std(scores)))
    all_scores[name] = scores

# %%
# Plotting the results
# .....................
# Finally, we plot the scores on a boxplot:

import seaborn
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
ax = seaborn.boxplot(data=pd.DataFrame(all_scores), orient='h')
plt.ylabel('Encoding', size=20)
plt.xlabel('Prediction accuracy     ', size=20)
plt.yticks(size=20)
plt.tight_layout()

# %%
# The clear trend is that encoders that use the string form
# of the category (similarity, minhash, and gap) perform better than
# those that discard it.
# 
# SimilarityEncoder is the best performer, but it is less scalable on big
# data than MinHashEncoder and GapEncoder. The most scalable encoder is
# the MinHashEncoder. GapEncoder, on the other hand, has the benefit that
# it provides interpretable features (see :ref:`sphx_glr_auto_examples_04_feature_interpretation_gap_encoder.py`)
#
# |
#

# %%
# An easier way: automatic vectorization
# =======================================
#
# .. |SV| replace::
#     :class:`~dirty_cat.SuperVectorizer`
#
# .. |OneHotEncoder| replace::
#     :class:`~sklearn.preprocessing.OneHotEncoder`
#
# .. |ColumnTransformer| replace::
#     :class:`~sklearn.compose.ColumnTransformer`
#
# .. |RandomForestRegressor| replace::
#     :class:`~sklearn.ensemble.RandomForestRegressor`
#
# .. |SE| replace:: :class:`~dirty_cat.SimilarityEncoder`
#
# .. |permutation importances| replace::
#     :func:`~sklearn.inspection.permutation_importance`
#
# The code to assemble the column transformer is a bit tedious. We will
# now explore a simpler, automated, way of encoding the data.
#
# Let's start again from the raw data:
X = employee_salaries['data'].copy()
y = employee_salaries['target']

# %%
# We'll drop a few columns we don't want
X.drop([
            'Current Annual Salary',  # Too linked with target
            'full_name',  # Not relevant to the analysis
            '2016_gross_pay_received',  # Too linked with target
            '2016_overtime_pay',  # Too linked with target
            'date_first_hired'  # Redundant with "year_first_hired"
        ], axis=1, inplace=True)

# %%
# We still have a complex and heterogeneous dataframe:
X

# %%
# The |SV| can to turn this dataframe into a form suited for
# machine learning.

# %%
# Using the SuperVectorizer in a supervised-learning pipeline
# ------------------------------------------------------------
#
# Assembling the |SV| in a pipeline with a powerful learner,
# such as gradient boosted trees, gives **a machine-learning method that
# can be readily applied to the dataframe**.

# The supervectorizer requires dirty_cat 0.2.0a1. If you have an older
# version, you can install the alpha release with
#
#   pip install -pre dirty_cat==0.2.0a1
#

from dirty_cat import SuperVectorizer

pipeline = make_pipeline(
    SuperVectorizer(auto_cast=True),
    HistGradientBoostingRegressor()
)

# %%
# Let's perform a cross-validation to see how well this model predicts

from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, X, y, scoring='r2')

import numpy as np
print(f'{scores=}')
print(f'mean={np.mean(scores)}')
print(f'std={np.std(scores)}')

# %%
# The prediction perform here is pretty much as good as above
# but the code here is much simpler as it does not involve specifying
# columns manually.

# %%
# Analyzing the features created
# -------------------------------
#
# Let us perform the same workflow, but without the `Pipeline`, so we can
# analyze its mechanisms along the way.
sup_vec = SuperVectorizer(auto_cast=True)

# %%
# We split the data between train and test, and transform them:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

X_train_enc = sup_vec.fit_transform(X_train, y_train)
X_test_enc = sup_vec.transform(X_test)

# %%
# The encoded data, X_train_enc and X_test_enc are numerical arrays:
X_train_enc

# %%
# They have more columns than the original dataframe, but not much more:
X_train_enc.shape

# %%
# Inspecting the features created
# .................................
#
# The |SV| assigns a transformer for each column. We can inspect this
# choice:
sup_vec.transformers_

# %%
# This is what is being passed to the |ColumnTransformer| under the hood.
# If you're familiar with how the later works, it should be very intuitive.
# We can notice it classified the columns "gender" and "assignment_category"
# as low cardinality string variables.
# A |OneHotEncoder| will be applied to these columns.
#
# The vectorizer actually makes the difference between string variables
# (data type ``object`` and ``string``) and categorical variables
# (data type ``category``).
#
# Next, we can have a look at the encoded feature names.
#
# Before encoding:
X.columns.to_list()

# %%
# After encoding (we only plot the first 8 feature names):
feature_names = sup_vec.get_feature_names()
feature_names[:8]

# %%
# As we can see, it created a new column for each unique value.
# This is because we used |SE| on the column "division",
# which was classified as a high cardinality string variable.
# (default values, see |SV|'s docstring).
#
# In total, we have reasonnable number of encoded columns.
len(feature_names)


# %%
# Feature importance in the statistical model
# ---------------------------------------------
#
# In this section, we will train a regressor, and plot the feature importances
#
# .. topic:: Note:
#
#    To minimize compute time, use the feature importances computed by the
#    |RandomForestRegressor|, but you should prefer |permutation importances|
#    instead (which are less subject to biases)
#
# First, let's train the |RandomForestRegressor|,

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train_enc, y_train)


# %%
# Retreiving the feature importances
importances = regressor.feature_importances_
std = np.std(
    [
        tree.feature_importances_
        for tree in regressor.estimators_
    ],
    axis=0
)
indices = np.argsort(importances)[::-1]

# %%
# Plotting the results:

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 9))
plt.title("Feature importances")
n = 20
n_indices = indices[:n]
labels = np.array(feature_names)[n_indices]
plt.barh(range(n), importances[n_indices], color="b", yerr=std[n_indices])
plt.yticks(range(n), labels, size=15)
plt.tight_layout(pad=1)
plt.show()

# %%
# We can deduce from this data that the three factors that define the
# most the salary are: being hired for a long time, being a manager, and
# having a permanent, full-time job :).
#
#
# .. topic:: The SuperVectorizer automates preprocessing
#
#   As this notebook demonstrates, many preprocessing steps can be
#   automated by the |SV|, and the resulting pipeline can still be
#   inspected, even with non-normalized entries.
#
