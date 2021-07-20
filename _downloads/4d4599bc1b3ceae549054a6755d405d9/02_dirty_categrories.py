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
using gradient boosted trees.

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
# Choosing columns
# -----------------
#
# For clean categorical columns, we will use one hot encoding to
# transform them:

clean_columns = {
    'gender': 'one-hot',
    'department_name': 'one-hot',
    'assignment_category': 'one-hot',
    'Year First Hired': 'numerical'}

# %%
# We then choose the categorical encoding methods we want to benchmark
# and the dirty categorical variable:

encoding_methods = ['one-hot', 'target', 'similarity', 'minhash',
                    'gap']
dirty_column = 'employee_position_title'


# %%
# The learning pipeline
# ----------------------------
# The encoders for both clean and dirty data are first imported:

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from dirty_cat import SimilarityEncoder, TargetEncoder, MinHashEncoder,\
    GapEncoder

# for scikit-learn 0.24 we need to require the experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingRegressor

encoders_dict = {
    'one-hot': OneHotEncoder(handle_unknown='ignore', sparse=False),
    'similarity': SimilarityEncoder(similarity='ngram'),
    'target': TargetEncoder(handle_unknown='ignore'),
    'minhash': MinHashEncoder(n_components=100),
    'gap': GapEncoder(n_components=100),
    'numerical': FunctionTransformer(None)}

# We then create a function that takes one key of our ``encoders_dict``,
# returns a pipeline object with the associated encoder,
# as well as a gradient-boosting regressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


def assemble_pipeline(encoding_method):
    # static transformers from the other columns
    transformers = [(enc + '_' + col, encoders_dict[enc], [col])
                    for col, enc in clean_columns.items()]
    # adding the encoded column
    transformers += [(encoding_method, encoders_dict[encoding_method],
                      [dirty_column])]
    pipeline = make_pipeline(
        # Use ColumnTransformer to combine the features
        ColumnTransformer(transformers=transformers, remainder='drop'),
        HistGradientBoostingRegressor()
    )
    return pipeline


# %%
# Comparing different encoding for supervised learning
# -----------------------------------------------------
# Eventually, we loop over the different encoding methods,
# instantiate each time a new pipeline, fit it
# and store the returned cross-validation score:

from sklearn.model_selection import cross_val_score
import numpy as np

all_scores = dict()

for method in encoding_methods:
    pipeline = assemble_pipeline(method)
    scores = cross_val_score(pipeline, df, y)
    print('{} encoding'.format(method))
    print('r2 score:  mean: {:.3f}; std: {:.3f}\n'.format(
        np.mean(scores), np.std(scores)))
    all_scores[method] = scores

# %%
# Plotting the results
# --------------------
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
# Automatic vectorization
# ========================
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
# Let's start again from the raw data, but this time well use an
# automated way of encoding the data
X = employee_salaries['data']
y = employee_salaries['target']

# %%
# We'll drop a few columns we don't want
X.drop(
    [
        'Current Annual Salary',  # Too linked with target
        'full_name',  # Not relevant to the analysis
        '2016_gross_pay_received',  # Too linked with target
        '2016_overtime_pay',  # Too linked with target
        'date_first_hired'  # Redundant with "year_first_hired"
    ],
    axis=1,
    inplace=True
)

# %%
# We have a fairly complex and heterogeneous dataframe:
X

# %%
# The challenge is to turn this dataframe into a form suited for
# machine learning.

# %%
# Using the SuperVectorizer in a supervised-learning pipeline
# ------------------------------------------------------------
#
# Assembling the |SV| in a pipeline with a powerful learner,
# such as gradient boosted trees, gives **a machine-learning method that
# can be readily applied to the dataframe**.

# The supervectorizer requires dirty_cat 0.2.0a1
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
# Inspecting the features created
# .................................
# Once it has been trained on data,
# we can print the transformers and the columns assignment it creates:

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
