import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize']=12,10
import seaborn as sns

import itertools as it
import pandas as pd
import numpy as np
#import os

import warnings
warnings.filterwarnings('ignore')


from sklearn.linear_model import LinearRegression as Lin_Reg
from sklearn.linear_model import Ridge as Ridge_Reg
from sklearn.linear_model import Lasso as Lasso_Reg
from statsmodels.regression.linear_model import OLS
import sklearn.preprocessing as Preprocessing

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

import itertools as it
from itertools import combinations

import scipy as sp



# Load data
data = np.loadtxt('input/dataset_1.txt', delimiter=',', skiprows=1)

# Split predictors and response
x = data[:, :-1]
y = data[:, -1]

df = pd.DataFrame(data)

corr_matrix = np.corrcoef(x.T)

plt.pcolor(corr_matrix)
plt.title('Heatmap of correlation matrix')
plt.show()


# How many and, which predictors would you choose, to perform the regression?
# Lets look what the best subset selection algorithm will have to say about that
from statsmodels.regression.linear_model import OLS

### Best Subset Selection
min_bic = 1e10  # set some initial large value for min BIC score
best_subset = []  # best subset of predictors

# Create all possible subsets of the set of 10 predictors
predictor_set = set(range(10))  # predictor set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# Repeat for every possible size of subset
for size_k in range(10):
    # Create all possible subsets of size 'size',
    # using the 'combination' function from the 'itertools' library
    subsets_of_size_k = it.combinations(predictor_set, size_k + 1)

    max_r_squared = -1e10  # set some initial small value for max R^2 score
    best_k_subset = []  # best subset of predictors of size k

    # Iterate over all subsets of our predictor set
    for predictor_subset in subsets_of_size_k:
        # Use only a subset of predictors in the training data
        x_subset = x[:, predictor_subset]

        # Fit and evaluate R^2
        model = OLS(y, x_subset)
        results = model.fit()
        r_squared = results.rsquared

        # Update max R^2 and best predictor subset of size k
        # If current predictor subset has a higher R^2 score than that of the best subset
        # we've found so far, remember the current predictor subset as the best!
        if (r_squared > max_r_squared):
            max_r_squared = r_squared
            best_k_subset = predictor_subset[:]

    # Use only the best subset of size k for the predictors
    x_subset = x[:, best_k_subset]

    # Fit and evaluate BIC of the best subset of size k.
    # (We haven't discused BIC on the lecture, but its another measure of accuracy
    model = OLS(y, x_subset)
    results = model.fit()
    bic = results.bic

    # Update minimum BIC and best predictor subset
    # If current predictor has a lower BIC score than that of the best subset
    # we've found so far, remember the current predictor as the best!
    if (bic < min_bic):
        min_bic = bic
        best_subset = best_k_subset[:]

print('Best subset by exhaustive search:')
print(sorted(best_subset))

### Step-wise Forward Selection
d = x.shape[1]  # total no. of predictors

# Keep track of current set of chosen predictors, and the remaining set of predictors
current_predictors = []
remaining_predictors = list(range(d))

# Set some initial large value for min BIC score for all possible subsets
global_min_bic = 1e10

# Keep track of the best subset of predictors
best_subset = []

# Iterate over all possible subset sizes, 0 predictors to d predictors
for size in range(d):
    max_r_squared = -1e10  # set some initial small value for max R^2
    best_predictor = -1  # set some throwaway initial number for the best predictor to add
    bic_with_best_predictor = 1e10  # set some initial large value for BIC score

    # Iterate over all remaining predictors to find best predictor to add
    for i in remaining_predictors:
        # Make copy of current set of predictors
        temp = current_predictors[:]
        # Add predictor 'i'
        temp.append(i)

        # Use only a subset of predictors in the training data
        x_subset = x[:, temp]

        # Fit and evaluate R^2
        model = OLS(y, x_subset)
        results = model.fit()
        r_squared = results.rsquared

        # Check if we get a higher R^2 value than than current max R^2, if so, update
        if (r_squared > max_r_squared):
            max_r_squared = r_squared
            best_predictor = i
            bic_with_best_predictor = results.bic

    # Remove best predictor from remaining list, and add best predictor to current list
    remaining_predictors.remove(best_predictor)
    current_predictors.append(best_predictor)

    # Check if BIC for with the predictor we just added is lower than
    # the global minimum across all subset of predictors
    if (bic_with_best_predictor < global_min_bic):
        best_subset = current_predictors[:]
        global_min_bic = bic_with_best_predictor

print('Step-wise forward subset selection:')
print(sorted(best_subset))  # add 1 as indices start from 0

###  Step-wise Backward Selection
d = x.shape[1]  # total no. of predictors

# Keep track of current set of chosen predictors
current_predictors = list(range(d))

# First, fit and evaluate BIC using all 'd' number of predictors
model = OLS(y, x)
results = model.fit()
bic_all = results.bic

# Set the minimum BIC score, initially, to the BIC score using all 'd' predictors
global_min_bic = bic_all
# Keep track of the best subset of predictors
best_subset = []

# Iterate over all possible subset sizes, d predictors to 1 predictor
for size in range(d - 1, 1, -1):  # stop before 0 to avoid choosing an empty set of predictors
    max_r_squared = -1e10  # set some initial small value for max R^2
    worst_predictor = -1  # set some throwaway initial number for the worst predictor to remove
    bic_without_worst_predictor = 1e10  # set some initial large value for min BIC score

    # Iterate over current set of predictors (for potential elimination)
    for i in current_predictors:
        # Create copy of current predictors, and remove predictor 'i'
        temp = current_predictors[:]
        temp.remove(i)

        # Use only a subset of predictors in the training data
        x_subset = x[:, temp]

        # Fit and evaluate R^2
        model = OLS(y, x_subset)
        results = model.fit()
        r_squared = results.rsquared

        # Check if we get a higher R^2 value than than current max R^2, if so, update
        if (r_squared > max_r_squared):
            max_r_squared = r_squared
            worst_predictor = i
            bic_without_worst_predictor = results.bic

    # Remove worst predictor from current set of predictors
    current_predictors.remove(worst_predictor)

    # Check if BIC for the predictor we just removed is lower than
    # the global minimum across all subset of predictors
    if (bic_without_worst_predictor < global_min_bic):
        best_subset = current_predictors[:]
        global_min_bic = bic_without_worst_predictor

print('Step-wise backward subset selection:')
print(sorted(best_subset))



