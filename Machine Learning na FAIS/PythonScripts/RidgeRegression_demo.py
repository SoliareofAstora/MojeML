import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import KFold,cross_val_score
from math import fabs
import time
from PythonScripts.RidgeRegression import Ridge_Regression

np.random.seed(100)

data = pd.read_csv('input/winequality-white.csv')

data = data.sample(frac=1).reset_index(drop=True)

y = np.array(data['quality'])
x = np.array(data.drop(['quality'], axis=1))

# corr_matrix = np.corrcoef(x.T)
# plt.pcolor(corr_matrix)
# plt.title('Heatmap of correlation matrix')
# plt.show()

trainSize = int(data.shape[0] * 0.90)
xtrain, xtest, ytrain, ytest = x[:trainSize], x[trainSize:], y[:trainSize], y[trainSize:]

xtrain_scaled = preprocessing.scale(xtrain)
xtest_scaled = preprocessing.scale(xtest)

lambdas=np.arange(-4,5)
train_r_squared = np.zeros_like(lambdas,dtype=float)
test_r_squared = np.zeros_like(lambdas,dtype=float)

for ind, i in enumerate(lambdas):
    # Fit ridge regression on train set
    model = Ridge_Regression(alpha=i)
    model.fit(xtrain_scaled, ytrain, 100, 10000)

    # Evaluate train & test performance
    train_r_squared[ind] = model.R2_score(xtrain_scaled, ytrain)
    test_r_squared[ind] = model.R2_score(xtest_scaled, ytest)

# Plotting
plt.plot(lambdas, train_r_squared, 'bo-', label=r'$R^2$ Training set', color="darkblue", alpha=0.6, linewidth=3)
plt.plot(lambdas, test_r_squared, 'bo-', label=r'$R^2$ Test set', color="darkred", alpha=0.6, linewidth=3)
plt.xlabel('Lamda value'); plt.ylabel(r'$R^2$')
plt.xlim(lambdas[0], lambdas[-1])
plt.title(r'Evaluate ridge regression $R^2$ with different lamdas')
plt.legend(loc='best')
plt.grid()
plt.show()


# Nie wiem jak wpakować swój piękny model do cross_val_score :(
# lambdas=np.arange(-4,5)
# train_r_squared = np.zeros_like(lambdas,dtype=float)
# test_r_squared = np.zeros_like(lambdas,dtype=float)
# kfold = KFold(n_splits=5)
#
# for ind, i in enumerate(lambdas):
#     # Fit ridge regression on train set
#     model = Ridge_Regression(alpha=i)
#     model.fit(xtrain_scaled, ytrain, 100, 10000)
#
#     results = cross_val_score(model, x, y, cv=kfold, scoring="r2")
#     # Evaluate train & test performance
#     train_r_squared[ind] = results.mean()
#     test_r_squared[ind] = model.R2_score(xtest_scaled, ytest)
#
# # Plotting
# plt.plot(lambdas, train_r_squared, 'bo-', label=r'$R^2$ Training set', color="darkblue", alpha=0.6, linewidth=3)
# plt.plot(lambdas, test_r_squared, 'bo-', label=r'$R^2$ Test set', color="darkred", alpha=0.6, linewidth=3)
# plt.xlabel('Lamda value'); plt.ylabel(r'$R^2$')
# plt.xlim(lambdas[0], lambdas[-1])
# plt.title(r'Evaluate 5-fold cv with different lamdas')
# plt.legend(loc='best')
# plt.grid()
# plt.show()




model = Ridge_Regression(alpha=0)
model.fit(xtrain_scaled, ytrain, 100, 20000)
pred=model.predict(xtest_scaled)
# Evaluate train & test performance
print("My ridge regression")
print('meansquare {}'.format(mean_squared_error(ytest, pred)))


model = Ridge()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print("sklearn Ridge")
print('meansquare {}'.format(mean_squared_error(ytest,pred)))


model = LinearRegression()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print("sklearn lin reg")
print('meansquare {}'.format(mean_squared_error(ytest,pred)))


# My ridge regression
# meansquare 0.5044287795218303
# sklearn Ridge
# meansquare 0.5059882185272939
# sklearn lin reg
# meansquare 0.499199345749985
#
# Mój model RidgeRegression daje wynik niewiele lepszy od implementacji tego modelu w SKlearn.
# Różnica prawdopodobnie wynika z tego, że model z Sklearn działa na lambdzie = 1.
# Nadal LinearRegression jest lepszy, co moim zdaniem wynika ze specyfiki problemu którego się podjąłem.

