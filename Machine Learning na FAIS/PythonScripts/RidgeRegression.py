import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import KFold,cross_val_score
from math import fabs
import time

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


def gradient_descent(b, x, y, lr):
    batch = y.size
    b[0] = b[0] - lr * (((x.dot(b[1:]) + b[0]) - y).sum())
    b[1:] = b[1:] - lr * ((x.dot(b[1:]) + b[0]) - y).sum() * x.sum(axis=0) / batch


def gradient_descent_for_ridge(b, x, y, lr,alpha):
    batch = y.size
    b[0] = b[0] - lr * (((x.dot(b[1:]) + b[0]) - y).sum()+alpha*b[0]*2)
    b[1:] = b[1:] - lr * (((x.dot(b[1:]) + b[0]) - y).sum() * x.sum(axis=0)+alpha*b[1:]*2) / batch


def R2(b,x,y):
    rss = (((x.dot(b[1:]) + b[0]) - y)**2).sum()
    tss = ((y-y.mean())**2).sum()
    return (tss-rss)/tss


class Ridge_Regression():
    alpha = 1
    lr = 0.001
    beta = np.array([])

    def __init__(self, alpha=1, learning_rate=0.001):
        self.alpha = alpha
        self.lr = learning_rate

    def fit(self, xtrain, ytrain, batchsize,epoch):
        self.beta = np.ones(xtrain.shape[1] + 1)
        old = np.ones_like(self.beta)
        new = np.zeros_like(self.beta)
        batch = 0
        if  ytrain.size<batchsize:
            batchsize = ytrain.size
        epoch += int(ytrain.size/batchsize)
        for a in range(epoch):
        # while fabs((old - new).sum()) > 0.00001:
            if batchsize * (batch + 1) > ytrain.size:
                batch = 0
                perm = np.random.permutation(ytrain.size)
                xtrain=xtrain[perm]
                ytrain=ytrain[perm]
            batch_range = np.arange(batchsize * batch, batchsize * (batch + 1))

            # old = np.copy(self.beta)
            gradient_descent_for_ridge(self.beta, xtrain[batch_range], ytrain[batch_range], self.lr,self.alpha)
            # gradient_descent(self.beta, xtrain[batch_range], ytrain[batch_range], self.lr)
            # new = np.copy(self.beta)
            batch += 1

    def predict(self, xtest):
        return xtest.dot(self.beta[1:]) + self.beta[0]

    def R2_score(self,x,y):
        return R2(self.beta,x,y)




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

