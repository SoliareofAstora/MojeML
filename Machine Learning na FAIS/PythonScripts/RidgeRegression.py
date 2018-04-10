import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from math import fabs
import time

np.random.seed(10)

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




class Ridge_Regression():
    alpha = 1
    lr = 0.001
    beta = np.array([])

    def __init__(self, alpha=1, learning_rate=0.001):
        self.alpha = alpha
        self.lr = learning_rate

    def fit(self, xtrain, ytrain, batchsize,epoch):
        self.beta = np.random.rand(xtrain.shape[1] + 1)
        old = np.ones_like(self.beta)
        new = np.zeros_like(self.beta)
        batch = 0
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


model = Ridge_Regression(alpha=1,learning_rate=0.001)
xtrain_scaled = preprocessing.scale(xtrain)

start = time.time()
model.fit(xtrain_scaled, ytrain, 100,10000)
end = time.time()
print(end-start)
#took 0.575 seconds to train
xtest_scaled = preprocessing.scale(xtest)
pred = model.predict(xtest_scaled)

print("My ridge regression")
print('meansquare {}'.format(mean_squared_error(ytest, pred))) #MSE 0.5941715624023438


model = Ridge()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print("sklearn Ridge")
print('meansquare {}'.format(mean_squared_error(ytest,pred))) #0.587126505288487


model = LinearRegression()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print("sklearn lin reg")
print('meansquare {}'.format(mean_squared_error(ytest,pred))) #0.571327137792465


