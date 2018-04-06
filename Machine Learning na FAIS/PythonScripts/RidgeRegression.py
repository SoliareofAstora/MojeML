import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_squared_error

data = pd.read_csv('input/winequality-white.csv')

data = data.sample(frac=1).reset_index(drop=True)

y=np.array(data['quality'])
x=np.array(data.drop(['quality'],axis=1))


trainSize = int(data.shape[0]*0.90)
xtrain, xtest, ytrain, ytest = x[:trainSize], x[trainSize:], y[:trainSize], y[trainSize:]

model = Ridge()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print("sklearn Ridge")
print('meansquare {}'.format(mean_squared_error(ytest,pred)))

import cv2
print(cv2.__version)


model = LinearRegression()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print("sklearn lin reg")
print('meansquare {}'.format(mean_squared_error(ytest,pred)))