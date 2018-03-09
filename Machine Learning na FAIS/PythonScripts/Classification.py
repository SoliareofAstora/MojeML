# To support both python 2 and python 3
# from __future__ import division, print_function, unicode_literals

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import seaborn as sns

import pandas as pd
import numpy as np
#import os

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()

# to make this notebook's output stable across runs
np.random.seed(42)

#Data Import
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

x,y = mnist['data'],mnist['target']
y = y.astype(int)

# data shuffle / train test split
shuffleIndex = np.random.permutation(70000)
x,y = x[shuffleIndex],y[shuffleIndex]
xtrain,xtest,ytrain,ytest = x[:50000],x[50000:],y[:50000],y[50000:]


# adress the problem of is it a 5 or not with Logistic regression from sklearn
from sklearn.linear_model import LogisticRegression

ytrain5 = (ytrain == 5)
ytest5 = (ytest == 5)

logreg5 = LogisticRegression(max_iter= 5, random_state= 42)
logreg5.fit(xtrain,ytrain5)

#model metrics
from sklearn.model_selection import cross_val_score
cross_val_score(logreg5,xtrain,ytrain5,cv=3,scoring='f1')
cross_val_score(logreg5, xtrain, ytrain5, cv=3, scoring="accuracy")

# Lets look at the confussion matrix and calculate some accuracy metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

ytrain5_pred= cross_val_predict(logreg5,xtrain,ytrain5,cv=3)
confMatrix = confusion_matrix(ytrain5,ytrain5_pred)

confMatrix_normalized = confMatrix.astype(float)/confMatrix.sum(axis=1)[:,np.newaxis]
sns.heatmap(confMatrix_normalized)
plt.ylabel("True label")
plt.xlabel("Prediction Label")
plt.show()

from sklearn.metrics import precision_score,recall_score
precision_score(ytrain5,ytrain5_pred)
recall_score(ytrain5,ytrain5_pred)

