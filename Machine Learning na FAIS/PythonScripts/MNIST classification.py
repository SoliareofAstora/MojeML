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


# import os

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()


# to make this notebook's output stable across runs
np.random.seed(42)

# Data Import
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

x, y = mnist['data'], mnist['target']
y = y.astype(int)

# data shuffle / train test split
shuffleIndex = np.random.permutation(70000)
x, y = x[shuffleIndex], y[shuffleIndex]
xtrain, xtest, ytrain, ytest = x[:50000], x[50000:], y[:50000], y[50000:]

# adress the problem of is it a 5 or not with Logistic regression from sklearn
from sklearn.linear_model import LogisticRegression

ytrain5 = (ytrain == 5)
ytest5 = (ytest == 5)

logreg5 = LogisticRegression(max_iter=5, random_state=42)
logreg5.fit(xtrain, ytrain5)

# model metrics
from sklearn.model_selection import cross_val_score

cross_val_score(logreg5, xtrain, ytrain5, cv=3, scoring='f1')
cross_val_score(logreg5, xtrain, ytrain5, cv=3, scoring="accuracy")

# Lets look at the confussion matrix and calculate some accuracy metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

ytrain5_pred = cross_val_predict(logreg5, xtrain, ytrain5, cv=3)
confMatrix = confusion_matrix(ytrain5, ytrain5_pred)

confMatrix_normalized = confMatrix.astype(float) / confMatrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(confMatrix_normalized)
plt.ylabel("True label")
plt.xlabel("Prediction Label")
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision score {}".format(precision_score(ytrain5, ytrain5_pred)))
print("Recall score {}".format(recall_score(ytrain5, ytrain5_pred)))
print("F1 score {}".format(f1_score(ytrain5, ytrain5_pred)))
# Now check of you can correctly calculate the two "by hand", from the precission matrix
# True positive rate

precision = confMatrix[1, 1] / confMatrix[:, 1].sum()
recall = confMatrix[1, 1] / confMatrix[1, :].sum()
f1 = 2 * confMatrix[1, 1] / (confMatrix[:, 1].sum() + confMatrix[1, :].sum())
print("My precision score {}".format(precision))
print("My recall score {}".format(recall))
print("My F1 score {}".format(f1))

digit = x[5020]
plot_digit(digit)
yscores = cross_val_predict(logreg5, xtrain, ytrain5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(ytrain5, yscores)


def plot_precision_recall_vs_threshold(prec, recal, thresh):
    plt.plot(thresh, prec[:-1], "b--", label="Precision")
    plt.plot(thresh, recal[:-1], "g-", label="Recall")
    plt.xlabel("threshold")
    plt.legend()
    plt.ylim(0, 1)


plot_precision_recall_vs_threshold(precision, recall, threshold)
plt.xlim(-6, 6)
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(ytrain5, yscores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()

# =======================
# Now lets repeat all this for teat data

ytest5_pred = cross_val_predict(logreg5,xtest,ytest5,cv=3)
confMatrix = confusion_matrix(ytest5, ytest5_pred)

confMatrix_normalized = confMatrix.astype(float) / confMatrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(confMatrix_normalized)
plt.ylabel("True label")
plt.xlabel("Prediction Label")
plt.show()

precision = confMatrix[1, 1] / confMatrix[:, 1].sum()
recall = confMatrix[1, 1] / confMatrix[1, :].sum()
f1 = 2 * confMatrix[1, 1] / (confMatrix[:, 1].sum() + confMatrix[1, :].sum())



print("My precision score {}".format(precision))
print("My recall score {}".format(recall))
print("My F1 score {}".format(f1))

yscores = cross_val_predict(logreg5,xtest,ytest5,cv=3,method="decision_function")
precision, recall, threshold = precision_recall_curve(ytest5, yscores)
plot_precision_recall_vs_threshold(precision, recall, threshold)
plt.xlim(-6, 6)
plt.show()
fpr, tpr, threshold = roc_curve(ytest5, yscores)
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()