from bootstrap import *
import xgboost as xgb

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.simplefilter("ignore")

np.random.seed(500)

data = pd.read_csv('data/heart.csv')
data = data.sample(frac=1)

y = data["target"]
x = data.drop(columns = ['target'])
x = np.array(x)
y = np.array(y)

trainSize = 100
print("total samples: %d \ttrain: %d\t test: %d" % (y.__len__(),trainSize,y.__len__()-trainSize))
xtrain, xtest = x[:trainSize],x[trainSize:]
ytrain, ytest = y[:trainSize],y[trainSize:]


print("the accuracy of models with the same \n")

tree = decisionTree()
tree.fit(xtrain,ytrain)
print(" Simple decision tree: \t%.4f"%accuracy_score(ytest,tree.predict(xtest)))

bagg = bagger()
bagg.fit(xtrain,ytrain)
print("Majority trees voting: \t%.4f"%accuracy_score(ytest,bagg.predict(xtest)))

xgboost = xgb.XGBClassifier()
xgboost.fit(xtrain,ytrain)
print("xgboost XGBClassifier: \t%.4f"%accuracy_score(ytest,xgboost.predict(xtest)))

sklearn = GradientBoostingClassifier()
sklearn.fit(xtrain,ytrain)
print("sklearn gradientboost: \t%.4f"%accuracy_score(ytest,sklearn.predict(xtest)))

forrest = boostedForrest()
forrest.fit(xtrain,ytrain)
print("\t  boosted forrest: \t%.4f"%accuracy_score(ytest,forrest.predict(xtest)))

print("bad score probably because i have got math mistake somewhere")

