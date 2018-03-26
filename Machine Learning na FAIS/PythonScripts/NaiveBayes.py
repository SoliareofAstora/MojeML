import numpy as np
import pandas as pd

# https://www.kaggle.com/uciml/pima-indians-diabetes-database/data
data = pd.read_csv("input/diabetes.csv")

data = data.sample(frac=1).reset_index(drop=True)

y = data['Outcome']
x = data.drop('Outcome', 1)

trainSize = 600
xtrain, xtest, ytrain, ytest = x[:trainSize], x[trainSize:], y[:trainSize], y[trainSize:]


class NaiveBayes:

    def fit(self):
        return
    def predict(self):
        return