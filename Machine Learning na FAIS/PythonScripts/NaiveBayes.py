import numpy as np
import pandas as pd
import math

class NaiveBayes:
    mean = np.array([])
    variance = np.array([])
    classes = np.array([])
    prediction=np.array([])

    def fit(self, x, y):
        classes = np.unique(y)
        mean = np.zeros((len(classes), x.shape[1]))
        variance = np.zeros((len(classes), x.shape[1]))

        for c in range(len(classes)):
            vector = x[np.where(y == classes[c])]
            mean[c] = vector.sum(axis=0) / vector.shape[0]
            variance[c] = (pow(mean[c] - vector, 2)).sum(axis=0) / vector.shape[0]
        self.classes=classes
        self.mean=mean
        self.variance=variance

    def predict(self, x):
        prediction = np.zeros((len(self.classes), x.shape[0]))
        for i in range(x.shape[0]):
            for c in range(len(self.classes)):
                prediction[c, i] =((1 / (np.sqrt(2 * math.pi * self.variance[c]))) *
                                    pow(math.e,-(pow(x[i] - self.mean[c], 2) /(2 * self.variance[c])))).sum()
        final = self.classes[np.argmax(prediction, axis=0)]
        self.prediction = prediction
        return final


# https://www.kaggle.com/uciml/pima-indians-diabetes-database/data
data = pd.read_csv("input/diabetes.csv")

data = data.sample(frac=1).reset_index(drop=True)

y = np.array(data['Outcome'])
x = np.array(data.drop('Outcome', 1))

trainSize = 50
xtrain, xtest, ytrain, ytest = x[:trainSize], x[trainSize:], y[:trainSize], y[trainSize:]

model = NaiveBayes()
model.fit(xtrain,ytrain)
pred = model.predict(ytest)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

conf_mat = confusion_matrix(ytest, pred)
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

ytest.sum()
pred.sum()

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

