import numpy as np
import pandas as pd
from PythonScripts.NaiveBayes import NaiveBayes

# https://www.kaggle.com/uciml/pima-indians-diabetes-database/data
data = pd.read_csv("input/diabetes.csv")

data = data.sample(frac=1).reset_index(drop=True)

y = np.array(data['Outcome'])
x = np.array(data.drop('Outcome', 1))

trainSize = 400
xtrain, xtest, ytrain, ytest = x[:trainSize], x[trainSize:], y[:trainSize], y[trainSize:]

model = NaiveBayes()
model.fit(xtrain, ytrain)
mypred = model.predict(xtest)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

conf_mat = confusion_matrix(ytest, mypred)
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('My NaiveBayes')
plt.show()

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, precision_score

naive = GaussianNB()
logic = LogisticRegression()

naive.fit(xtrain, ytrain)
logic.fit(xtrain, ytrain)

naivepred = naive.predict(xtest)
logicpred = logic.predict(xtest)

conf_mat = confusion_matrix(ytest, naivepred)
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('sklearn GaussianNB')
plt.show()

conf_mat = confusion_matrix(ytest, logicpred)
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('skleanr logic regresison')
plt.show()

print("My naive bayes model")
print('recall {}'.format(recall_score(ytest, mypred)))
print('accuary {}'.format(accuracy_score(ytest, mypred)))
print('precision {}'.format(precision_score(ytest, mypred)))
print()

print("sklearn logic model")
print('recall {}'.format(recall_score(logicpred, mypred)))
print('accuary {}'.format(accuracy_score(logicpred, mypred)))
print('precision {}'.format(precision_score(logicpred, mypred)))
print()
print("sklearn gaussianNB model")
print('recall {}'.format(recall_score(naivepred, mypred)))
print('accuary {}'.format(accuracy_score(naivepred, mypred)))
print('precision {}'.format(precision_score(naivepred, mypred)))
