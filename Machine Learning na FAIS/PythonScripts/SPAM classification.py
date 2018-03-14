import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("input/spam.csv", encoding='latin-1')

# Drop columns and rename
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data = data.rename(columns={'v1': 'label', 'v2': 'text'})

data.label.value_counts()

data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(data['text'], data['label'], test_size=0.2, random_state=10)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
vect.fit(xtrain)
# gets all the words present in text
# np.array(vect.get_feature_names())

xtrainDF = vect.transform(xtrain)
xtestDF = vect.transform(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.naive_bayes import MultinomialNB
prediction = dict()
model = MultinomialNB()
model.fit(xtrainDF,ytrain)
prediction["multinomial"]=model.predict(xtestDF)
accuracy_score(ytest,prediction['multinomial'])
# 0.9883408071748879
testText = vect.transform(['just click this link'])
model.predict(testText)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrainDF,ytrain)
prediction['logistic'] = model.predict(xtestDF)
accuracy_score(ytest,prediction['logistic'])
# 0.9802690582959641

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(xtrainDF,ytrain)
prediction["knn"]=model.predict(xtestDF)
accuracy_score(ytest,prediction['knn'])
# 0.9273542600896861

from sklearn.model_selection import GridSearchCV
krange = np.arange(1,30)
paramGrid = dict(n_neighbors=krange)
model = KNeighborsClassifier()
grid = GridSearchCV(model,paramGrid)
grid.fit(xtrainDF,ytrain)

grid.best_params_
grid.best_score_
grid.cv_results_

print(classification_report(ytest, prediction['multinomial'], target_names = ["Ham", "Spam"]))

conf_mat = confusion_matrix(ytest, prediction['multinomial'])
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

pd.set_option('display.max_colwidth',-1)
xtest[ytest<prediction['multinomial']]

xtest[ytest>prediction['multinomial']]




