import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell

# games = pd.read_csv("input/ign.csv")
# games.drop(['Unnamed: 0', 'url'], axis=1, inplace=True)
#
# games.drop(games.index[516], inplace=True)
#
# games08 = games#.loc[games['release_year'] == 2008]
# games08 = games08.round({'score':0})
#
# x = games08.groupby('score')['title'].count().index.values
# y = games08.groupby('score')['title'].count().values
#
# plt.bar(x, y)
# plt.title('Games by score in 2008')
# plt.ylabel('Amount');
# plt.xlabel('Score');
#
# plt.hlines(y.mean(),0,11)
# plt.xlim(0,11)
# plt.show()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def true(X):
    return np.sin(5*X)

np.random.seed(0)
samples = 30

xtrain=np.sort(np.random .rand(samples))
ytrain = true(xtrain) + np.random.randn(samples)*0.01

xval= np.linspace(0,1,100)
yval = true(xval)


degrees = np.arange(0,16)

plt.figure(figsize=(10,10))
for i in range(degrees.size):
    ax = plt.subplot(4,4,i+1)
    plt.setp(ax, xticks=(), yticks=())

    polynimial = PolynomialFeatures(degree=degrees[i],include_bias=True)
    linreg = LinearRegression()
    pipe = Pipeline([("polynomial", polynimial),("linreg", linreg)])

    pipe.fit(xtrain[:,np.newaxis],ytrain)

    score = cross_val_score(pipe,xtrain[:,np.newaxis],ytrain,scoring="neg_mean_squared_error", cv=10)

    plt.plot(xval, pipe.predict(xval[:, np.newaxis]), label="Model")
    plt.plot(xval, yval, label="True function")
    plt.xlabel("poly degree{}".format(degrees[i]))
    plt.scatter(xtrain, ytrain, edgecolor='b', s=20, label="Samples")
    # plt.legend(loc="best")
plt.show()







