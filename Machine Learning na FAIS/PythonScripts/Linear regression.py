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
# games08 = games.loc[games['release_year'] == 2008]
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
from sklearn.metrics import mean_squared_error


def true(X):
    return np.sin(15 * X)+X


np.random.seed(0)
samples = 300

xtrain = np.sort(np.random.rand(samples))
ytrain = true(xtrain) + np.random.randn(samples)*2

xval = np.linspace(0, 1, 1000)
yval = true(xval)

degrees = np.arange(0, 25)
scores = np.zeros_like(degrees,dtype=float)
plt.figure(figsize=(16, 16))
figsize=np.sqrt(degrees.size)

for i in range(degrees.size):
    ax = plt.subplot(figsize, figsize, i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynimial = PolynomialFeatures(degree=degrees[i], include_bias=True)
    linreg = LinearRegression()
    pipe = Pipeline([("polynomial", polynimial), ("linreg", linreg)])

    pipe.fit(xtrain[:, np.newaxis], ytrain)
    scores[i] = mean_squared_error(yval, pipe.predict(xval[:, np.newaxis]))

    plt.plot(xval, yval, label="True function")
    plt.xlabel("polynomial degree {}".format(degrees[i]))
    plt.scatter(xtrain, ytrain, edgecolor='b', s=20, label="Samples",alpha=0.1)
    plt.plot(xval, pipe.predict(xval[:, np.newaxis]), label="Model")
    print("{} / {}".format(i+1,degrees.size))
plt.show()

plt.plot(degrees, scores)
plt.xlabel('polynomial degree')
plt.ylabel('Mean Squared Error')
plt.title('polynomial degree vs mean square error')
plt.show()
