import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# change to "last" to go back to previous settings

games = pd.read_csv("LinearRegression/ign.csv")
games.drop(['Unnamed: 0', 'url'], axis=1, inplace=True)

games.drop(games.index[516], inplace=True)

games08 = games#.loc[games['release_year'] == 2008]
games08 = games08.round({'score':0})

x = games08.groupby('score')['title'].count().index.values
y = games08.groupby('score')['title'].count().values

plt.bar(x, y)
plt.title('Games by score in 2008')
plt.ylabel('Amount');
plt.xlabel('Score');

plt.hlines(y.mean(),0,11)
plt.xlim(0,11)
plt.show()




