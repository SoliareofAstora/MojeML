import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tables = [pd.read_csv("input/lol/game1.csv"),
          pd.read_csv("input/lol/game2.csv"),
          pd.read_csv("input/lol/game3.csv"),
          pd.read_csv("input/lol/game4.csv"),
          pd.read_csv("input/lol/game5.csv")]

data = pd.concat(tables)
lclicks = data.loc[data['Left'] == 1]
rclicks = data.loc[data['Right'] == 1]

plt.figure(figsize=[10, 10])
plt.axes().set_aspect(aspect='equal')

plt.scatter(rclicks['mouseX'], rclicks['mouseY'], s=0.5, alpha=0.1, edgecolors="red")
plt.scatter(lclicks['mouseX'], lclicks['mouseY'], s=0.5, alpha=0.1)
plt.plot(data['mouseX'], data['mouseY'], linewidth=0.01)
plt.xlim(-20, 1940)
plt.ylim(-20, 1100)
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=[10, 10])
plt.axes().set_aspect(aspect='equal')
plt.hist2d(rclicks['mouseX'], rclicks['mouseY'],50)
plt.title('Heatmap of right clicks')
plt.xlim(-20, 1940)
plt.ylim(-20, 1100)
plt.gca().invert_yaxis()
plt.show()


data.shape
