from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from PythonScripts.MyPerceptron import MyPerceptron
import numpy as np

iris = load_iris()
X = iris.data[:, (2, 3)]  # use only petal length and petal width
y = (iris.target == 0).astype(np.int) # check only if it's an Iris-Setosa (1), or no (0)

model = Perceptron()
model.fit(X,y)

x = np.linspace(0,8)

f = -model.coef_[0,0]/model.coef_[0,1]*x - model.intercept_[0]/model.coef_[0,1]
plt.scatter(X[y==1,0],X[y==1,1])
plt.scatter(X[y==0,0],X[y==0,1])
plt.plot(x,f)

plt.ylim(0,3)
plt.xlim(0,8)
plt.show()



model = MyPerceptron()
model.fit(X,y,epoch=10)
model.predict(X)-y
x = np.linspace(0,8)

f = -model.coef[0]/model.coef[1]*x - model.intercept/model.coef[1]
plt.scatter(X[y==1,0],X[y==1,1])
plt.scatter(X[y==0,0],X[y==0,1])
plt.plot(x,f)

plt.ylim(0,3)
plt.xlim(0,8)
plt.show()
