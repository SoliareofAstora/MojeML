import numpy as np
import matplotlib.pyplot as plt

data = np.zeros((20000, 2))
for i in range(0, data.shape[0], 2):
    data[i] = [np.random.multivariate_normal([-2], [[2]]), 0]
    data[i + 1] = [np.random.multivariate_normal([2], [[2]]), 1]

plt.hist(data[0::2, 0], 200, density=True, alpha=0.7)
plt.hist(data[1::2, 0], 200, density=True, alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Histogram of two gaussians')
plt.axvline(0)
plt.grid(True)
plt.show()


shuffleIndex = np.random.permutation(20000)
x,y = data[shuffleIndex,0],data[shuffleIndex,1]

testSize = 15000

xtrain,xtest,ytrain,ytest= x[:testSize], x[testSize:], y[:testSize], y[testSize:]

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()
model.fit(xtrain[:,np.newaxis],ytrain)

predictions = model.predict(xtest[:,np.newaxis])
predictions.sum()

data = np.array((xtest,predictions))
data = data.transpose()
classA = data[np.where(data[:,1]==1)]
classB= data[np.where(data[:,1]==0)]
plt.hist(classA[:,0],100,alpha=0.7)
plt.hist(classB[:,0],100,alpha=0.7)
plt.show()

model.score(xtest[:,np.newaxis],ytest)
model.get_params()

