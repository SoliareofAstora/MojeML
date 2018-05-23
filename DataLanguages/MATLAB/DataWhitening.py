import numpy as np

matrixSize = 5
A = np.random.rand(matrixSize,matrixSize)
B = np.dot(A,A.T)

mi = np.random.rand(5)
sigma = B
x = np.random.multivariate_normal(mi,sigma,size=[10000])

z =np.divide(np.subtract(x,mi),sigma[:,np.newaxis])

print(z[1].mean(axis=0))
print(np.cov(z[1].T))