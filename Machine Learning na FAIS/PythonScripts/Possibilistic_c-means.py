import numpy as np

x = np.array([1,2,3,7,8,9])
K = 2

class PCM:
    centroids = np.array([])
    assignation_matrix = np.array([])

    def __init__(self, data_dimension, m=2, classes=1):
        self.m = m
        self.classes = classes
        self.centroids = np.random.rand(classes,data_dimension)

    def D(self, i, k):
        return np.sqrt(np.dot(i - k, i - k))

    def u(self, i, k):
        return (1 + (self.D(i, k) / self.eta(i)) ** (2 / self.m - 1)) ** -1

    def eta(self, i):
        return (().sum() / ().sum())**self.m

    def fit(self,x_train):
        self.assignation_matrix=0

    def predict(self,x_test):
        return 0


x = np.array([1,2,3,7,8,9])
centroids =np.array([2,8])

out = np.zeros((6,2))
out[:,0] = (x-centroids[0])**2
out[:,1] = (x-centroids[1])**2


