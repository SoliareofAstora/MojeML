import numpy as np

x = np.array([1, 1, 1])
cluster = np.array([1, 2, 3])


class PCM():
    m = 2
    centers = np.array([])

    def __init__(self, m=2):
        self.m = m

    def D(self, i, k):
        return np.sqrt(np.dot(i - k, i - k))

    def u(self, i, k):
        return (1 + (self.D(i, k) / self.eta(i)) ** (2 / self.m - 1)) ** -1

    def eta(self, i):
        return ().sum() / ().sum()