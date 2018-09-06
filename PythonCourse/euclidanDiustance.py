import numpy as np
import scipy.spatial

def distance(x):
    return scipy.spatial.distance.cdist(x,x,'euclidean')

x = np.random.rand(100,10)
distance(x)