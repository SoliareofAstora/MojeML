import numpy as np
import tensorflow as tf

def cube_sum(x):
    return np.power(x,3).sum()

def cube_sum2(x):
    result = 0
    for i in range(len(x)):
        result += x[i] ** 3
    return result

def almost_variance(x):
     return np.divide(np.power(np.subtract(x,x.mean()),4).sum(),x.size)

def almost_variance2(x):
    m = sum(x) / len(x)
    result = 0
    for i in range(len(x)):
        result += (x[i] - m) ** 4
    result /= len(x)
    return result

import timeit

def test1():
    x = np.random.rand(10000)
    cube_sum(x)
    almost_variance(x)


print(timeit.timeit("test1()",setup="from __main__ import test1", number=1000))

def test2():
    x = np.random.rand(10000)
    cube_sum2(x)
    almost_variance2(x)


print(timeit.timeit("test2()",setup="from __main__ import test2", number=1000))
