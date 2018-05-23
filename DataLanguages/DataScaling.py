from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import numpy as np

X, y = load_svmlight_file('input/breast-cancer.txt')

y[y == 4] = 1
y[y == 2] = -1

min_max_scaler = preprocessing.maxabs_scale(X)