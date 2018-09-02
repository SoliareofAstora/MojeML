import numpy as np
from dtree import decisionTree
from bagger import bagger

class WeightedBagger:
    def fit(self, x, y, nTrees=30, maxLevel=7, bagVolume=50):
        trees = bagger()
        y[y==0]=-1
        trees.fit(x, y, nTrees, maxLevel, bagVolume)

        predictionMatrix = trees.predictionMatrix(x)
        self.treeWeights = (predictionMatrix * y).sum(axis=1)
        self.trees = trees

    def predict(self, x):
        pred = self.trees.predictionMatrix(x)
        result = self.treeWeights.dot(pred)
        return (result > 0).astype(int)