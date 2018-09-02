import numpy as np
from dtree import decisionTree
from bagger import bagger

class boost:
    def fit(self, x, y, nepoch=4, nTrees=30, maxLevel=7, bagVolume=50):
        trees = bagger()
        y[y==0]=-1
        trees.fit(x, y, nTrees, maxLevel, bagVolume)

        sampleWeights = np.ones_like(y) / len(y)
        treeWeights = np.ones_like(nTrees)
        predictionMatrix = (trees.predictionMatrix(x))

        # for i in range(nepoch):
        res = sampleWeights * predictionMatrix * y
        res[res > 0] = 0
        errorPerTree = -res.sum(1) + 10e-10
        treeWeights = 0.5 * np.log((1 - errorPerTree) / errorPerTree)
        sampleWeights = sampleWeights * np.power(np.e, -treeWeights.dot(predictionMatrix * y))
        sampleWeights = sampleWeights / sampleWeights.sum()

        self.treeWeights = treeWeights
        self.trees = trees

    def predict(self, x):
        pred = self.trees.predictionMatrix(x)
        result = self.treeWeights.dot(pred)
        return (result > 0).astype(int)


# Mathematics playground
# a = np.array([-1,-1,1,1,-1,1])
# b = np.array([[-1,-1,-1,1,-1,1],[-1,-1,1,1,-1,-1],[-1,1,-1,-1,1,-1]])
#
# sw = np.ones_like(a)/len(a)
# tw = np.ones(b.shape[0])
#
#
# res = sw*b*a
# res[res>0]=0
# errorPerTree = -res.sum(1)+10e-10
# tw = 0.5*np.log((1-errorPerTree)/errorPerTree)
# newsw = sw*np.power(np.e,-tw.dot(b*a))
# sw = newsw/newsw.sum()
#
# tw
#
# result = (tw.dot(b)>0).astype(int)
