from dtree import decisionTree
import numpy as np

class bagger:
    def fit(self, x, y, nTrees=25, maxLevel=7, bagVolume=50):
        rowSample = (np.random.rand(nTrees, bagVolume) * len(x)).astype(int)

        self.trees = []
        for i in range(nTrees):
            tree = decisionTree()
            featureCount = int(np.random.rand() * len(x[0] - 1)) + 1
            featureSample = np.random.choice(len(x[0]), featureCount, replace=True)
            featureSample = np.unique(featureSample)
            maxlvl = np.random.randint(0, maxLevel)
            a = x[rowSample[i]]
            a = a[:, featureSample]

            tree.fit(a, y[rowSample[i]], maxlvl + 1, featureNames=featureSample)
            self.trees.append(tree)

    def predictionMatrix(self, x):
        return np.array([tree.predict(x) for tree in self.trees])

    def predict(self, x):
        predictionmatrix = self.predictionMatrix(x)
        return [np.argmax(np.bincount(predictions)) for predictions in predictionmatrix.T]

    def getPredictions(self, x):
        return [tree.predictWithWeights(x) for tree in self.trees]