import numpy as np

class decisionTree:

    def classify(self, datapoint):
        try:
            return self.pred
        except:
            try:
                return self.tree[0].classify(datapoint) if datapoint[self.FeatureName] < self.SplitValue \
                    else self.tree[1].classify(datapoint)
            except:
                return self.default

    def predictWithWeights(self, X):
        return [self.classify(x) for x in X]

    def predict(self, X):
        return [self.classify(x)[1] for x in X]

    def fit(self, X, Y, maxLevel=-1, level=0, featureNames=None):

        nData = len(X)

        if nData == 0:
            return

        if featureNames is not None:
            self.featureNames = featureNames
        else:
            self.featureNames = np.arange(len(X[0]))

        classes, frequency = np.unique(Y, return_counts=True)
        self.default = [np.max(frequency) / frequency.sum(), classes[np.argmax(frequency)]]

        if len(classes) == 1:
            self.pred = [1., classes[0]]
            return

        if (maxLevel >= 0 and level > maxLevel):
            self.pred = self.default
            return

        giniMatrix = self.calcGini(X, Y)

        self.FeatureName = 0
        minGinis = [np.min(g) for g in giniMatrix]
        bestFeatureIndex = np.argmin(minGinis)
        self.FeatureName = self.featureNames[bestFeatureIndex]
        self.SplitValue = np.unique(X[:, bestFeatureIndex])[np.argmin(giniMatrix[bestFeatureIndex])]

        binarySplit = X[:, bestFeatureIndex] < self.SplitValue

        treea = decisionTree()
        treeb = decisionTree()
        treea.fit(X[binarySplit], Y[binarySplit], maxLevel, level + 1, self.featureNames)
        treeb.fit(X[~binarySplit], Y[~binarySplit], maxLevel, level + 1, self.featureNames)

        self.tree = (
            treea,
            treeb
        )

    def calcGini(self, x, y):
        ndata = len(x)
        ginis = []
        for column in x.T:
            columnGini = []
            values, valuesCount = np.unique(column, return_counts=True)

            for value in values:
                split = (column < value)
                groups = [
                    y[split],
                    y[~split]
                ]
                gini = 0
                for group in groups:
                    classes, classesCount = np.unique(group, return_counts=True)
                    if len(classes) == 0:
                        continue
                    score = np.sum((classesCount / np.sum(classesCount)) ** 2)
                    gini += (1 - score) * (np.sum(classesCount) / ndata)
                columnGini.append(gini)
            ginis.append(columnGini)
        return ginis


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


class boostedForrest:
    def fit(self, x, y, nepoch=4, nTrees=30, maxLevel=7, bagVolume=50):
        trees = bagger()
        trees.fit(x, y, nTrees, maxLevel, bagVolume)

        sampleWeights = np.ones_like(y) / len(y)
        treeWeights = np.ones_like(nTrees)
        predictionMatrix = (trees.predictionMatrix(x))
        predictionMatrix[predictionMatrix == 0] = -1

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
