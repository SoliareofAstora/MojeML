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



