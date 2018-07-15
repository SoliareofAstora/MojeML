import numpy as np
from PythonScripts.dtree import dtree

class randomForrest:
    def __init__(self):
        self.tree = dtree()

    def train(self, data, targets, features, nSamples):

        nPoints =np.shape(data)[0]
        nDim = np.shape(data)[1]
        self.nSamples = nSamples

        # Compute bootstrap samples
        samplePoints = np.random.randint(0, nPoints, (nPoints, nSamples))
        classifiers = []

        for i in range(nSamples):
            sample = []
            sampleTarget = []
            FeatureA = np.random.randint(0,nDim-2)
            FeatureB = np.random.randint(FeatureA+1,nDim)
            for j in range(nPoints):
                sample.append(data[samplePoints[j, i]][FeatureA:FeatureB])
                sampleTarget.append(targets[samplePoints[j, i]])
            # Train classifiers
            classifiers.append(self.tree.make_tree(sample, sampleTarget, features[FeatureA:FeatureB], 5))

        return classifiers

    def predict(self, classifiers, data):

        decision = []
        # Majority voting
        for j in range(len(data)):
            outputs = []
            # print data[j]
            for i in range(self.nSamples):
                out = self.tree.classify(classifiers[i], data[j])
                if out is not None:
                    outputs.append(out)
            # List the possible outputs
            out = []
            for each in outputs:
                if out.count(each) == 0:
                    out.append(each)
            frequency = np.zeros(len(out))

            index = 0
            if len(out) > 0:
                for each in out:
                    frequency[index] = outputs.count(each)
                    index += 1
                decision.append(out[frequency.argmax()])
            else:
                decision.append(None)
        return decision
