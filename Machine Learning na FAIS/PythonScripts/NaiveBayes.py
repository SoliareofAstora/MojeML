import numpy as np


class NaiveBayes:
    mean = np.array([])
    variance = np.array([])
    classes = np.array([])

    def fit(self, x, y):
        classes = np.unique(y)
        mean = np.zeros((len(classes), x.shape[1]))
        variance = np.zeros((len(classes), x.shape[1]))

        for c in range(len(classes)):
            vector = x[np.where(y == classes[c])]
            mean[c] = vector.sum(axis=0) / vector.shape[0]
            variance[c] = (pow(mean[c] - vector, 2)).sum(axis=0) / vector.shape[0]
        self.classes = classes
        self.mean = mean
        self.variance = variance

    def predict(self, x):
        prediction = np.zeros((len(self.classes), x.shape[0]))
        for i in range(x.shape[0]):
            for c in range(len(self.classes)):
                prediction[c, i] = ((1 / (np.sqrt(2 * np.pi * self.variance[c]))) *
                                    pow(np.e, -(pow(x[i] - self.mean[c], 2) / (2 * self.variance[c])))).sum()
        final = self.classes[np.argmax(prediction, axis=0)]
        return final
