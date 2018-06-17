import numpy as np

def f(x):
    return x >= 0

class MyPerceptron:

    def __init__(self,activation = f):
        self.activation = activation
        self.coef = np.array([])
        self.intercept = 0

    def fit(self,x,y,learningRate = 0.01,epoch = 10):
        self.coef = np.zeros(x.shape[1],dtype=float)
        for e in range(epoch):
            for i in range(x.shape[0]):
                pred = self.singlePredict(x[i])
                error = y[i] - pred
                self.coef += learningRate*error*x[i]
                self.intercept += learningRate * error

    def singlePredict(self,x):
        return self.activation((x * self.coef).sum() + self.intercept)

    def predict(self,x):
        return self.activation((x*self.coef).sum(axis=1)+self.intercept)



