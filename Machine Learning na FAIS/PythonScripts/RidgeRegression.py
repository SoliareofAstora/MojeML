import numpy as np

def gradient_descent(b, x, y, lr):
    batch = y.size
    b[0] = b[0] - lr * (((x.dot(b[1:]) + b[0]) - y).sum())
    b[1:] = b[1:] - lr * ((x.dot(b[1:]) + b[0]) - y).sum() * x.sum(axis=0) / batch


def gradient_descent_for_ridge(b, x, y, lr,alpha):
    batch = y.size
    b[0] = b[0] - lr * (((x.dot(b[1:]) + b[0]) - y).sum()+alpha*b[0]*2)
    b[1:] = b[1:] - lr * (((x.dot(b[1:]) + b[0]) - y).sum() * x.sum(axis=0)+alpha*b[1:]*2) / batch


def R2(b,x,y):
    rss = (((x.dot(b[1:]) + b[0]) - y)**2).sum()
    tss = ((y-y.mean())**2).sum()
    return (tss-rss)/tss


class Ridge_Regression():
    alpha = 1
    lr = 0.001
    beta = np.array([])

    def __init__(self, alpha=1, learning_rate=0.001):
        self.alpha = alpha
        self.lr = learning_rate

    def fit(self, xtrain, ytrain, batchsize=100,epoch=10000):
        self.beta = np.ones(xtrain.shape[1] + 1)
        old = np.ones_like(self.beta)
        new = np.zeros_like(self.beta)
        batch = 0
        if  ytrain.size<batchsize:
            batchsize = ytrain.size
        epoch += int(ytrain.size/batchsize)
        for a in range(epoch):
            if batchsize * (batch + 1) > ytrain.size:
                batch = 0
                perm = np.random.permutation(ytrain.size)
                xtrain=xtrain[perm]
                ytrain=ytrain[perm]
            batch_range = np.arange(batchsize * batch, batchsize * (batch + 1))
            gradient_descent_for_ridge(self.beta, xtrain[batch_range], ytrain[batch_range], self.lr,self.alpha)
            batch += 1

    def predict(self, xtest):
        return xtest.dot(self.beta[1:]) + self.beta[0]

    def R2_score(self,x,y):
        return R2(self.beta,x,y)
