
#
# x = np.array([1,2,3,7,8,9])
# K = 2
#
# class PCM:
#     centroids = np.array([])
#     assignation_matrix = np.array([])
#
#     def __init__(self, data_dimension, m=2, classes=1):
#         self.m = m
#         self.classes = classes
#         self.centroids = np.random.rand(classes,data_dimension)
#
#     def D(self, i, k):
#         return np.sqrt(np.dot(i - k, i - k))
#
#     def u(self, i, k):
#         return (1 + (self.D(i, k) / self.eta(i)) ** (2 / self.m - 1)) ** -1
#
#     def eta(self, i):
#         return (().sum() / ().sum())**self.m
#
#     def fit(self,x_train):
#         self.assignation_matrix=0
#
#     def predict(self,x_test):
#         return 0
#
# def eta(u, m, d):
#     um = u**m
#     numerator = np.sum(np.dot(um, d**2))
#     return numerator/np.sum(um)
#


import numpy as np

x = np.array([(4,1940),(9,2960),(9,4630),(78,1528),(90,2040),(50,3700),(467,14815),(509,15200),(290,15700),(215,6045)])
train_data = np.array(x)
max_values = train_data.max(0)
X_norm = np.divide(train_data,max_values)
x=np.copy(X_norm)

centroids =np.random.rand(2,2)
assignation_matrix = np.random.rand(centroids.shape[0],x.shape[0])
m=3

def new_assignation(data,centr,old_assignation):
    dist2 = np.zeros((centr.shape[0],data.shape[0]))
    for i in range(centr.shape[0]):
        dist2[i] = ((data-centr[i])**2).sum(axis=1)
    eta = ((old_assignation**m)*dist2).sum(axis=0)/old_assignation.sum(axis=0)
    return (1+(np.sqrt(dist2)/eta)**2./(m-1))**(-1)



def new_centroids(data,centr,assignation):
    new_cen = np.empty_like(centr)
    for i in range(centr.shape[0]):
        u_scalar = assignation[i].sum()
        u_vector = (assignation[i]**m * data.sum(axis=1)).sum(axis=0)
        new_cen[i] = u_vector/u_scalar
    return new_cen

n_assignation_matrix=new_assignation(x,centroids,assignation_matrix)

n_centroids = new_centroids(x,centroids,assignation_matrix)

centroids = np.copy(n_centroids)
assignation_matrix = np.copy(n_assignation_matrix)




