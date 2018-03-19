import numpy as np
import matplotlib.pyplot as plt
# #
x=np.zeros(10000)
for i in range(x.size):
    x[i] = np.random.multivariate_normal([0],[[2]])

n, bins, patches = plt.hist(x, 200, density=True)
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Histogram of multivariate_normal 1D')
plt.plot(bins[:-1],n,)
plt.grid(True)
plt.show()
#
x=np.zeros((10000,2))
for i in range(x.shape[0]):
    x[i] = np.random.multivariate_normal([0,0],[[16,0],[0,4]])

plt.plot(x[:,0],x[:,1],".")
plt.grid(True)
plt.axis("equal")
plt.show()
#
# x=np.zeros((10000,2))
# for i in range(x.shape[0]):
#     x[i] = np.random.multivariate_normal([0,0],[[1,1],[1,1]])
#
# plt.plot(x[:,0],x[:,1],".")
# plt.grid(True)
# plt.axis("equal")
# plt.show()
#
x=np.zeros((10000,2))
for i in range(x.shape[0]):
    x[i] = np.random.multivariate_normal([0,0],[[10,5],[5,5]])

plt.axis("equal")
plt.hist2d(x[:,0],x[:,1],100)
plt.title('Heatmap of multivariate_normal 2D')
plt.show()

plt.plot(x[:,0],x[:,1],".")
plt.grid(True)
plt.axis("equal")
plt.show()




