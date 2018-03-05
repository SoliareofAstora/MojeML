import numpy as np #you can remove numpy
arr = np.zeros((7,3))
for x in range(7):
    arr[x,:] = [1 + x, 9 - x, 2 ** (2 + x)]


# x = 10
# a = [2 + i * 2 for i in range(x)]
# b = [8 - i * 2 for i in range(x)]
# c = [1 / (i + 1) for i in range(x)]
# d = [i/(i+1) for i in range(x)]

arr[[4,6],:].sum()

arr[:,2]

temp = np.copy(arr[3])
arr[3]= arr[2]
arr[2]=temp

arr = np.zeros((3,5))
for x in range(5):
    arr[:,x]=x+1


arr = np.zeros((6,3))
for x in range(6):
    arr[x,:]=x