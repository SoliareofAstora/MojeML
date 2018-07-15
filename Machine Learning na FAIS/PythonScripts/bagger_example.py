from PythonScripts.dtree import dtree
from PythonScripts.bagger import bagger
import pandas as pd
import numpy as np

tree = dtree()
bag = bagger()

data, classes, features = tree.read_data('input/car.data')

train = data[::2][:]
test = data[1::2][:]
trainc = classes[::2]
testc = classes[1::2]

# t = tree.make_tree(train, trainc, features)
# out = tree.classifyAll(t, test)
# #tree.printTree(t, ' ')
#
# a = np.zeros(len(out))
# b = np.zeros(len(out))
# d = np.zeros(len(out))
#
# for i in range(len(out)):
#     if testc[i] == 'good' or testc[i] == 'v-good':
#         b[i] = 1
#         if out[i] == testc[i]:
#             d[i] = 1
#     if out[i] == testc[i]:
#         a[i] = 1
# print("Tree")
# print("Number correctly predicted", str(np.sum(a)))
# print("Number of testpoints ", str(len(a)))
# print("Percentage Accuracy ", str(np.sum(a) / len(a) * 100.0))
# print("")
# print("Number of cars rated as good or very good", str(np.sum(b)))
# print("Number correctly identified as good or very good", str(np.sum(d)))
# print("Percentage Accuracy", str(np.sum(d) / np.sum(b) * 100.0))

##############################
#
c = bag.bag(train, trainc, features, 100)
out = bag.bagclass(c, test)

a = np.zeros(len(out))
b = np.zeros(len(out))
d = np.zeros(len(out))

for i in range(len(out)):
    if testc[i] == 'good' or testc[i] == 'v-good':
        b[i] = 1
        if out[i] == testc[i]:
            d[i] = 1
    if out[i] == testc[i]:
        a[i] = 1
print("-----")
print("Bagger")
print("Number correctly predicted", str(np.sum(a)))
print("Number of testpoints ", str(len(a)))
print("Percentage Accuracy ", str(np.sum(a) / len(a) * 100.0))

print("Number of cars rated as good or very good", str(np.sum(b)))
print("Number correctly identified as good or very good", str(np.sum(d)))
print("Percentage Accuracy", str(np.sum(d) / np.sum(b) * 100.0))

##############################

from PythonScripts.randomForest import randomForrest
forest = randomForrest()

c = forest.train(train, trainc, features, 100)
out = forest.predict(c, test)

a = np.zeros(len(out))
b = np.zeros(len(out))
d = np.zeros(len(out))

for i in range(len(out)):
    if testc[i] == 'good' or testc[i] == 'v-good':
        b[i] = 1
        if out[i] == testc[i]:
            d[i] = 1
    if out[i] == testc[i]:
        a[i] = 1
print("-----")
print("RandomForest")
print("Number correctly predicted", str(np.sum(a)))
print("Number of testpoints ", str(len(a)))
print("Percentage Accuracy ", str(np.sum(a) / len(a) * 100.0))

print("Number of cars rated as good or very good", str(np.sum(b)))
print("Number correctly identified as good or very good", str(np.sum(d)))
print("Percentage Accuracy", str(np.sum(d) / np.sum(b) * 100.0))






