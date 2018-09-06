import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('input/breast-cancer.txt')
data[data['bare_nuclei']=='?'] = 0
arr = np.array(data,dtype=np.float32)

for i in range(arr.shape[1]):
    plt.hist(arr[:,i])
    plt.savefig('out/'+str(i))
    plt.show()

