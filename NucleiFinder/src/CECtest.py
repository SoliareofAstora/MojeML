import os
import pandas as pd
from glob import glob
import numpy as np
from skimage.io import imread

datadir='input\\'

datatypes = pd.read_csv(os.path.join(datadir,'classes.csv'))

trainPath = datadir+'stage1_train'
traindata = next(os.walk(trainPath))
testPath = datadir + 'stage1_test'
testdata = next(os.walk(testPath))




imageDF= pd.DataFrame({'path': glob(os.path.join(datadir, 'stage1_*', '*', '*', '*'))})
img_id = lambda in_path: in_path.split('\\')[-3]
img_type = lambda in_path: in_path.split('\\')[-2]

imageDF['ImageId'] = imageDF['path'].map(img_id)
imageDF['ImageType'] = imageDF['path'].map(img_type)



arr = np.array([[1,1,20,70,70,70],
               [1,20,20,20,20,70]])

a = np.where(arr==1,1,0)
b = np.where(arr==20,1,0)
c = np.where(arr==70,1,0)
