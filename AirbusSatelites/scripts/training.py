import numpy as np
import pandas as pd
from scripts import rle
from skimage import io,img_as_uint
import matplotlib.pyplot as plt
from os import listdir
import scripts.model as cnn
from scripts.rle import decode
from keras.optimizers import *
import os
import random
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

trainPath = "input/train/"
imageIDs = listdir(trainPath)
segments = pd.read_csv("input/segmentation.csv")
shape = (768, 768)
batchSize = 100
nepoch = 10


def getImage(imageID):
    return np.array([io.imread(trainPath+x)for x in imageID])

def getMask(imageID):
    output = []
    dec = lambda x: decode(x, shape)
    for a in imageID:
        masksRLE = segments[segments["ImageId"] == a]["EncodedPixels"].values
        mask = np.zeros((768, 768))
        if str(masksRLE[0]) != 'nan':
            temp = map(dec, masksRLE)
            masks = temp.__iter__()
            for m in masks:
                mask += m
        output.append(mask)
    return np.array(output)

def getSomeImages(batch):
    return getImage(imageIDs[batch*100:(batch+1)*100]),getMask(imageIDs[batch*100:(batch+1)*100])



datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True
)

model = cnn.build()
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['binary_accuracy'])
model.summary()


for e in range(nepoch):
    print("epoch %d" % e)
    for a in range(int(imageIDs.__len__()/100)-1):
        xtrain, ytrain = getSomeImages(a)
        model.fit(xtrain,np.expand_dims(ytrain,axis=-1),batch_size=10)










