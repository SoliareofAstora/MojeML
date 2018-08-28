import os
from os import listdir
import numpy as np
import pandas as pd
from skimage import io
from scripts.rle import decode


def nothing():
    trainPath = "input/train/"
    imageIDs = listdir(trainPath)
    segments = pd.read_csv("input/segmentation.csv")

    shape = (768,768)
    dec = lambda x:decode(x,shape)

    imageIDs[0][0:-4]

    os.mkdir("input/segmentationMask")
    os.mkdir("input/instanceMasks")

    for i in range(imageIDs.__sizeof__()):
        if i%100==0:
            print(str((i/imageIDs.__sizeof__())*100)+'%')
        masks = segments[segments["ImageId"] == imageIDs[i]]["EncodedPixels"].values
        mask = np.zeros((768,768))
        os.mkdir("input/instanceMasks/"+imageIDs[i][0:-4])
        io.imsave('input/instanceMasks/' + imageIDs[i][0:-4]+"/0_"+imageIDs[i], mask)
        if str(masks[0])!='nan':
            temp = map(dec,masks)
            masks = temp.__iter__()
            a = 0
            for m in masks:
                mask += m
                io.imsave('input/instanceMasks/' + imageIDs[i][0:-4] + "/" + str(a)+"_" + imageIDs[i], mask)
                a+=1

        io.imsave('input/segmentationMask/'+imageIDs[i],mask)


