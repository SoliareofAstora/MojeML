import numpy as np
import pandas as pd
from skimage import io,img_as_uint
import matplotlib.pyplot as plt
from os import listdir
import cv2
import scripts.model as cnn
from scripts.rle import decode
from keras.optimizers import *
import os
import random
import io
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator




trainPath = "input/train/"
trainImages = pd.DataFrame({'ImageId':listdir(trainPath)})
segments = pd.read_csv("input/segmentation.csv")
shape = (768, 768)
batchSize = 100
nepoch = 10

# A.merge(B, left_on='lkey', right_on='rkey', how='outer')

shipsCounted = segments[segments["EncodedPixels"].isna()==False].groupby("ImageId").count()

trainImages = trainImages.merge(shipsCounted,right_index= True,left_on='ImageId',how='outer')
trainImages.rename(index = str,columns = {'EncodedPixels':"shipCount"},inplace=True)
trainImages.loc[trainImages['shipCount'].isna(),['shipCount']] = 0
abc = trainImages

import matplotlib.pyplot as plt
bars = trainImages.groupby('shipCount').count()

plt.figure(figsize=(10,4))
plt.subplot(131)
plt.bar(bars.index.values,bars["ImageId"])
plt.xlabel("Amount of ships per image")
plt.ylabel("Amount of images")

plt.subplot(132)
plt.yticks([])
plt.xticks([0,1],['no ship','ships'])
plt.bar([0,1],[bars['ImageId'][0],bars['ImageId'][1:].sum()])

plt.subplot(133)
plt.xticks([0,1],['1 ship','2 or more'])
plt.yticks([])
plt.bar([0,1],[bars['ImageId'][1],bars['ImageId'][2:].sum()])
plt.show()


from skimage import io


from skimage.morphology import label
def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks



arr = np.array(
    [[0,0,0.2,0.5],
     [0, 0, 0.2, 0.5],
     [0, 1, 0.2, 0.5],
     [0.1, 0.7, 0.2, 0.5]]
)

arr>0.4
from skimage import morphology
from skimage import measure

measure.label(arr>0.4)





import pickle


trainImages['shipCount'].sum()

imagesWithShip = trainImages[trainImages['shipCount']>0].shape
id = '00021ddc3.jpg'
trainImages.drop(['6384c3e78.jpg'],inplace = True)

trainImages[trainImages['ImageId']=='6384c3e78.jpg']

sum = 0
ships = []
for id in trainImages[trainImages['shipCount']>0]['ImageId'].sample(1):
    image = io.imread(trainPath+id)
    masks = segments[segments['ImageId']==id]['EncodedPixels']
    for i in masks:
        m = rle_decode(i)
        coords = np.argwhere(m)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        cropped = image[x0:x1, y0:y1]
        m = m[x0:x1, y0:y1]
        ships.append(cv2.bitwise_and(cropped,cropped, mask = m))


a=0
for s in ships:
    a+=1
    plt.show()
    io.imsave('input/ships/'+str(a)+".png",s)




m = rle_decode(i)

coords = np.argwhere(m)
x0, y0 = coords.min(axis=0)
x1, y1 = coords.max(axis=0) + 1
cropped = image[x0:x1, y0:y1]
m = m[x0:x1, y0:y1]




io.imshow(image)
plt.show()
io.imshow(masks_as_color(masks))
plt.show()


trainImages = abc
for i in range(1,5,1):
    trainImages = pd.concat([trainImages,trainImages[trainImages['shipCount']>i]],ignore_index=True)


bars = trainImages.groupby('shipCount').count()

plt.figure(figsize=(10,4))
plt.subplot(131)
plt.bar(bars.index.values,bars["ImageId"])
plt.xlabel("Amount of ships per image")
plt.ylabel("Amount of images")

plt.subplot(132)
plt.yticks([])
plt.xticks([0,1],['no ship','ships'])
plt.bar([0,1],[bars['ImageId'][0],bars['ImageId'][1:].sum()])

plt.subplot(133)
plt.xticks([0,1],['1 ship','2 or more'])
plt.yticks([])
plt.bar([0,1],[bars['ImageId'][1],bars['ImageId'][2:].sum()])
plt.show()


trainImages.drop(trainImages[trainImages['ImageId'] == '6384c3e78.jpg'].index,inplace = True) #remove corrupted image


valImages = trainImages['ImageId'].sample(int(trainImages.size/100))
trainImages.drop(valImages.index,inplace=True)

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

getImage(valImages)












