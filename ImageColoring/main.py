import matplotlib.pyplot as plt
import numpy as np
import cv2
from os import listdir

trainPath = 'input/256/flowerpower/'
ids = listdir('input/256/flowerpower/')


def getImages(imageID):
    return np.array([cv2.imread(trainPath + x) for x in imageID])

def getGrayImages(images):
    return np.array([np.array(cv2.cvtColor(BGR,cv2.COLOR_BGR2GRAY),dtype=float)/255 for BGR in images])

def getYUVImages(images):
    return np.array([np.array(cv2.cvtColor(BGR,cv2.COLOR_BGR2YUV),dtype=float)/255 for BGR in images])

def getRGB(images):
    try:
        return np.array([cv2.cvtColor(BGR,cv2.COLOR_YUV2RGB) for BGR in images])
    except:
        return cv2.cvtColor(images,cv2.COLOR_YUV2RGB)


allBGR = getImages(ids)
allYUV = getYUVImages(allBGR)
allGray = getGrayImages(allBGR)


# plt.imshow(getRGB(np.array(allYUV[0]*255,dtype=np.uint8)))
# plt.show()


from keras.models import Input, Model
from keras.layers import concatenate
from keras.layers import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout

dropoutRate = 0.001

def convBlock(x,kernels):
    conv = Conv2D(kernels, (3, 3), padding="same")(x)
    activated = Activation("relu")(conv)
    normalised = BatchNormalization(axis=-1)(activated)
    dropout = Dropout(dropoutRate)(normalised)
    return dropout


def buildModel():
    x = Input(shape=(256, 256,1))
    conv1 = convBlock(x, 4)
    output = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = convBlock(output, 8)
    output = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = convBlock(output, 16)
    output = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = convBlock(output, 32)
    output = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = convBlock(output, 64)
    output = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = convBlock(output, 64)
    output = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = convBlock(output, 128)
    output = MaxPooling2D(pool_size=(2, 2))(conv7)

    output = convBlock(output, 128)

    output = convBlock(UpSampling2D((2, 2))(output), 128)
    output = concatenate([output, conv7])

    output = convBlock(UpSampling2D((2, 2))(output), 128)
    output = concatenate([output, conv6])

    output = convBlock(UpSampling2D((2, 2))(output), 128)
    output = concatenate([output, conv5])

    output = convBlock(UpSampling2D((2, 2))(output), 64)
    output = concatenate([output, conv4])

    output = convBlock(UpSampling2D((2, 2))(output), 32)
    output = concatenate([output, conv3])

    output = convBlock(UpSampling2D((2, 2))(output), 18)
    output = concatenate([output, conv2])

    output = convBlock(UpSampling2D((2,2))(output), 16)
    output = concatenate([output, conv1])

    output = Conv2D(2, (1,1), padding="same")(output)

    output = Activation("sigmoid")(output)
    output = concatenate([x,output])

    model = Model(input=x, output=output)
    print(model.summary())
    return model


model = buildModel()
from keras.optimizers import Adam
model.compile(optimizer = Adam(lr=1e-1,decay = 0.05),
                   loss = 'mean_squared_error',
                   metrics = ['categorical_accuracy'])


history = model.fit(np.expand_dims(allGray,axis=-1),allYUV,10,10)

plt.imshow(getRGB(np.array(allYUV[0]*255,dtype=np.uint8)))
plt.show()

plt.imshow(getRGB(np.array(model.predict(np.expand_dims(allGray[0:5],axis=-1))[3]*255,dtype=np.uint8)))
plt.show()