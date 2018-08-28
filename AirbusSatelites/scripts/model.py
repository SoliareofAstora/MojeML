from keras.models import Input, Model
from keras.layers import concatenate
from keras.layers import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout

dropoutRate = 0.25

def convBlock(x,kernels):
    conv = Conv2D(kernels, (3, 3), padding="same")(x)
    activated = Activation("relu")(conv)
    normalised = BatchNormalization(axis=-1)(activated)
    return normalised

def buildSmall():
    x = Input(shape=(768, 768, 3))
    conv1 = convBlock(x,32)
    output = MaxPooling2D(pool_size=(3,3))(conv1)
    output = Dropout(dropoutRate)(output)

    conv2 = convBlock(output,64)
    output = MaxPooling2D(pool_size=(2,2))(conv2)
    output = Dropout(dropoutRate)(output)

    conv3 = convBlock(output,128)
    output = MaxPooling2D(pool_size=(2,2))(conv3)
    output = Dropout(dropoutRate)(output)

    conv4 = convBlock(output,256)
    output = MaxPooling2D(pool_size=(2,2))(conv4)
    output = Dropout(dropoutRate)(output)

    output = convBlock(output,512)

    output = convBlock(UpSampling2D((2, 2))(output),256)
    output = concatenate([output, conv4])

    output = convBlock(UpSampling2D((2, 2))(output),128)
    output = concatenate([output, conv3])

    output = convBlock(UpSampling2D((2, 2))(output),64)
    output = concatenate([output, conv2])

    output = convBlock(UpSampling2D((3,3))(output),32)
    output = concatenate([output, conv1])

    output = Conv2D(1, (1, 1), padding="same")(output)
    output = Activation("hard_sigmoid")(output)

    model = Model(input=x, output=output)
    print(model.summary())
    return model


def build():

    x = Input(shape=(768, 768, 3))
    # 768
    output = Conv2D(64, (3, 3), padding="same")(x)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    c1 = Conv2D(64, (3, 3), padding="same")(output)
    output = Activation("relu")(c1)
    output = BatchNormalization(axis=-1)(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Dropout(dropoutRate)(output)

    # 384
    output = Conv2D(128, (3, 3), padding="same")(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    c2 = Conv2D(128, (3, 3), padding="same")(output)
    output = Activation("relu")(c2)
    output = BatchNormalization(axis=-1)(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Dropout(dropoutRate)(output)

    # 192
    output = Conv2D(256, (3, 3), padding="same")(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    c3 = Conv2D(256, (3, 3), padding="same")(output)
    output = Activation("relu")(c3)
    output = BatchNormalization(axis=-1)(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Dropout(dropoutRate)(output)

    # 96
    output = Conv2D(512, (3, 3), padding="same")(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    c4 = Conv2D(512, (3, 3), padding="same")(output)
    output = Activation("relu")(c4)
    output = BatchNormalization(axis=-1)(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Dropout(dropoutRate)(output)

    # 48
    output = Conv2D(1024, (3, 3), padding="same")(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    output = Conv2D(1024, (3, 3), padding="same")(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)

    # now lets go upstream
    output = Conv2D(512, (3, 3), padding="same")(UpSampling2D((2, 2))(output))

    # 96
    output = concatenate([output, c4])
    output = Dropout(dropoutRate)(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    output = Conv2D(512, (3, 3), padding="same")(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    output = Conv2D(512, (3, 3), padding="same")(output)
    output = Conv2D(256, (3, 3), padding="same")(UpSampling2D((2, 2))(output))

    # 192
    output = concatenate([output, c3])
    output = Dropout(dropoutRate)(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    output = Conv2D(256, (3, 3), padding="same")(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    output = Conv2D(256, (3, 3), padding="same")(output)
    output = Conv2D(126, (3, 3), padding="same")(UpSampling2D((2, 2))(output))

    # 384
    output = concatenate([output, c2])
    output = Dropout(dropoutRate)(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    output = Conv2D(126, (3, 3), padding="same")(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    output = Conv2D(126, (3, 3), padding="same")(output)
    output = Conv2D(64, (3, 3), padding="same")(UpSampling2D((2, 2))(output))

    # 768
    output = concatenate([output, c1])
    output = Dropout(dropoutRate)(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    output = Conv2D(4, (3, 3), padding="same")(output)
    output = Activation("relu")(output)
    output = BatchNormalization(axis=-1)(output)
    output = Conv2D(1, (1, 1), padding="same")(output)
    output = Activation("hard_sigmoid")(output)

    model = Model(input=x, output=output)
    return model
