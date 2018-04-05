import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from src.config import Config
import src.utils
import src.model as modellib
from src.model import log
import src.utils as utils
from skimage.io import imread,imread_collection


class NucleiData(utils.Dataset):
    def uploadImages(self,ImagesID):
        self.add_class("Nuclei",1,"Nuclei")
        for imageID in imageIDs:
            impath = "input/stage1_train/{}/images/{}.png".format(imageID, imageID)
            self.add_image('Nuclei', imageID, impath)

    def load_mask(self, image_id):
        path = "input/stage1_train/{}/masks/*.png".format(self.image_info[image_id]['id'])
        mask=imread_collection(path).concatenate()
        mask = np.swapaxes(mask,2,0)
        mask = np.swapaxes(mask,1,0)
        class_IDS = np.ones([mask.shape[2]])
        return mask.astype(np.bool),class_IDS.astype(np.int32)


class NucleuiConfig(Config):
    NAME = "Nuclei"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1024

    TRAIN_ROIS_PER_IMAGE = 100

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

config = NucleuiConfig()
config.display()

train_labels = pd.read_csv('input/stage1_train_labels.csv')
imageIDs = np.array(train_labels['ImageId'])
trainsplit = int(imageIDs.size*0.8)


trainSet = NucleiData()
trainSet.uploadImages(imageIDs[:trainsplit])
trainSet.prepare()


validationSet=NucleiData()
validationSet.uploadImages(imageIDs[trainsplit:])
validationSet.prepare()


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "maskRCNN")
MODEL_DIR = 'maskRCNN'
model = modellib.MaskRCNN(mode="training",
                          config=config,
                          model_dir=MODEL_DIR
                          )



model.train(trainSet,validationSet,
            learning_rate=NucleuiConfig.LEARNING_RATE,
            epochs=2,layers='heads')
