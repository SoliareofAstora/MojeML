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
import src.visualize as visualize
import src.model as modellib
from src.model import log
import src.utils as utils
from skimage.io import imread, imread_collection


class NucleiData(utils.Dataset):
    def uploadImages(self, imagesIDs,train=True):
        self.add_class("Nuclei", 1, "Nuclei")
        if train:
            for imageID in imagesIDs:
                impath = "input/stage1_train/{}/images/{}.png".format(imageID, imageID)
                self.add_image('Nuclei', imageID, impath)
        else:
            for imageID in imagesIDs:
                impath = "input/stage1_test/{}/images/{}.png".format(imageID, imageID)
                self.add_image('Nuclei', imageID, impath)

    def load_mask(self, image_id):
        path = "input/stage1_train/{}/masks/*.png".format(self.image_info[image_id]['id'])
        mask = imread_collection(path).concatenate()
        mask = np.swapaxes(mask, 2, 0)
        mask = np.swapaxes(mask, 1, 0)
        class_IDS = np.ones([mask.shape[2]])
        return mask.astype(np.bool), class_IDS.astype(np.int32)


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
trainsplit = int(imageIDs.size * 0.8)

trainSet = NucleiData()
trainSet.uploadImages(imageIDs[:trainsplit])
trainSet.prepare()

validationSet = NucleiData()
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

model.train(trainSet, validationSet,
            learning_rate=NucleuiConfig.LEARNING_RATE,
            epochs=2, layers='heads')


model.build()
# uncomment to save
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


class InferenceConfig(NucleuiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# model_path = model.find_last()[1]
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


testlabels = pd.read_csv('input/stage1_sample_submission.csv')
testIDS = np.array(testlabels['ImageId'])
testset = NucleiData()
testset.uploadImages(testIDS,False)
testset.prepare()

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


result = pd.DataFrame([],columns=['ImageId','EncodedPixels'])

for i in range(testIDS.shape[0]):
    print(i)
    image = testset.load_image(i)
    image.shape
    rest = model.detect([image])
    res = rest[0]
    res['masks'].shape
    num_masks = int(res['masks'].shape[2])
    masks = res['masks']
    for a in range(num_masks):
        rle = str(rle_encoding(masks[:,:,a]))[1:-1]
        rle = rle.replace(",","")
        result = result.append({'ImageId':testset.image_info[i]['id'],"EncodedPixels":rle},ignore_index=True)

    if num_masks==0:
        result = result.append({'ImageId': testset.image_info[i]['id'], "EncodedPixels": '0 0'}, ignore_index=True)

result.to_csv('MyResult.csv',index=False)

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)
plt.imshow(original_image)
plt.show()
height, width, _ = original_image.shape
num_masks = int(gt_mask.shape[2])
labels = np.zeros((height, width), np.uint16)
for index in range(0, num_masks):
    labels[gt_mask[:, :, index] == True] = index + 1


image = original_image[image_meta[7]:image_meta[9],image_meta[8]:image_meta[10],:]
plt.imshow(image, alpha=0.5)
plt.show()

image_file = "input/stage1_train/{}/images/{}.png".format(image_id, image_id)


image = imread(validationSet.image_info[image_id]['path'])
plt.imshow(image)
plt.show()

test_id = "3bfa8b3b01fd24a28477f103063d17368a7398b27331e020f3a0ef59bf68c940"#"4596961c789d3b41916492918797724fe75128239fefc516c3ee75322b7926f0"
image_id=test_id
image_file = "input/stage1_train/{}/images/{}.png".format(image_id, image_id)
mask_file = "input/stage1_train/{}/masks/*.png".format(image_id)
image = imread(image_file)
result = model.detect(image)