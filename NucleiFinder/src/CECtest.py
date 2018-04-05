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



import pandas as pd
import numpy as np
import random
from skimage.io import imread,imread_collection
import matplotlib.pyplot as plt
from skimage import transform
import tensorflow as tf
import time

random.seed(time.clock())


def readTrainImage(image_id):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel
    # by 'William Cukierski'
    image_file = "input/stage1_train/{}/images/{}.png".format(image_id,image_id)
    mask_file = "input/stage1_train/{}/masks/*.png".format(image_id)
    image = imread(image_file)
    masks = imread_collection(mask_file).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1
    return image, labels



test_id = "3bfa8b3b01fd24a28477f103063d17368a7398b27331e020f3a0ef59bf68c940"#"4596961c789d3b41916492918797724fe75128239fefc516c3ee75322b7926f0"
image_id=test_id
image_file = "input/stage1_train/{}/images/{}.png".format(image_id, image_id)
mask_file = "input/stage1_train/{}/masks/*.png".format(image_id)
image = imread(image_file)
masks = imread_collection(mask_file).concatenate()
masks.shape
image.shape

mask = np.swapaxes(masks,0,2)
mask = np.swapaxes(mask,0,1)
mask.shape
mask.dtype = bool



# def data_aug(image,label):
#     flipx = random.randint(0, 1)
#     flipy = random.randint(0, 1)
#     resize_rate = random.randint(10,100)/100.0
#     size = image.shape[0]
#     rsize = random.randint(np.floor(resize_rate*size),size)
#     w_s = random.randint(0,size - rsize)
#     h_s = random.randint(0,size - rsize)
#     sh = random.random()/2-0.25
#     angle=random.randint(1,89)
#     rotate_angel = random.random()/180*np.pi*angle
#     # Create Afine transform
#     afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
#     # Apply transform to image data
#     image = transform.warp(image, inverse_map=afine_tf,mode='constant')
#     label = transform.warp(label, inverse_map=afine_tf,mode='constant')
#     # Randomly corpping image frame
#     image = image[w_s:w_s+size,h_s:h_s+size,:]
#     label = label[w_s:w_s+size,h_s:h_s+size]
#     # Ramdomly flip frame
#
#     # Ramdomly flip frame
#     if flipx:
#         image = image[:,::-1,:]
#         label = label[:,::-1]
#     if flipy:
#         image = image[::-1,:,:]
#         label = label[::-1,:]
#     return image, label

train_labels = pd.read_csv('input/stage1_train_labels.csv')

test_id = "4596961c789d3b41916492918797724fe75128239fefc516c3ee75322b7926f0"
image_id=test_id
image,label = readTrainImage(test_id)

# Plot predictions of each model
plt.imshow(image)
plt.imshow(label,alpha=0.3)
plt.show()


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


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height//4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y-s, x-s, y+s, x+s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(10, 200, 100)
dataset_train.prepare()


import src.visualize as visualize

# Load and display random samples
image_ids = dataset_train.image_ids
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

imageid=image_ids[1]
mask, class_ids = dataset_train.load_mask(imageid)
image = dataset_train.load_image(imageid)
visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

np.arange(10)