"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, load_size, fine_size, num_images=2, flip=True, is_test=False):
    images = load_image(image_path, num_images)
    images = preprocess_images(images, load_size=load_size, fine_size=fine_size, flip=flip, is_test=is_test)
    images = [img/127.5 - 1. for img in images]

    ret = np.concatenate(images, axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return ret

def load_image(image_path, num_images=2):
    input_img = imread(image_path)
    full_width = int(input_img.shape[1])

    if full_width % num_images != 0:
        raise Exception("invalid dimention. width must be divisible by {},"
                        " but input_img.shape={}".format(num_images, input_img.shape))
    img_width = int(full_width/num_images)

    return [input_img[:, img_width*x:img_width*(x+1)] for x in range(num_images)]


def preprocess_images(images, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        ret = [scipy.misc.imresize(img, [fine_size, fine_size]) for img in images]
    else:
        ret = [scipy.misc.imresize(img, [load_size, load_size]) for img in images]

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        ret = [img[h1:h1+fine_size, w1:w1+fine_size] for img in ret]

        if flip and np.random.random() > 0.5:
            ret = [np.fliplr(img) for img in ret]

    return ret

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


